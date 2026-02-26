"""
camera_tracker.py — ROS2 Publisher Node (OAK-D + MediaPipe)

Captures human arm motion from OAK-D camera using MediaPipe Pose + Hands.
Publishes joint positions directly as a ROS2 topic — no UDP needed.

Topic: /human/skeletal_data  (geometry_msgs/PoseArray)
  poses[0].position    → shoulder  (x, y, z)
  poses[1].position    → elbow     (x, y, z)
  poses[2].position    → hand/wrist (x, y, z)
  poses[2].orientation → hand quaternion (qx, qy, qz, qw)

To run:
  conda activate ros2_humble
  export ROS_DOMAIN_ID=0
  python3 camera_tracker.py

On Akshay's machine (same WiFi, same ROS_DOMAIN_ID=0):
  ros2 topic echo /human/skeletal_data
"""

import cv2
import depthai as dai
import mediapipe as mp
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String

ROBOT_UPPER_ARM = 0.20
ROBOT_FOREARM   = 0.265


def vec3(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def to_mujoco(mp_vec):
    """MediaPipe world → MuJoCo: X=same, Y=-Z_mp, Z=-Y_mp"""
    return np.array([mp_vec[0], -mp_vec[2], -mp_vec[1]], dtype=np.float32)


def rotation_matrix_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    return [float(qx), float(qy), float(qz), float(qw)]


def init_oakd():
    pipeline = dai.Pipeline()
    camRgb   = pipeline.create(dai.node.ColorCamera)
    xoutRgb  = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setFps(15)
    camRgb.setPreviewSize(640, 360)
    camRgb.preview.link(xoutRgb.input)
    return pipeline


class CameraTrackerNode(Node):
    def __init__(self):
        super().__init__('camera_tracker')

        # ── Publisher: matches /human/skeletal_data that ocra_node.py subscribes to ──
        self.pub       = self.create_publisher(PoseArray, '/human/skeletal_data', 10)
        self.pub_state = self.create_publisher(String, '/mocap/state', 10)

        self.get_logger().info(
            'Camera Tracker Node started — publishing on /human/skeletal_data'
        )

    def make_pose(self, xyz):
        """Pose with position only (identity orientation)."""
        p = Pose()
        p.position.x = float(xyz[0])
        p.position.y = float(xyz[1])
        p.position.z = float(xyz[2])
        p.orientation.w = 1.0
        return p

    def make_pose_with_quat(self, xyz, quat):
        """
        Pose with position + orientation.
        quat = [qx, qy, qz, qw]
        Used for poses[2] → hand position + hand quaternion.
        Matches what ocra_node.py reads from poses[2].orientation.
        """
        p = Pose()
        p.position.x    = float(xyz[0])
        p.position.y    = float(xyz[1])
        p.position.z    = float(xyz[2])
        p.orientation.x = float(quat[0])
        p.orientation.y = float(quat[1])
        p.orientation.z = float(quat[2])
        p.orientation.w = float(quat[3])
        return p

    def publish_joints(self, shoulder, elbow, hand, quat):
        """Publish PoseArray: shoulder, elbow, hand+quat."""
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'mocap_world'
        msg.poses = [
            self.make_pose(shoulder),              # poses[0] → shoulder
            self.make_pose(elbow),                 # poses[1] → elbow
            self.make_pose_with_quat(hand, quat),  # poses[2] → hand + quaternion
        ]
        self.pub.publish(msg)

    def publish_state(self, state_str):
        msg      = String()
        msg.data = state_str
        self.pub_state.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraTrackerNode()

    print("Initializing OAK-D Pipeline (RGB only, no depth)...")
    pipeline = init_oakd()

    mp_pose  = mp.solutions.pose
    pose     = mp_pose.Pose(
        static_image_mode=False, model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    state          = "CALIBRATION"
    CALIB_FRAMES   = 30
    calib_upper, calib_fore = [], []
    scale_upper, scale_fore  = 1.0, 1.0
    last_hand_quat = [0.0, 0.0, 0.0, 1.0]
    frame_count    = 0

    print("Stand in T-POSE (arms out to both sides) to calibrate.")

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)   # process ROS callbacks

            inRgb = qRgb.tryGet()
            if inRgb is None:
                continue

            cv_frame    = inRgb.getCvFrame()
            h, w        = cv_frame.shape[:2]
            frame_count += 1

            frame_rgb                 = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            pose_results              = pose.process(frame_rgb)
            hand_results              = None
            if frame_count % 2 == 0:
                hand_results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # ── Hand Orientation ──────────────────────────────────────────────
            if hand_results and hand_results.multi_hand_world_landmarks:
                hlm       = hand_results.multi_hand_world_landmarks[0].landmark
                wrist_pt  = vec3(hlm[mp_hands.HandLandmark.WRIST])
                index_mcp = vec3(hlm[mp_hands.HandLandmark.INDEX_FINGER_MCP])
                pinky_mcp = vec3(hlm[mp_hands.HandLandmark.PINKY_MCP])
                fwd   = index_mcp - wrist_pt
                side  = pinky_mcp - wrist_pt
                y_ax  = fwd  / (np.linalg.norm(fwd)  + 1e-8)
                z_tmp = np.cross(side, fwd)
                z_ax  = z_tmp / (np.linalg.norm(z_tmp) + 1e-8)
                x_ax  = np.cross(y_ax, z_ax)
                R              = np.column_stack([x_ax, y_ax, z_ax])
                last_hand_quat = rotation_matrix_to_quat(R)
                mp_drawing.draw_landmarks(
                    cv_frame,
                    hand_results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,200,200), thickness=1)
                )

            # ── Pose Joints ───────────────────────────────────────────────────
            if pose_results.pose_landmarks and pose_results.pose_world_landmarks:
                mp_drawing.draw_landmarks(
                    cv_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,100,0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(200,80,0), thickness=2)
                )

                wl = pose_results.pose_world_landmarks.landmark
                L  = mp_pose.PoseLandmark

                r_shoulder_mp = vec3(wl[L.RIGHT_SHOULDER.value])
                r_elbow_mp    = vec3(wl[L.RIGHT_ELBOW.value])
                r_wrist_mp    = vec3(wl[L.RIGHT_WRIST.value])

                pl          = pose_results.pose_landmarks.landmark
                r_wrist_vis = pl[L.RIGHT_WRIST.value].visibility
                r_elbow_vis = pl[L.RIGHT_ELBOW.value].visibility
                l_wrist_vis = pl[L.LEFT_WRIST.value].visibility

                # ═══ CALIBRATION ══════════════════════════════════════════════
                if state == "CALIBRATION":
                    node.publish_state("CALIBRATION")
                    shoulder_y = r_shoulder_mp[1]
                    elbow_y    = r_elbow_mp[1]
                    is_t_pose  = (
                        r_wrist_vis > 0.5 and r_elbow_vis > 0.5 and
                        l_wrist_vis > 0.5 and abs(elbow_y - shoulder_y) < 0.12
                    )
                    progress = int((len(calib_upper) / CALIB_FRAMES) * 100)

                    cv2.rectangle(cv_frame, (0, 0), (w, 120), (0, 0, 0), -1)
                    cv2.putText(cv_frame, "T-POSE  Stretch arms out to both sides",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    bar_w = w - 40
                    cv2.rectangle(cv_frame, (20, 55), (20 + bar_w, 80), (50, 50, 50), -1)
                    fill = int(bar_w * progress / 100)
                    cv2.rectangle(cv_frame, (20, 55), (20 + fill, 80), (0, 220, 0), -1)
                    cv2.putText(cv_frame, f"{progress}%", (20 + bar_w + 4, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

                    if is_t_pose:
                        cv2.putText(cv_frame, "Hold still...", (20, 108),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                        calib_upper.append(np.linalg.norm(r_elbow_mp - r_shoulder_mp))
                        calib_fore.append(np.linalg.norm(r_wrist_mp  - r_elbow_mp))
                        if len(calib_upper) >= CALIB_FRAMES:
                            ua          = float(np.median(calib_upper))
                            fa          = float(np.median(calib_fore))
                            scale_upper = ROBOT_UPPER_ARM / ua
                            scale_fore  = ROBOT_FOREARM   / fa
                            state       = "TRACKING"
                            node.get_logger().info(
                                f"[Calibrated] upper={ua*100:.1f}cm  scale={scale_upper:.3f} | "
                                f"fore={fa*100:.1f}cm  scale={scale_fore:.3f}"
                            )
                    else:
                        if calib_upper: calib_upper.pop()
                        if calib_fore:  calib_fore.pop()

                # ═══ TRACKING ═════════════════════════════════════════════════
                elif state == "TRACKING":
                    node.publish_state("TRACKING")
                    cv2.rectangle(cv_frame, (0, 0), (w, 52), (0, 0, 0), -1)
                    cv2.putText(cv_frame,
                                f"TRACKING  |  R=recal  scale_u:{scale_upper:.2f}  scale_f:{scale_fore:.2f}",
                                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                    # Convert to MuJoCo frame and publish directly as ROS2
                    shoulder = to_mujoco(r_shoulder_mp)
                    elbow    = to_mujoco(r_elbow_mp)
                    hand     = to_mujoco(r_wrist_mp)
                    node.publish_joints(shoulder, elbow, hand, last_hand_quat)

            cv2.imshow("Mocap Camera Tracker", cv_frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                state = "CALIBRATION"
                calib_upper.clear()
                calib_fore.clear()
                node.get_logger().info("Recalibrating...")

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
