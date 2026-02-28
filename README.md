# OCRA — Human-to-Robot Arm Motion Retargeting

Real-time teleoperation of a **ReactorX 200** robot arm by retargeting human arm motion captured via an **OAK-D depth camera** and **MediaPipe** pose estimation. Implements the OCRA algorithm (Optimization-based Customizable Retargeting Algorithm) from Mohan & Kuchenbecker, ICRA 2023.

---

## Overview

```
OAK-D Camera
    │  MediaPipe skeleton (shoulder, elbow, wrist)
    ▼
camera_tracker.py  ──►  /human/skeletal_data  (geometry_msgs/PoseArray)
                                  │
                                  ▼
                         ocra_sim_node.py  (or ocra_node.py for hardware)
                              │  OCRA loss minimized via SLSQP
                              │  JAX forward kinematics + gradients
                              ▼
                   /rx200/arm_controller/joint_trajectory
                              │
                              ▼
                    Gazebo sim  /  RX200 hardware
```

---

## Algorithm — OCRA

The OCRA loss function jointly minimizes two terms:

```
L(q) = α · ε_s² + β · ε_o²
```

**Skeleton error `ε_s`** — bidirectional chain-to-chain distance, normalized by total arm length:

```
ε_s = Σ s_i + Σ t_j   /   ℓ
```

where `s_i` = distance from human joint `i` to robot chain, `t_j` = distance from robot joint `j` to human chain, and `ℓ` = sum of arc lengths of both chains.

**Orientation error `ε_o`** — axis-angle magnitude of relative end-effector rotation:

```
Q_d  = q_robot · q_target⁻¹
ε_o  = 2·arctan2(|xyz|, |w|) / π       # stable at identity
```

The optimizer runs **SLSQP** (Sequential Least Squares Programming) with JAX-computed analytical gradients at 10 Hz.

---

## Repository Structure

```
robot_retarget/
├── rx200_kinematics.py     # JAX FK, OCRA loss, value_and_grad
├── ocra_sim_node.py        # ROS2 node → Gazebo (JointTrajectoryController)
├── ocra_node.py            # ROS2 node → hardware (Interbotix SDK)
├── camera_tracker.py       # OAK-D + MediaPipe → PoseArray publisher
├── fake_skele_pub.py       # Synthetic skeleton for testing
└── ocra_visualizer.html    # Live browser visualizer (via rosbridge)
```

---

## Dependencies

### ROS2 & Robot
- ROS2 Humble
- `interbotix_xsarm_sim` (Gazebo simulation)
- `interbotix_xs_msgs` (hardware only)
- `rosbridge_suite` — for live visualizer

```bash
sudo apt install ros-humble-rosbridge-suite
```

### Python
```bash
pip install jax jaxlib scipy numpy --break-system-packages
pip install depthai mediapipe --break-system-packages   # camera tracker only
```

---

## Setup & Launch

### Simulation

```bash
# Terminal 1 — Gazebo
ros2 launch interbotix_xsarm_sim xsarm_gz_classic.launch.py robot_model:=rx200

# Terminal 2 — OCRA controller
ros2 run robot_retarget ocra_sim_node

# Terminal 3 — Skeleton source (pick one)
python3 fake_skele_pub.py        # synthetic test data
# OR
python3 camera_tracker.py        # real OAK-D camera
```

### Hardware

```bash
# Terminal 1 — Interbotix driver
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=rx200

# Terminal 2
ros2 run robot_retarget ocra_node

# Terminal 3
python3 camera_tracker.py
```

### Live Visualizer

```bash
# Terminal — rosbridge
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

Open `ocra_visualizer.html` in a browser. Subscribes to:
- `/human/skeletal_data` — cyan chain (human)
- `/rx200/arm_controller/joint_trajectory` — orange chain (robot FK)

Drag to rotate, scroll to zoom. EE glow: 🟢 good / 🟡 ok / 🔴 bad loss.

---

## Key Parameters

| Parameter | Location | Value | Description |
|---|---|---|---|
| `LOOP_RATE` | `ocra_sim_node.py` | 10 Hz | Control loop frequency |
| `ALPHA` | `ocra_sim_node.py` | 0.67 | Skeleton error weight |
| `BETA` | `ocra_sim_node.py` | 0.33 | Orientation error weight |
| `maxiter` (first solve) | `ocra_sim_node.py` | 50 | SLSQP iterations, cold start |
| `maxiter` (warm start) | `ocra_sim_node.py` | 10 | SLSQP iterations, subsequent |
| `ftol` | `ocra_sim_node.py` | 1e-4 | Optimizer convergence tolerance |
| `time_from_start` | `ocra_sim_node.py` | 200 ms | Trajectory execution time |

### Joint Limits (RX200)

| Joint | Min (rad) | Max (rad) |
|---|---|---|
| waist | -3.1416 | 3.1416 |
| shoulder | -1.8849 | 1.9722 |
| elbow | -1.8849 | 1.6231 |
| wrist_angle | -1.7453 | 2.1467 |
| wrist_rotate | -3.1416 | 3.1416 |

---

## Forward Kinematics

Product of Exponentials (space frame):

```
T_hand  = exp([S₁]θ₁) · exp([S₂]θ₂) · exp([S₃]θ₃) · exp([S₄]θ₄) · exp([S₅]θ₅) · M_HOME
T_elbow = exp([S₁]θ₁) · exp([S₂]θ₂) · M_ELBOW
```

Screw axes `S` defined in the base frame. `M_HOME` and `M_ELBOW` are the zero-configuration end-effector and elbow poses respectively.

---

## Bugs Fixed During Development

| Bug | Impact |
|---|---|
| Missing skeleton normalization factor ℓ | Loss terms on incompatible scales |
| `arccos` instead of `arctan2` for orientation | NaN gradients at identity rotation |
| Scale factors computed but not applied in camera_tracker | Human arm never rescaled to robot proportions |
| Wrong message type for `/rx200/joint_states` | Joint state callback never fired |
| Joint name ordering assumed, not looked up | Wrong joints moved |
| Missing comma in `JOINT_NAMES` list | Python silently concatenated strings |
| JAX DeviceArray passed directly to scipy | "Inequality constraints incompatible" error |
| `MultiThreadedExecutor` flooding controller | Trajectory preempted before execution |
| `jnp.minimum` in distance → zero gradient | Optimizer made no progress |

---

## Known Limitations

- **IK redundancy** — the optimizer can find kinematically valid but visually unnatural solutions (robot folding on itself) when the target is reached by a contorted configuration. Mitigated by adding an explicit hand↔EE correspondence term to the loss.

- **Workspace boundary** — the RX200 has ~0.4 m reach. Targets outside this range cause the optimizer to slam joints to limits. The node filters skeleton frames where `|hand| > 0.4 m`.

- **Orientation term at identity** — `arccos(1.0)` has undefined gradient. Replaced with `arctan2(|xyz_norm|, |w|)` which is numerically stable everywhere.

- **Single-arm only** — currently tracks right arm kinematic chain. Left arm and full-body retargeting not implemented.

---

## References

Mohan, S. & Kuchenbecker, K. J. (2023). *OCRA: Optimization-based Customizable Retargeting Algorithm for Robotic Teleoperation*. ICRA 2023 Avatar Workshop.
