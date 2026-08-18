# Franka Research 3 teleoperation and demo collection

This repository provides Cartesian EEF teleoperation and synchronized two-camera
demonstration collection for an FR3 through the running
[fitz0401 Bamboo controller](https://github.com/fitz0401/bamboo). It records a
local, failure-preserving raw format and converts annotated episodes to the exact
LeRobot feature layout used by OpenPI's `pi05_droid_finetune` setup.

The implementation uses the official FR3 kinematic chain and converts a base- or
tool-frame Cartesian velocity into damped-Jacobian joint velocity. A persistent
controller on the real-time machine consumes setpoints at 30 Hz while retaining
the 1 kHz Bamboo joint-impedance loop. Every update is bounded by a workspace
box, joint limits, and velocity/acceleration caps.
Controller and transport implementations live in Bamboo; this repository only
contains the workstation-side teleoperation and demo-collection code.

## Safety

- Clear the workspace, inspect wrist-camera cabling, and keep the physical E-stop
  in hand.
- Start with `--check`, then `--dry-run`. Only omit `--dry-run` after confirming
  the axes and EEF directions.
- Teleoperation becomes active immediately after startup; there is no joystick
  deadman button. Returning every motion control to neutral commands zero
  velocity. A controller-side watchdog also brakes the arm if updates disappear
  for 250 ms.
- This first version enforces an EEF box and joint limits, but it does **not** do
  environment or self-collision checking.
- A software stop is not a replacement for the Franka physical E-stop.

## Install

Python 3.10+ is required. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For camera collection and joystick vibration, install the recording extra:

```bash
python -m pip install -e '.[recording]'
```

The connected BETOP controller is available as `/dev/input/js0`, and Bamboo was
detected at `172.16.0.20:5555` with the Robotiq service on port `5559`; these are
the defaults. To use another machine or device, pass `--server-ip` or `--joystick`.

Persistent robot, gripper, joystick, camera, recording, speed, homing, and
workspace settings live in [`config.toml`](config.toml). All commands load this
file automatically. Use `--config /another/file.toml` (or set
`FR3_DEMO_CONFIG`) to select another setup; explicit command-line arguments
remain one-run overrides.

## Validate and run

### Start Bamboo on the real-time machine

Install the merged `fitz0401/bamboo` version and launch its arm and gripper
services before starting this program:

```bash
cd /path/to/bamboo
bash RunTeleopController start --robot_ip 172.16.0.2 --robot_model fr3
```

Do not run `RunBambooController` and `RunTeleopController` at the same time.

Read-only hardware/model check (never sends a trajectory):

```bash
source .venv/bin/activate
fr3-teleop --check
```

The check must report `Streaming protocol: available` before smooth live mode.

Exercise the gamepad while only calculating and printing targets:

```bash
fr3-teleop --dry-run
```

Run live, initially at the conservative default of 8 cm/s, 0.35 rad/s, and a
30 Hz setpoint rate:

```bash
fr3-teleop
```

The convenience form `python teleop.py` accepts the same options. Use
`fr3-teleop --help` for speed, workspace, port, frame, and gripper settings.
The old blocking Bamboo API remains available as `--legacy-waypoints` for
diagnosis only; it necessarily retains the stop-and-go motion.

### BETOP controls

| Control | Action |
| --- | --- |
| Left stick up/down | EEF +X/-X in the selected frame |
| Left stick left/right | EEF -Y/+Y |
| LT / LB | EEF -Z/+Z |
| Right stick left/right | EEF roll |
| Right stick up/down | EEF pitch |
| RT / RB | EEF negative/positive yaw |
| A | Close gripper |
| B | Open gripper |
| X | Start/stop an episode when using `fr3-collect` |
| Menu/Start | Move to the nominal FR3 home configuration |
| Back/Select | Exit |

The default frame is the robot base. For tool-relative motion, use
`fr3-teleop --frame tool`. Homing uses closed-loop joint-velocity streaming at
up to 0.20 rad/s by default; change the limit with `--home-speed`. Its timeout
automatically grows for large moves and can be raised with `--home-timeout`.

## Collect demonstrations

### 1. Assign and preview the cameras

List the connected RealSense devices:

```bash
fr3-camera-list
```

This workstation currently detects these two serials:

```text
309622300781  Intel RealSense D456
047322071010  Intel RealSense D435I
```

The likely assignment is D456 as the exterior view and D435I as the wrist view.
That assignment is already stored in `config.toml`. Verify the physical views in
RViz and swap `cameras.external_serial` and `cameras.wrist_serial` in the file if
the displays are reversed:

```bash
source /opt/ros/humble/setup.bash
fr3-camera-rviz
```

The preview publishes `/fr3_demo/exterior_image_left` and
`/fr3_demo/wrist_image`. Stop it with Ctrl+C before collection because a
RealSense device cannot be owned by the preview and recorder simultaneously.

### 2. Record episodes

With `RunTeleopController` still running on the real-time machine:

```bash
source .venv/bin/activate
fr3-collect
```

Camera serials and all normal collection parameters are read from `config.toml`.

- Press X once to start. One vibration confirms recording.
- Teleoperate and use the gripper normally.
- Press X again to finish. Two vibrations confirm that the episode was safely
  finalized.
- Repeat for more episodes. Back exits; an active episode is finalized first.
- Homing is rejected while recording; stop the episode before pressing Menu.

The recorder samples two RGB streams, robot joint state, commanded joint
velocity, and normalized gripper state/action at 15 Hz. Camera capture and JPEG
encoding run outside the 30 Hz control loop. Completed sessions are placed under
`data/raw/session_YYYYMMDD_HHMMSS`; an interrupted episode retains the
`.inprogress` suffix and is never converted or uploaded.

### 3. Add language

After exiting, annotate each episode interactively:

```bash
fr3-annotate --data-dir data/raw/session_YYYYMMDD_HHMMSS
```

To apply one instruction to every unannotated episode in a session:

```bash
fr3-annotate \
  --data-dir data/raw/session_YYYYMMDD_HHMMSS \
  --language "pick up the red block and place it in the bowl"
```

### 4. Convert to LeRobot and upload

Install the conversion extra. It pins the same LeRobot revision used by the
current OpenPI repository (LeRobot pulls a full ML stack, including PyTorch):

```bash
python -m pip install -e '.[convert]'
hf auth login
```

If OpenPI is already installed, its locked environment is preferable and avoids
installing that stack twice:

```bash
cd /path/to/openpi
uv sync
uv pip install -e /home/u0161364/fr3_demo
uv run hf auth login
uv run fr3-convert --help
```

Convert and upload (private by default):

```bash
fr3-convert \
  --data-dir data/raw/session_YYYYMMDD_HHMMSS \
  --repo-id YOUR_HF_USERNAME/fr3_task \
  --output-root data/lerobot \
  --push-to-hub
```

Add `--public` only if the images and demonstrations are safe to publish. Omit
`--push-to-hub` to validate the local dataset first. The converter refuses
missing language, incomplete episodes, mismatched frame counts, and an existing
output directory so it does not silently overwrite data.

The resulting DROID-style LeRobot schema is:

| LeRobot feature | Recorded source |
| --- | --- |
| `exterior_image_1_left` | `exterior_image_left`, resized to 320x180 RGB |
| `wrist_image_left` | `wrist_image`, resized to 320x180 RGB |
| `exterior_image_2_left` | Black compatibility view; pi0.5 masks/ignores it |
| `joint_position` | Seven measured FR3 joint positions |
| `gripper_position` | Normalized measured opening, shape 1 |
| `actions` | Seven commanded joint velocities plus gripper target |
| `task` | Your episode language instruction |

In OpenPI, set `repo_id` in the `pi05_droid_finetune` configuration to the Hub
dataset ID above. That configuration uses `LeRobotDROIDDataConfig`, the
pi0.5-DROID checkpoint, and its original DROID normalization statistics.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests verify FK against a read-only pose sampled from the current Bamboo/FR3
setup, validate the Jacobian numerically, exercise input shaping and safety
limits, atomically finalize a synthetic raw episode, and validate the OpenPI
DROID conversion schema without moving hardware.
