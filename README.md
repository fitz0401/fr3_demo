# FR3 teleoperation and demonstration collection

Joystick teleoperation, synchronized two-RealSense recording, LeRobot
conversion, and guarded OpenPI pi0.5 execution for a Franka Research 3. Robot
control runs through [fitz0401/bamboo](https://github.com/fitz0401/bamboo) on
the real-time machine; this repository runs on the operator workstation.

## Safety

- Keep the physical E-stop ready and clear the workspace before motion.
- Run `fr3-teleop --check`, then `fr3-teleop --dry-run`, before live control.
- Teleoperation has no deadman button and becomes active immediately.
- The software enforces joint and Cartesian limits, but does not provide
  environment or self-collision avoidance.

## 1. Install

Python 3.10 or newer is required:

```bash
git clone git@github.com:fitz0401/fr3_demo.git
cd fr3_demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[recording]'
```

Install the conversion dependencies only when needed:

```bash
python -m pip install -e '.[convert]'
```

## 2. Configure

Edit [`config.toml`](config.toml). It contains the Bamboo address, gripper,
joystick, camera serials, motion rates, home pose, workspace, recording, and
pi0.5 server settings. Every command loads this file automatically.

Useful details:

- `teleop.frame = "base"` uses fixed robot-base axes; `"tool"` follows the EEF
  orientation.
- `workspace.min` and `workspace.max` are `[x, y, z]` in metres in the robot
  base frame. They remain base-frame limits in tool mode.
- `gripper.close_force` is normalized from `0.0` to `1.0`.
- `cameras.wrist_rotate_180` changes image orientation only, not motion axes.

CLI options override the file for one run. Use `--config PATH` or set
`FR3_DEMO_CONFIG` to load a different configuration.

## 3. Start and verify teleoperation

On the real-time machine, start the Bamboo arm and gripper services:

```bash
cd /path/to/bamboo
bash RunTeleopController start --robot_ip 172.16.0.2 --robot_model fr3
```

Do not run `RunBambooController` and `RunTeleopController` together.

On the operator workstation:

```bash
source .venv/bin/activate
fr3-teleop --check       # read-only hardware and model check
fr3-teleop --dry-run     # joystick input without robot commands
fr3-teleop               # live control
```

The check must report `Streaming protocol: available`. Use
`fr3-teleop --frame tool` for gripper-relative stick motion.

### BETOP controls

| Control | Action |
| --- | --- |
| Left stick up/down | EEF +X/-X in the selected frame |
| Left stick left/right | EEF +Y/-Y in the selected frame |
| LT / LB | EEF -Z/+Z in the selected frame |
| D-pad up/down | Tool-frame +Z/-Z |
| D-pad left/right | Tool-frame -Y/+Y |
| Right stick left/right | Roll |
| Right stick up/down | Pitch |
| RT / RB | Yaw -/+ |
| A / B | Close/open gripper |
| Menu / Back | Home/quit |
| X | Start/stop recording in `fr3-collect` |

D-pad commands always use the tool frame. Its Y and Z axes are mutually
exclusive, so diagonal input cannot command both directions simultaneously.

## 4. Collect demonstrations

Check the camera assignment before recording:

```bash
fr3-camera-list
source /opt/ros/humble/setup.bash
fr3-camera-rviz
```

Use `fr3-camera-rviz --no-wrist-rotate-180` to inspect the raw wrist
orientation. Stop the preview before collection because a RealSense device can
only have one owner.

Start collection:

```bash
source .venv/bin/activate
fr3-collect
```

- Press X once to start: `⬆️ Recording started` and one vibration.
- Press X again to finish: `✅ Recording stopped` and two vibrations.
- Stop recording before homing. Back exits and safely finalizes an active
  episode.

Sessions are stored under `data/raw/session_YYYYMMDD_HHMMSS`. Incomplete
episodes retain an `.inprogress` suffix and are ignored during conversion.

## 5. Add language instructions

Annotate every episode separately:

```bash
fr3-annotate --data-dir data/raw/session_YYYYMMDD_HHMMSS
```

Prompt once and label every currently unlabeled episode, preserving existing
labels:

```bash
fr3-annotate --data-dir data/raw/session_YYYYMMDD_HHMMSS --all
```

For non-interactive use:

```bash
fr3-annotate --data-dir data/raw/session_YYYYMMDD_HHMMSS \
  --language "pick up the object and place it in the bowl"
```

## 6. Convert to LeRobot

Convert locally first:

```bash
fr3-convert \
  --data-dir data/raw/session_GIN data/raw/session_LILLET \
  --repo-id USERNAME/fr3_task \
  --output-root data/lerobot
```

Provide one or more paths after `--data-dir`. Each path is recursive, overlapping
inputs are deduplicated, and every episode retains its own language instruction.
Repeating the option also works, but the one-flag form above is shorter.

After inspecting the local result, upload it privately:

```bash
hf auth login
fr3-convert \
  --data-dir data/raw/session_GIN data/raw/session_LILLET \
  --repo-id USERNAME/fr3_task \
  --output-root data/lerobot \
  --push-to-hub
```

Add `--public` only when the images are safe to publish. Conversion rejects
missing language, mismatched frames, and an existing output directory. The
output follows the OpenPI DROID layout with exterior and wrist RGB, joint and
gripper state, eight-dimensional actions, and per-episode tasks. The required
second exterior image is a black compatibility field ignored by pi0.5.

## 7. Run pi0.5

GPU deployment, checkpoint selection, networking, safety checks, and RViz
instructions are in [fr3_pi05/README.md](fr3_pi05/README.md).

On the operator workstation:

```bash
fr3-pi05-check --checkpoint pi05_droid --prompt "task instruction"
fr3-pi05-run --checkpoint pi05_droid --execute --prompt "task instruction"
```

`fr3-pi05-check` never moves the robot. `fr3-pi05-run` homes the arm and opens
the gripper before inference; policy motion additionally requires `--execute`.

## Commands

| Command | Purpose |
| --- | --- |
| `fr3-teleop` | Joystick teleoperation |
| `fr3-collect` | Teleoperation with synchronized recording |
| `fr3-camera-list` | List RealSense serial numbers |
| `fr3-camera-rviz` | Preview both cameras in RViz |
| `fr3-annotate` | Add episode language instructions |
| `fr3-convert` | Build and optionally upload a LeRobot dataset |
| `fr3-pi05-check` | Non-moving policy integration check |
| `fr3-pi05-run` | Guarded pi0.5 rollout |

Use `COMMAND --help` for all options.

## Tests

```bash
python -m unittest discover -s tests -v
```
