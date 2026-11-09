# FR3 teleoperation and demonstration collection

[简体中文快速流程](README_CN.md)

Joystick teleoperation, synchronized RealSense recording, LeRobot conversion,
and guarded OpenPI pi0.5 execution for a Franka Research 3. Robot
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
- `cameras.external_rotate_180`, `wrist_rotate_180`, and
  `external2_rotate_180` independently rotate each image by 180 degrees. They
  affect preview, recording, and inference pixels, but never robot motion axes.
- `cameras.external2_serial` is optional. Leave it empty to use only the
  required exterior and wrist cameras. When configured and connected, the L515
  is recorded as `exterior_image_2_left`; its configured mode is `960x540@60`.
- `[recording]` contains the synchronization limits. The defaults sample Bamboo
  at 1 kHz, align every stream to an exact 15 Hz Ubuntu monotonic-time grid,
  and reject camera/robot samples outside the configured tolerances.
- `recording.record_force` records joint torque and the full-rate `raw_force_*`
  stream. With it off, torque is omitted rather than zero-filled, and a
  controller that cannot report `tau_J` no longer refuses to collect.
- `recording.record_tactile` records the GelSight tactile stream, configured
  under `[tactile]`. With it on the sensor must be present: collection refuses
  to start rather than quietly producing episodes without tactile data.
- `recording.weigh` asks for scale readings after each episode. It is
  task-specific (the pouring task uses it) and off by default.

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
| D-pad left/right | Tool-frame +Y/-Y |
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
orientation. Stop the preview with `Ctrl+C` before collection: a RealSense
device can only have one owner, so `fr3-camera-rviz` and `fr3-collect` cannot
run at the same time. `fr3-collect` opens its own RViz instead — see below.

Start collection:

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
fr3-collect
```

`fr3-collect` opens RViz and previews the recorder's own frames without opening
the cameras a second time. A preview failure does not stop recording.

- `--no-preview` records without RViz (also `preview = false` in `config.toml`).
- `--preview-publish-only` publishes the topics without launching RViz, for
  viewing from another machine.
- `--preview-rate-hz` changes only the preview rate.
- With the GelSight in use (`--record-tactile` or `[gel]` on), the GelSight
  panel shows its live image, from the stream the session already reads.
  With the gel guard on, the guard's level is drawn along the bottom: green
  below half the limit, amber up to it, red at or past it, with the shift
  (px), twist (deg) and pressing (%) behind it.

If ROS 2 is not sourced, collection still runs and logs
`Camera preview disabled: ROS 2 is not sourced`.

- Press X once to start: `⬆️ Recording started` and one vibration.
- Press X again to finish: `✅ Recording stopped` and two vibrations.
- Stop recording before homing. Back exits and safely finalizes an active
  episode.

Sessions are stored under `data/raw/session_YYYYMMDD_HHMMSS`. Incomplete
episodes retain an `.inprogress` suffix and are ignored during conversion.

### Tactile sensor

The GelSight Mini is a UVC camera looking at a deformable gel pad. It exposes
exactly one mode, **3280x2464 MJPEG**, advertised at 25 fps. Measured here it
delivers a steady **18.75 Hz** (53.3 ms, p95 53.4, no retrieve failures) — the
device's own limit, not a software one: the rate is unchanged with the decode
removed entirely and at every driver buffer depth, and the sensor is a USB 2.0
device, so a USB 3 port will not raise it. That is still above the 15 Hz
recording grid.

Every frame is an 8 MP JPEG. Decoding in full costs about 44 ms, so capture
asks V4L2 for the raw MJPEG payload and lets libjpeg scale during decode,
which costs about 18 ms at 1/8 — comfortably inside the 53 ms the device
allows. Frames are downscaled to `tactile.width` x `tactile.height`; the gel
image carries nothing like 8 MP of information.

Check the sensor and measure what it actually delivers before collecting:

```bash
fr3-tactile-check --seconds 10
```

A UVC device has a single owner, so close any viewer first (`fuser -v
/dev/video20` names the holder) — the same constraint as `fr3-camera-rviz`
versus `fr3-collect`.

Then collect with it:

```bash
fr3-collect --record-tactile
```

Tactile capture runs at the sensor's own rate, **not** on the 15 Hz grid:

- `frames/tactile_image/` holds one JPEG per captured frame, so the full
  temporal resolution is kept.
- `tactile_frame_index` maps each 15 Hz row to the nearest captured frame, so
  an aligned view costs no duplicated images.
- `raw_tactile_timestamp` gives every frame's host-monotonic time.

`sync_report.json` reports the achieved tactile rate and the grid alignment
error, and a row whose nearest frame is older than `tactile.max_age_ms` fails
the episode the same way a stale camera does.

### Hand-guided collection

```bash
fr3-collect --hand-guide                 # add --record-tactile with the GelSight
```

`fr3-collect` sends no motion commands in this mode; you move the arm by hand.
How depends on the Bamboo server:

- **Guide mode** (a server with the `guide_mode` feature): the arm floats.
  Grab it anywhere and move it, with no guiding buttons. No joint can be moved
  faster than `[guide] max_joint_velocity` (0.35 rad/s by default, the
  streaming limit, so every demonstration can be replayed): below it you feel
  nothing, above it a stiff spring holds you back (Bamboo's speed leash, up to
  40 N m on the large joints). Left still for a second, the arm holds
  its pose and breaks free when pushed. Near a joint limit it pushes back,
  harder the closer you get, but never blocks moving away. If the update
  stream stops, the arm damps to a stop and holds; press **RB** to resume once
  it is running again.
- **Pilot buttons** (an older server, or `--pilot-guide`): the robot's own
  guiding buttons move the arm, as before.

Guide mode is a Franka motion like teleop, so the robot must be ready the same
way: Desk in Execution mode and the user stop released. White lights mean
Programming mode, where only the Pilot buttons (`--pilot-guide`) move the arm;
guide mode refuses to start there and says why. Pressing the user stop while
guiding stops the arm and ends the session; release it and run `fr3-collect`
again.

Guide mode refuses to start while Franka's external force estimate is above
`[guide] max_start_force` (5 N): usually the gripper's weight is not
compensated, so a released arm would sink. Weigh the gripper, sensor and tool
and declare them in `[payload]` (unless Desk already lists that mass).

The action is the arm's own motion: each row stores the average joint
velocity over the next row period, from the 1 kHz joint positions, and the
episode metadata says `action_source: measured_joint_velocity`. In guide
mode **Menu** homes the arm between episodes: guide mode stops, so let go,
the arm streams home (under `[contact]` when it is set), and guide mode comes
back on its own, waiting for your hand to leave the arm if it is still on it.
It is refused while recording, and with the Pilot buttons. With `[gel]` on, the GelSight guard below protects the
gel while you guide; without it nothing limits the force on the gel.

### Replaying each demonstration

```bash
fr3-collect --hand-guide --replay --record-tactile
```

While you guide the arm, Franka's force estimate mixes your hand with the
contact. With `--replay`, each demonstration is replayed by the robot alone
after you record it, and the replay is recorded too:

1. **X** starts recording, always at home: if the arm is elsewhere (the
   first demonstration of a session, say), it homes first, hands off, and
   recording starts once it is there. Guide the demonstration; **X** stops
   it. End with the tool clear of the fruit: the arm next moves straight (in
   joint space) home.
2. Hands off: guide mode stops and the arm homes, then holds there. Arrange
   the scene for the replay (turn the lemon to fresh skin, say), keeping
   clear of the tool.
3. **X** replays it: the arm settles onto where the demonstration started
   (within 0.003 rad, about 1 mm at the blade) and replays it while a new
   episode records, then homes. **B** skips the replay instead.
4. Guide mode comes back on; the next **X** starts the next demonstration.

Nobody touches the arm during a replay, so Franka's force estimate is the
force of the task itself; it is recorded as `external_wrench` and
`external_joint_torque` (see Recording format). Your hands are out of the
camera views too. The replay follows the joint path the arm actually took.
Where the blade meets resistance the arm is pushed off that path, so the
replay presses until the blade is where yours was, up to `[contact]
max_force_n`. Because a replay still came out shallower than the
demonstration, it replays the path `--replay-depth-mm` deeper (1.5 mm by
default, 0 to 3; 0 replays it as recorded), along `--replay-depth-direction`
in the robot base frame (default: straight down). The whole tool moves, without
turning, so the offset does not depend on where exactly the blade is; it eases
in over the first second and out over the last, so the start and end poses
are the demonstration's own. The push cap, the gel guard and Franka's contact
guard bound the force exactly as before, and the replay's metadata records
the offset. It runs under the streaming limits a policy has
(`[teleop] max_joint_speed` and `max_joint_acceleration`), so the fast parts
of a demonstration come out a little smoother.

A replay needs `[contact]` enabled (measure `contact_point_m`), `[gel]`
enabled while the GelSight is in use, and Bamboo's guide mode; the session
refuses to start otherwise. During a replay **X** or **B** stops it (the arm
stops and holds) and **Back** stops it and quits. It also stops when the gel
reaches its limit, when Bamboo's contact guard trips, or when the arm is
pushed more than `--replay-max-path-error` (0.08 rad) off the path; these
yield, so the arm goes soft and the load falls. After any stop, guide mode
comes back where the arm is: free the blade by hand, then **Menu** homes.
While the arm waits at home, anything that loads the gel makes it go soft and
skips that replay, so keep clear of the tool while you arrange the scene.

**Re-grip when the tool has turned in the grip.** The gel guard measures from
its last zero, taken when the session starts and after every gripper button.
Peeling slowly turns a round handle in the grip, and the turn stays in the gel
at rest (a few degrees of twist after a few cycles), so later strokes start
that much nearer the limit; this is what stopped a replay at the twist limit.
`fr3-collect` checks the gel at rest:

- When you press **X** to record and the gel rests above 30% of its limit with
  nothing touching the tool, recording waits and says why. Re-grip: hold the
  tool, press **B** (the gripper opens), then **A** (it closes, and the gel
  re-zeroes on the new grasp), then press **X**.
- While the arm waits at home to replay, the same check gives a heads-up that
  the replay starts with less room; re-grip before the next demonstration.
- A replay the gel guard stopped reminds you to re-grip too.

The GelSight panel shows the same reading at any time: at home with nothing
touching the tool it should be near 0%. Flat faces on the handle where the
fingers grip (a printed sleeve, say) keep it from turning in the first place.

The replay is an ordinary episode whose actions are the joint velocities the
robot was commanded (`action_source: replay_joint_velocity_command`). Its
metadata names its demonstration (`replay_of`) and says how the replay went
(`replay.completed`, why it stopped, how far it was pushed off the path); the
demonstration lists its replays. A demonstration that opens or closes the
gripper is not replayed. To train on the robot's replays only, convert with
`fr3-convert --episodes replays`; replays that stopped early are always left
out.

### Protecting the GelSight

Two independent layers, both off until enabled in `config.toml`:

| | `[contact]` (Bamboo, 1 kHz) | `[gel]` (GelSight, ~25 Hz) |
|---|---|---|
| Senses | Franka's force estimate and the stream's own push | the marker gel itself |
| Teleop, policy runs | push cap, force guard that yields, Franka reflex | blocks driving into the load (teleop); stops the run at the limit or if the signal is lost |
| Hand guiding | not applicable (your hand is part of the force) | the arm stiffens against further loading; at the limit it will not go further that way |
| Caveat | counts lifted objects (1 kg reads ~10 N): contact tasks only | needs the marker gel; limits are in the gel's own units |

The contact limits make a policy run unable to push the tool harder than the
set force, and Franka's own reflex is the backstop if anything else fails.
While you guide by hand, the gel guard resists and warns, but it cannot stop
a hand that pushes through it; only a mechanical limit could.

Bring-up, with the user stop at hand:

1. Rebuild Bamboo on the robot PC (`feature/contact-safety`) and restart it.
2. Fit the marker gel and the tool. The guard's level is the gel's own
   shift, twist and pressing against `[gel]`'s limits, so it works without
   any calibration. For guide mode to block only the loading direction at
   the limit (rather than hold), run `fr3-gel-calibrate`: with the arm still
   and no loop running, press the tool where it meets the fruit, gently
   sideways (a newton or two twists the gel a lot) and a few newtons along
   the other directions, touching only the tool. A direction that barely
   moves the gel is reported as going around it (through a hard opposite
   finger), not refused. Copy the lines it prints into `[gel]`, check the
   limits against what it says your pushes reached, and set `enabled = true`.
3. `[contact]`: measure your tool tip (`contact_point_m`), set
   `enabled = true` and `shadow = true`, and teleop through a normal task
   while `fr3-contact-check` runs in a second terminal. In free motion the
   push should stay well under `max_force_n` and the tool force near 0 N.
4. Set `shadow = false` and press the tool into a kitchen scale by teleop:
   the scale should stop near `max_force_n`, and pushing on harder makes the
   guard yield. Then the same with a policy run.
5. Hand guiding: press the tool into the scale by hand; the arm should
   stiffen as the gel's level rises and stop advancing at level 1.

`fr3-collect` and `fr3-pi05-run` set the limits when they start and clear them
when they exit; `--no-contact-limits` and `--no-gel-guard` turn either off for
one session.

### Recording format

- Cameras capture at 60 Hz; all streams are aligned to an exact 15 Hz Ubuntu
  monotonic-time grid. Each completed episode includes `sync_report.json`, and
  an out-of-tolerance episode is aborted.
- Robot position, velocity, and measured joint torque (`tau_J`, Nm), gripper
  state, actions, and camera timing metadata share that timeline. With
  `record_force`, so does Franka's external force estimate when Bamboo reports
  it: `external_wrench` (N and N m, base frame) and `external_joint_torque`
  (N m), positive for what the robot applies to its surroundings, with the
  unsampled stream in `raw_force_external_*`. It counts every push from
  outside, a hand on the arm included.
- JPEGs are stored at the DROID size, 320x180. Set both configured image
  dimensions to `0` for native resolution. Collection stops before the disk
  falls below `min_free_gb`.

To shrink older native-resolution recordings, preview the change and then apply
it (rewriting JPEGs is lossy):

```bash
fr3-downsize --data-dir data/raw/session_20260821_122332_lillet
fr3-downsize --data-dir data/raw/session_20260821_122332_lillet --apply
```

## 5. Add language instructions

Annotate every episode separately:

```bash
fr3-annotate --data-dir data/raw/session_YYYYMMDD_HHMMSS
```

A robot replay (`fr3-collect --replay`) is not asked about: it takes its
demonstration's instruction, and follows it when the demonstration's is
replaced with `--replace`. Annotating a session that is still recording is
safe: only the instruction is written, into the metadata as it is on disk
at that moment.

Prompt once and label every currently unlabeled episode, preserving existing
labels:

```bash
fr3-annotate --data-dir data/raw/session_YYYYMMDD_HHMMSS --all
```

Add `--replace` to overwrite every episode, including existing labels, from the
same single prompt:

```bash
fr3-annotate --data-dir data/raw/session_YYYYMMDD_HHMMSS --all --replace
```

For non-interactive use:

```bash
fr3-annotate --data-dir data/raw/session_YYYYMMDD_HHMMSS \
  --language "pick up the object and place it in the bowl"
```

## 6. Convert to LeRobot

Current `fr3-collect` sessions use synchronized schema v3. Convert locally
first:

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

Sessions recorded with `--replay` hold both hand demonstrations and the
robot's replays of them: `--episodes demos` or `--episodes replays` converts
one kind only (default: both), and replays that stopped early are always left
out.

Add `--public` only when the images are safe to publish. Conversion rejects
missing language, mismatched frames, and an existing output directory. The
output follows the OpenPI DROID layout with exterior and wrist RGB, joint and
gripper state, eight-dimensional actions, and per-episode tasks. The second
exterior image uses the optional camera when recorded and falls back to the
black DROID compatibility image for older/two-camera episodes.

### Tactile in the dataset

When every input episode has tactile, conversion adds a `tactile_image` feature
holding the gel frame nearest each 15 Hz row, alongside the three RealSense
images. Because the sensor runs faster than the grid, that column is a
resampling, so the complete stream is written beside the dataset as well:

- `tactile/episode_NNNNNN.npz` — every captured JPEG byte-for-byte as recorded,
  concatenated into `jpeg_blob` with a `jpeg_offsets` table, plus each frame's
  `timestamp` and the per-row `row_frame_index`. One file per episode rather
  than thousands of small ones, and readable without `allow_pickle`.
- `meta/tactile_report.json` — per-episode frame count, achieved rate, largest
  gap and grid alignment error.

If only some episodes have tactile, the `tactile_image` feature is omitted
rather than padded, while the sidecars for the episodes that do have it are
still written so nothing recorded is discarded.

The gel image barely changes while peeling: the markers move 1-3 px, about 1%
of the frame. Conversion therefore also measures that motion, following the
markers through every captured frame from each episode's first gel frame:

- `tactile_deformation` — marker shift (`shift_x_px`, `shift_y_px`, image
  axes), `twist_rad` and `spread` (how much the marker field grew: pressing),
  per 15 Hz row.
- `tactile_tracked` — 1 where the markers were tracked; 0 (with zero
  deformation) where they were lost.

These need the marker gel and OpenCV (`pip install -e '.[tactile]'`); with a
plain gel, they are left out and conversion says why.

### Force in the dataset

- `joint_torque` — Franka's measured joint torque, `tau_J` (N m).
- `external_wrench` and `external_joint_torque` — Franka's own force
  estimate: the wrench in the robot base frame (N, N m) and per joint (N m),
  positive for what the robot applies to its surroundings. During hand
  guiding it includes the hand; in a replay or teleop, nobody touches the arm,
  so it is the task's own force. It has an offset of a few newtons that
  depends on how the arm is held, so compare it with its value just before
  contact.
- `force/episode_NNNNNN.npz` — the arm stream as sampled (up to 1 kHz):
  `joint_torque`, `joint_position`, `external_wrench`,
  `external_joint_torque`, their `robot_timestamp`, and the rows'
  `row_robot_timestamp` on the same controller clock.

Each feature is carried only when every episode has it, as for tactile.

`meta/fr3_episodes.json` records where each episode came from: its source
episode, `kind` (`hand_guided`, `replay` or `teleop`), `action_source`, and
for a replay its demonstration (`replay_of`, with that demonstration's index
here when it was converted too) and how the replay went. A hand-guided
demonstration's action is the motion it measured, while a replay's or
teleop's is the velocity it commanded. Conversion notes when a dataset mixes
them; `--episodes replays` converts the replays alone.

Schema-v1 recordings used loop-index alignment and are rejected by default. To
convert one only after manually accepting its timing risk, add:

```bash
fr3-convert ... --allow-legacy-unsynchronized
```

This flag is only a validation bypass for recordings created before synchronized
collection was added. It is not an upload option and must not be added for
schema-v2/v3 data; `--push-to-hub` is the upload option.

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
| `fr3-collect --hand-guide` | Hand-guided recording (Bamboo guide mode or Pilot buttons) |
| `fr3-collect --hand-guide --replay` | Hand-guided recording, each demonstration replayed and recorded by the robot |
| `fr3-camera-list` | List RealSense serial numbers |
| `fr3-camera-rviz` | Preview all cameras in RViz (standalone; not while collecting) |
| `fr3-annotate` | Add episode language instructions |
| `fr3-convert` | Build and optionally upload a LeRobot dataset |
| `fr3-downsize` | Rewrite older native-resolution episodes at the stored size |
| `fr3-tactile-check` | List tactile devices and measure the achieved rate |
| `fr3-gel-calibrate` | Fit the GelSight guard's force calibration against Franka's estimate |
| `fr3-contact-check` | Watch the tool force, push cap and guard live (read-only) |
| `fr3-pi05-check` | Non-moving policy integration check |
| `fr3-pi05-run` | Guarded pi0.5 rollout |

Use `COMMAND --help` for all options.

## Tests

```bash
python -m unittest discover -s tests -v
```
