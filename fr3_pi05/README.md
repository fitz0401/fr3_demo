# FR3 pi0.5 DROID deployment

This package connects the current FR3 workstation to Physical Intelligence's
official `pi05_droid` policy. The workstation owns the RealSense devices and
Bamboo robot connection; the GPU workstation only performs GPU inference. Its
`enp103s0` interface is `172.16.0.30/24`, directly reachable from this
workstation's `eno1` at `172.16.0.3/24` with measured sub-millisecond latency.
The default transport is direct ZMQ over this dedicated inference LAN; the
GPU's separate `enp106s0` campus connection remains `10.38.32.253`.
The official OpenPI WebSocket transport remains selectable as a fallback.

The exact request fields are:

- `observation/exterior_image_1_left`: exterior RGB, padded to 224x224 for
  DROID/custom or resized to the wine training geometry of 320x180;
- `observation/wrist_image_left`: wrist RGB with the same profile-specific resize;
- `observation/joint_position`: seven measured FR3 joint positions (`(3, 7)`
  for `wine_hybrid`: current, t-45, t-75);
- `observation/gripper_position`: one normalized opening value (three for
  `wine_hybrid`, with the same history offsets);
- `prompt`: the operator's language instruction.

The installed wrist camera is upside-down relative to the DROID hand-camera
mount. Pinhao's validated reproduction rotates it 180 degrees, so the shared
`cameras.wrist_rotate_180 = true` setting applies the same rotation before
policy inference, RViz publication, and future demonstration recording. The
exterior image is not changed. Use `--no-wrist-rotate-180` only after physically
remounting the camera, or temporarily to compare the raw orientation:

```bash
fr3-pi05-check --no-wrist-rotate-180 --prompt "test camera orientation"
```

The override applies to one run only. Set `cameras.wrist_rotate_180 = false` in
`config.toml` to keep rotation disabled.

New raw sessions record these transforms in `session.json` and each episode's
`metadata.json`. With the current configuration the stored wrist JPEGs are
already rotated; conversion does not rotate them again. Older sessions created
before this metadata field was added must be interpreted from the configuration
used during collection.

Both RealSense color streams use 424x240 at 30 Hz, matching Pinhao's validated
deployment. This 16:9 mode preserves essentially the same wide calibrated FOV
as 1280x720 while reducing USB and network latency. The policy samples the
latest frame at DROID's 15 Hz control rate and applies OpenPI's official 224x224
padded resize. The wrist D456 remains close to the wide ZED Mini geometry used
by DROID, although the RealSense setup is not an exact stereo-hardware match.

The client accepts the official 15x8 and custom/wine 16x8 responses and
interprets each row as seven joint velocities plus one gripper-position target.
It runs at the dataset rate of 15 Hz. The wine profile executes all 16 actions
and replans from a fresh boundary observation; it does not reuse action zero of
a chunk predicted from an earlier state.

## 1. GPU server from its local console

The KU Leuven certificate authenticates successfully on `jump-l`, but the
gateway policy currently denies forwarding to `10.38.32.253`. SSH is therefore
not needed for inference itself. From a terminal physically on the GPU machine,
update the repository already present at `~/fr3_demo` (or clone it there):

```bash
cd ~
git clone https://github.com/fitz0401/fr3_demo.git fr3_demo
cd fr3_demo
bash fr3_pi05/remote/audit_host.sh
```

Send the audit output back before installing anything. It checks
both GPUs, active GPU processes, disk space, `uv`/Conda, candidate OpenPI trees,
checkpoint caches, and port 8000 without changing the host.

Do not use `enxbe3af2b6059f` on the GPU or `enxb03af2b6059f` on the robot
workstation for inter-host inference. They are ASPEED RNDIS Ethernet Gadget
interfaces connected internally to each machine's BMC, despite their similar
MAC-derived names. A dedicated link requires a real spare PCIe/USB Ethernet
adapter on the GPU, connected directly to a real workstation NIC or to the same
unisolated switch.

The deployed GPU interface is configured persistently through NetworkManager:

```text
enp103s0: 172.16.0.30/24, no gateway, never-default
eno1:     172.16.0.3/24
```

`ping -c 3 172.16.0.30` must succeed from the robot workstation before starting
inference. Do not add a gateway or DNS server to the dedicated connection.

This deployment uses checkpoints already managed on the GPU host and never
downloads checkpoint weights itself:

| Local selection | GPU checkpoint | ZMQ port |
| --- | --- | --- |
| `pi05_droid` | `/mnt/data/yurui/.cache/openpi/openpi-assets/checkpoints/pi05_droid` | 8000 |
| `custom_droid` | `/mnt/data/yurui/models/pi05_custom_droid_14999` | 8001 |
| `wine_hybrid` | `/mnt/data/yurui/models/pi05_wine_hybrid_17500` | 8002 |

The audit confirmed both checkpoints. The custom checkpoint is loaded through
its own `serve_custom_droid.py`, including `gemma_2b_lora_r32`, its 16-step
horizon, and `assets/fitz0401/custom_droid/norm_stats.json`. The launcher refuses
to substitute stock DROID normalization statistics. Its mandatory
`deploy.patch` must already be applied to the OpenPI checkout as described in
the checkpoint's `DEPLOY.md`.

Run only one policy at a time on the A6000. Each model needs roughly the large
majority of that GPU's memory after JAX initialization; the separate ports make
selection unambiguous but do not imply that both models should be resident
simultaneously. Stop the current policy cleanly before switching checkpoints.

When the OpenPI checkout and checkpoint structure are confirmed, choose the idle
A6000 index reported by `nvidia-smi`. Start the official policy on port 8000:

```bash
CUDA_VISIBLE_DEVICES=1 PORT=8000 \
  bash fr3_pi05/remote/start_pi05_zmq_server.sh \
  /absolute/path/to/openpi \
  "$HOME/fr3_demo" \
  /mnt/data/yurui/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
  pi05_droid
```

After stopping the official policy, start the custom policy on port 8001 with
its exact supplied loader:

```bash
CUDA_VISIBLE_DEVICES=1 PORT=8001 \
  bash fr3_pi05/remote/start_pi05_zmq_server.sh \
  /absolute/path/to/openpi \
  "$HOME/fr3_demo" \
  /mnt/data/yurui/models/pi05_custom_droid_14999 \
  custom_droid
```

The wine checkpoint must use the dedicated `fr3_train/serve_wine.py` recipe,
not a copied custom-DROID loader. The launcher imports that recipe directly and
uses `assets/fitz0401/franka_pour_wine/norm_stats.json`. It requires the
training-time proprio history `[current, t-45, t-75]`: joints shaped `(3, 7)`
and gripper shaped `(3,)`. At 15 Hz the lagged frames are 3 and 5 seconds old.
It also enforces the training OpenPI revision
`15a9616a00943ada6c20a0f158e3adb39df2ccac` and both changes in
`fr3_train/deploy.patch`; use a dedicated checkout if the existing OpenPI tree
serves other experiments. Start it on port 8002:

```bash
CUDA_VISIBLE_DEVICES=1 PORT=8002 \
  bash fr3_pi05/remote/start_pi05_zmq_server.sh \
  /mnt/data/yurui/openpi \
  "$HOME/fr3_demo" \
  /mnt/data/yurui/models/pi05_wine_hybrid_17500 \
  wine_hybrid
```

The launcher refuses a missing local checkpoint path. The large models remain
read-only under `/mnt/data/yurui`; small writable OpenPI and uv runtime caches
default to `$HOME/.cache/fr3_pi05`. Override `OPENPI_DATA_HOME` and
`UV_CACHE_DIR` if a larger writable cache location is available. The launcher
uses OpenPI's existing `.venv/bin/python` directly when it already contains
`msgpack` and `pyzmq`; otherwise an isolated `uv --with pyzmq` overlay supplies
only the transport dependency. Neither path changes OpenPI's lock file or
downloads model weights.

All server profiles perform one synthetic inference before opening the ZMQ
endpoint. The wine warm-up uses its `(3, 7)/(3,)` history shapes. This pays the JAX
compilation cost before a robot observation can be accepted; `warmup_ms` and
the proprio contract are included in server metadata.

The normal mode has this workstation connect to the configured GPU address. If
a deployment filters that direction but permits GPU-to-workstation TCP, set
`pi05.zmq_mode = "bind"` in `config.toml` and start the GPU process with an
outbound endpoint:

```bash
CUDA_VISIBLE_DEVICES=1 PORT=8000 CONNECT_ENDPOINT=tcp://10.34.97.197:8000 \
  bash fr3_pi05/remote/start_pi05_zmq_server.sh \
  /mnt/data/yurui/openpi \
  "$HOME/fr3_demo" \
  /mnt/data/yurui/.cache/openpi/openpi-assets/checkpoints/pi05_droid \
  pi05_droid
```

In reverse mode the request direction is unchanged: the workstation still
sends observations and the GPU still returns actions. Only which host initiates
the TCP connection changes.

After a foreground smoke test succeeds, a supervised persistent session can use:

```bash
tmux new-session -d -s pi05_droid \
  "CUDA_VISIBLE_DEVICES=1 PORT=8000 bash $HOME/fr3_demo/fr3_pi05/remote/start_pi05_zmq_server.sh /mnt/data/yurui/openpi $HOME/fr3_demo /mnt/data/yurui/.cache/openpi/openpi-assets/checkpoints/pi05_droid pi05_droid"
tmux capture-pane -pt pi05_droid
```

If an approved SSH path becomes available later, the stock OpenPI WebSocket
launcher remains at `fr3_pi05/remote/start_pi05_server.sh`; select it locally
with `pi05.transport = "websocket"`.

## 2. Workstation

Install the OpenPI-protocol client, camera support, and this package:

```bash
cd /home/u0161364/fr3_demo
source .venv/bin/activate
python -m pip install -e '.[pi05]'
source /opt/ros/humble/setup.bash
```

All transport, direction, host, port, camera, Bamboo, rate, and safety values live in the
shared `config.toml` under `[pi05]`. The default is `transport = "zmq"`, host
`172.16.0.30`, and checkpoint `pi05_droid`. Ports are selected automatically:
8000 for `pi05_droid`, 8001 for `custom_droid`, and 8002 for `wine_hybrid`. With the selected GPU ZMQ server
ready, first run the network-only metadata check. It does not open the cameras
or Bamboo:

```bash
fr3-pi05-check --server-only
```

Select the custom endpoint persistently by changing `pi05.checkpoint` in
`config.toml`, or for one command with:

```bash
fr3-pi05-check --server-only --checkpoint custom_droid
```

For wine, select `--checkpoint wine_hybrid`. The client verifies the server's
exact history, model, task, and 16-step action metadata before the first
inference or policy action, then collects 76
samples at 15 Hz (five seconds of history) before its first inference. It
continues sampling at 15 Hz while inference is pending and throughout the
rollout:

```bash
fr3-pi05-check --checkpoint wine_hybrid --prompt "pour gin into the jigger"
fr3-pi05-run --checkpoint wine_hybrid --execute --prompt "pour gin into the jigger"
```

The only accepted wine prompts are `pour gin into the jigger` and
`pour lillet into the jigger`, exactly as annotated during training. Wine
execution uses the collection envelope (0.35 rad/s, 1.5 rad/s², 0.08 rad joint
margin). When its gripper target changes, arm velocity is held at zero until
the blocking Bamboo gripper operation completes; the remaining chunk is then
discarded and replanned from measured state, matching demonstration timing.

Then run the non-moving end-to-end check:

```bash
fr3-pi05-check --prompt "pick up the red block"
```

It opens both cameras, reads Bamboo and the gripper, makes one real GPU inference,
checks the complete selected-checkpoint action horizon, and republishes a three-second RViz
preview so the camera and marker displays can subscribe. It never starts Bamboo
streaming. To validate only local configuration and kinematics:

```bash
fr3-pi05-check --offline --no-rviz --prompt test
```

Next run a paced inference-only rollout. Cameras, live joint state, current
kinematic robot, final predicted robot, and predicted EEF path are visible in
RViz. By default, `fr3-pi05-run` first moves the arm to the configured home pose
using a temporary watchdog-protected Bamboo stream, then opens the gripper. The
gripper opening is blocking, so inference starts only after Bamboo confirms it.
After startup, no policy action reaches Bamboo unless `--execute` is present:

```bash
fr3-pi05-run --prompt "pick up the red block"
```

Automatic homing uses `[home].speed` and `[home].timeout` from `config.toml` and
the shared home pose `[-0.047, -0.735, -0.028, -2.278, -0.007, 1.578, 0.031]`.
Use `--no-home` only when the arm has already been deliberately positioned.
Use `--no-gripper` to disable both startup and policy gripper control.
`fr3-pi05-check` remains strictly non-moving: it neither homes nor opens the
gripper.

Only after both checks look correct, enable motion:

```bash
fr3-pi05-run --execute --prompt "pick up the red block"
```

For a different language instruction every run, omit `--prompt`:

```bash
source /opt/ros/humble/setup.bash
cd /home/u0161364/fr3_demo
source .venv/bin/activate
fr3-pi05-run --checkpoint pi05_droid --execute
```

The program first asks `Language instruction:` and then arms automatically once
the initial inference and safety preview pass. For a repeatable scripted
instruction, pass it explicitly:

```bash
fr3-pi05-run --checkpoint pi05_droid --execute \
  --prompt "place the object into the container"
```

There is no second typed confirmation: `--execute` is the explicit motion gate.
The repository configuration sets `max_rollout_steps = 0`, so execution has no
fixed duration and continues until Back, Ctrl+C, or a safety stop. Set it to a
positive number in `config.toml`, or pass `--max-steps N`, to restore a limit.
During motion, Back on the joystick, Ctrl+C, a stale camera/chunk, invalid
action, workspace violation, joint-margin violation, gripper failure, or
joystick disconnect stops the arm command. The Bamboo watchdog remains the
final software brake. Keep the physical E-stop in hand; this client does not
provide scene or self-collision checking.

## RViz topics

- `/fr3_pi05/exterior_image_left`
- `/fr3_pi05/wrist_image`
- `/fr3_pi05/markers`

Blue is the measured FR3 kinematic chain, orange is the predicted final chain,
and green is the predicted EEF trajectory. Use `--rviz-publish-only` when RViz
is already running, or `--no-rviz` on a headless workstation.
