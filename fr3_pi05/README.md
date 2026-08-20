# FR3 pi0.5 DROID deployment

This package connects the current FR3 workstation to Physical Intelligence's
official `pi05_droid` policy. The workstation owns the RealSense devices and
Bamboo robot connection; `10.38.32.253` only performs GPU inference. The default
transport is direct ZMQ because these two machines have routed application
connectivity even though the managed SSH gateway denies forwarding to the GPU.
The official OpenPI WebSocket transport remains selectable as a fallback.

The exact request fields are:

- `observation/exterior_image_1_left`: exterior RGB, padded to 224x224;
- `observation/wrist_image_left`: wrist RGB, padded to 224x224;
- `observation/joint_position`: seven measured FR3 joint positions;
- `observation/gripper_position`: one normalized opening value;
- `prompt`: the operator's language instruction.

The client accepts the current 15x8 checkpoint response, executes only the
configured first eight actions, and interprets each row as seven joint
velocities plus one gripper-position target. It runs at the DROID dataset rate
of 15 Hz and prefetches the next chunk while the current chunk is executing.

## 1. GPU server from its local console

The KU Leuven certificate authenticates successfully on `jump-l`, but the
gateway policy currently denies forwarding to `10.38.32.253`. SSH is therefore
not needed for inference itself. From a terminal physically on the GPU machine,
put the repository under the requested `fr3_pi05` directory:

```bash
mkdir -p ~/fr3_pi05
cd ~/fr3_pi05
git clone https://github.com/fitz0401/fr3_demo.git
cd fr3_demo
bash fr3_pi05/remote/audit_host.sh
```

Send the audit output back before installing or downloading anything. It checks
both GPUs, active GPU processes, disk space, `uv`/Conda, candidate OpenPI trees,
checkpoint caches, and port 8000 without changing the host.

When an existing OpenPI checkout and checkpoint are confirmed, choose the idle
A6000 index reported by `nvidia-smi` and start the direct ZMQ wrapper:

```bash
CUDA_VISIBLE_DEVICES=<A6000_INDEX> \
  bash fr3_pi05/remote/start_pi05_zmq_server.sh \
  /absolute/path/to/openpi \
  "$HOME/fr3_pi05/fr3_demo"
```

The wrapper loads the official `pi05_droid` configuration and
`gs://openpi-assets/checkpoints/pi05_droid`, then binds a ZMQ request/reply
server on `0.0.0.0:8000`. `uv run --with pyzmq` supplies only the small transport
dependency without changing OpenPI's lock file. The first checkpoint load can
be much slower if its cache is incomplete.

After a foreground smoke test succeeds, a supervised persistent session can use:

```bash
tmux new-session -d -s pi05_droid \
  "CUDA_VISIBLE_DEVICES=<A6000_INDEX> bash $HOME/fr3_pi05/fr3_demo/fr3_pi05/remote/start_pi05_zmq_server.sh /path/to/openpi $HOME/fr3_pi05/fr3_demo"
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

All transport, host, port, camera, Bamboo, rate, and safety values live in the
shared `config.toml` under `[pi05]`. The default is `transport = "zmq"`, host
`10.38.32.253`, port `8000`. With the GPU ZMQ server listening, first run the
network-only metadata check. It does not open the cameras or Bamboo:

```bash
fr3-pi05-check --server-only
```

Then run the non-moving end-to-end check:

```bash
fr3-pi05-check --prompt "pick up the red block"
```

It opens both cameras, reads Bamboo and the gripper, makes one real GPU inference,
checks the entire eight-action prefix, and opens RViz. It never starts Bamboo
streaming. To validate only local configuration and kinematics:

```bash
fr3-pi05-check --offline --no-rviz --prompt test
```

Next run a paced inference-only rollout. Cameras, live joint state, current
kinematic robot, final predicted robot, and predicted EEF path are visible in
RViz, but no action reaches Bamboo:

```bash
fr3-pi05-run --prompt "pick up the red block"
```

Only after both checks look correct, enable motion:

```bash
fr3-pi05-run --execute --prompt "pick up the red block"
```

The program requires typing `EXECUTE`. During motion, Back on the joystick,
Ctrl+C, a stale camera/chunk, invalid action, workspace violation, joint-margin
violation, gripper failure, or joystick disconnect stops the arm command. The
Bamboo watchdog remains the final software brake. Keep the physical E-stop in
hand; this client does not provide scene or self-collision checking.

## RViz topics

- `/fr3_pi05/exterior_image_left`
- `/fr3_pi05/wrist_image`
- `/fr3_pi05/markers`

Blue is the measured FR3 kinematic chain, orange is the predicted final chain,
and green is the predicted EEF trajectory. Use `--rviz-publish-only` when RViz
is already running, or `--no-rviz` on a headless workstation.
