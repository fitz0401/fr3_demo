# FR3 pi0.5 DROID deployment

This package connects the current FR3 workstation to Physical Intelligence's
official `pi05_droid` policy. The workstation owns the RealSense devices and
Bamboo robot connection; `10.38.32.253` only performs GPU inference over the
OpenPI WebSocket protocol.

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

## 1. GPU server

SSH currently routes through `jump-l.icts.kuleuven.be`. This managed KU Leuven
jump host does not accept a permanently registered personal public key: it
requires a short-lived MFA SSH certificate. The `kmk` tool is already installed
and configured for `u0161364` on this workstation. Activate and verify a
certificate with:

```bash
kmk renew
kmk check
ssh-add -l
ssh 10.38.32.253
```

`kmk renew` opens the KU Leuven MFA flow. Certificates are temporary, so renew
again when `kmk check` reports that the certificate expired. If authentication
then succeeds on `jump-l` but fails specifically on `10.38.32.253`, the GPU host
itself must either trust KU Leuven SSH certificates or have the workstation's
public key added to `~/.ssh/authorized_keys`; a plain key is never installed on
the managed jump host.

After access works, copy and run the read-only inventory:

```bash
scp fr3_pi05/remote/audit_host.sh 10.38.32.253:/tmp/fr3_pi05_audit.sh
ssh 10.38.32.253 'bash /tmp/fr3_pi05_audit.sh'
```

Do not start a download or installation until that output confirms whether an
OpenPI checkout and `pi05_droid` cache already exist. With an existing checkout,
copy the launcher and choose the idle A6000 index reported by `nvidia-smi`:

```bash
CUDA_VISIBLE_DEVICES=<A6000_INDEX> \
  fr3_pi05/remote/start_pi05_server.sh /absolute/path/to/openpi
```

The launcher runs this official command:

```bash
uv run scripts/serve_policy.py --env DROID --port 8000
```

For a persistent supervised session after a foreground smoke test succeeds:

```bash
tmux new-session -d -s pi05_droid \
  "CUDA_VISIBLE_DEVICES=<A6000_INDEX> /path/to/start_pi05_server.sh /path/to/openpi"
tmux capture-pane -pt pi05_droid
```

The checkpoint selected by OpenPI is
`gs://openpi-assets/checkpoints/pi05_droid`. The first launch may populate the
local cache and can therefore be much slower than later starts.

If TCP port 8000 remains blocked from the workstation, keep this in a second
terminal:

```bash
fr3_pi05/remote/open_inference_tunnel.sh
```

Then set `pi05.server_host = "127.0.0.1"` in `config.toml`. The client traffic
will travel through the authenticated SSH jump instead of exposing the policy
port to the network.

## 2. Workstation

Install the OpenPI-protocol client, camera support, and this package:

```bash
cd /home/u0161364/fr3_demo
source .venv/bin/activate
python -m pip install -e '.[pi05]'
source /opt/ros/humble/setup.bash
```

All host, port, camera, Bamboo, rate, and safety values live in the shared
`config.toml` under `[pi05]`. First run the non-moving end-to-end check:

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
