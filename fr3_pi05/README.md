# FR3 pi0.5 deployment

The robot workstation owns Bamboo, cameras, RViz, and guarded execution. The
GPU machine (`172.16.0.30`) only serves policy inference over ZMQ.

## GPU: one-time setup

```bash
cd ~/fr3_demo
git pull --ff-only
bash fr3_pi05/remote/prepare_wine_openpi.sh
```

The preparation command creates and validates `~/openpi_wine`. It does not
download checkpoint weights.

## GPU: deploy any newly trained dataset

For every checkpoint trained with the existing Wine contract, run one command:

```bash
bash ~/fr3_demo/fr3_pi05/remote/start_checkpoint.sh \
  /mnt/data/yurui/models/NEW_CHECKPOINT
```

The launcher automatically discovers the single
`assets/<huggingface-dataset>/norm_stats.json`, loads those statistics, warms up
the model, and listens on port 8002. A changed Hugging Face dataset name needs
no code edit and no copied loader.

Prompts are unrestricted by default. The normal case needs no dataset name,
loader path, model profile, port, or task-list argument.

For a checkpoint that was trained with the L515 as its third visual input,
change one setting in `config.toml` on the GPU checkout:

```toml
[pi05]
use_external2 = true
```

Then use the same short launch command:

```bash
bash ~/fr3_demo/fr3_pi05/remote/start_checkpoint.sh \
  /mnt/data/yurui/models/NEW_CHECKPOINT
```

Keep `use_external2 = false` for existing two-camera checkpoints. Enabling it
changes the model input contract: `exterior_image_2_left` fills the third image
slot and its mask becomes valid. The client checks the advertised contract and
refuses to run if the L515 is unavailable. No training-owned file is modified by
this deployment switch; use the value specified by the checkpoint owner.

Stop the server with `Ctrl+C` before loading another large model.

These commands assume the Wine contract is otherwise unchanged: 3-frame
proprioception, 16×8 actions, and the same OpenPI model variants. A new dataset,
normalization asset, checkpoint step, prompt, or optional L515 input needs no
new profile.

Other existing profiles remain available:

```bash
bash fr3_pi05/remote/start_checkpoint.sh pi05_droid
bash fr3_pi05/remote/start_checkpoint.sh custom_droid
```

## Robot workstation

One-time installation:

```bash
cd ~/fr3_demo
source .venv/bin/activate
python -m pip install -e '.[pi05]'
source /opt/ros/humble/setup.bash
```

`config.toml` selects `wine_hybrid` and port 8002. It does not contain the GPU
checkpoint or dataset path, so no local edit is needed for another Wine
checkpoint.

```bash
# Network and model-contract check; no robot or cameras.
fr3-pi05-check --server-only

# One inference and RViz preview; never moves.
fr3-pi05-check --prompt "the exact instruction used in training"

# Home, open the gripper, ask for the language instruction, and execute.
fr3-pi05-run --execute
```

Useful options:

```bash
--debug-chunks       # print 16×8 actions and gripper decisions
--require-joystick   # require Back-button abort; joystick is otherwise optional
--no-rviz            # headless run
```

The Wine client samples state at current, t-45, and t-75 frames (0, 3, and 5
seconds at 15 Hz), then executes all 16 action rows. Gripper values above
`pi05.gripper_threshold` open; lower values close. A gripper transition stops
the arm, discards the remaining chunk, waits for Bamboo, and replans.

Exterior 1 and wrist cameras are required. The L515 serial/mode is configured
under `[cameras]` in `config.toml` and defaults to `960x540@30`. It is recorded
whenever connected; if absent, collection and ordinary two-camera inference
continue. It becomes required only when the GPU model is launched with
`pi05.use_external2 = true`.

Use Back when available, `Ctrl+C`, or the physical E-stop to stop. The Bamboo
watchdog and joint/workspace checks remain active, but the client does not
provide scene or self-collision checking.

RViz topics are `/fr3_pi05/exterior_image_left`,
`/fr3_pi05/exterior_image_2_left`, `/fr3_pi05/wrist_image`, and
`/fr3_pi05/markers`.
