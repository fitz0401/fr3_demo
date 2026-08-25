# FR3 pi0.5 deployment

The GPU machine (`172.16.0.30`) runs inference. The robot workstation owns
Bamboo, the cameras, RViz, and robot execution.

This guide assumes `~/fr3_demo` and `~/openpi_wine` are already installed on
the GPU machine. `prepare_wine_openpi.sh` was only needed to create that OpenPI
environment the first time.

## Start a new checkpoint on the GPU

Stop the previous server with `Ctrl+C`, then run:

```bash
bash ~/fr3_demo/fr3_pi05/remote/start_checkpoint.sh \
  /mnt/data/yurui/models/NEW_CHECKPOINT
```

That is the complete command. The launcher automatically:

- uses the Wine profile and port `8002`;
- finds the checkpoint's Hugging Face dataset normalization statistics;
- loads and warms up the model;
- detects LoRA (`lora`/`hybrid` in the directory name) versus full
  (`aefull`/`full` in the directory name);
- exposes the model at `172.16.0.30:8002`.

For a checkpoint trained with the L515 image, set this on the GPU machine in
`~/fr3_demo/config.toml` before starting it:

```toml
[pi05]
use_external2 = true
```

For a checkpoint trained without the L515 image, use:

```toml
use_external2 = false
```

This switch only changes deployment behavior. It does not modify training code.

## Run it on the robot workstation

The command does not change when the GPU checkpoint changes. The checkpoint
path, dataset name, and normalization statistics stay on the GPU machine.

First, optionally verify the new server without opening Bamboo or cameras:

```bash
fr3-pi05-check --server-only
```

Then start the rollout:

```bash
source .venv/bin/activate
source /opt/ros/humble/setup.bash
fr3-pi05-run --execute
```

`fr3-pi05-run` asks for the language instruction interactively. You may instead
provide it directly:

```bash
fr3-pi05-run --execute --prompt "the instruction used in training"
```

No local configuration change is needed for another Wine checkpoint as long as
the GPU server remains on port `8002`. If the server advertises L515 input, the
workstation automatically sends `exterior_image_2_left` and refuses execution
when that camera is unavailable.

Exterior 1 and wrist cameras are always required. Use Back when available,
`Ctrl+C`, or the physical E-stop to stop execution.
