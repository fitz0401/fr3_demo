# Deploying `pi05_wine_hybrid` step 17500

Fine-tuned from `pi05_base` on `fitz0401/franka_pour_wine`: 40 episodes, 57,564 frames, 15 fps, FR3,
**two tasks** -- `"pour lillet into the jigger"` and `"pour gin into the jigger"`. Trained on
Leonardo (jobs 53441114 + 53732122), 30k steps; this is the best retained checkpoint by held-out
action MAE.

## Contents

| path | what |
| --- | --- |
| `params/` | orbax weights, bf16, 6.0 GiB. LoRA adapters are **not** merged. |
| `assets/fitz0401/franka_pour_wine/norm_stats.json` | quantile stats for the **24-dim** state and the 8-dim actions. Loaded from here automatically. |
| `serve_wine.py` | stand-alone websocket server; builds the TrainConfig inline. |
| `deploy.patch` | **mandatory** patch against upstream openpi (adds the `gemma_2b_lora_r32` variant). |

## Setup

```bash
git clone https://github.com/Physical-Intelligence/openpi.git && cd openpi
git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
git apply /path/to/pi05_wine_hybrid_17500/deploy.patch
uv sync
.venv/bin/python /path/to/pi05_wine_hybrid_17500/serve_wine.py \
    --checkpoint-dir /path/to/pi05_wine_hybrid_17500 \
    --action-expert-variant gemma_300m_lora --self-test
```

`--action-expert-variant gemma_300m_lora` is the default and is correct for **this** checkpoint. The
sibling package `pi05_wine_aefull_17500` needs `gemma_300m` instead: its action expert was trained in
full, so the LoRA layers do not exist in it and the parameter tree has 61 groups rather than 71.
Passing the wrong value raises at load time. Everything else -- the request format, the norm stats,
the patch -- is identical between the two packages, so switching models means pointing at the other
directory and changing that one flag. No client change.

Then drop `--self-test` and add `--port 8000` to serve. Invoke the interpreter directly rather than
through `uv run`.

## Request / response contract

**Two things changed from the custom_droid model. A client written against that one will be wrong
in ways that do not raise.**

| key | shape | notes |
| --- | --- | --- |
| `observation/exterior_image_1_left` | `(H, W, 3)` uint8 | resized to 224x224 internally |
| `observation/wrist_image_left` | `(H, W, 3)` uint8 | match the training reference; this rig rotates raw wrist frames 180° |
| `observation/joint_position` | **`(3, 7)`** float | **current first**, then the t-45 and t-75 frames |
| `observation/gripper_position` | **`(3,)`** or `(3, 1)` float | same ordering |
| `prompt` | str | **mandatory**, one of the two task strings |

Response: `actions`, shape `(16, 8)`, dtype **float64**. Columns 0-6 are joint **velocity** commands
in rad/s, column 7 is the gripper position in [0, 1]. 16 steps at 15 fps = 1.07 s per chunk.

**The state history is the client's job.** Keep a ring buffer of proprioception and send frames t,
t-45 and t-75 (3 s and 5 s back at 15 fps). For a lag before episode start, repeat the startup
sample; LeRobot clamps negative indices to the episode's first frame. The FR3 client instead fills
five seconds of stationary home-state history before its first request, so every lag is available.
Sending a single unstacked frame raises rather than silently meaning something else.

**The prompt is mandatory by design.** The two tasks share a scene and differ only in the
instruction, so a default would quietly pour the wrong bottle. A request without a prompt is
rejected.

## Verified on Leonardo (job 53845095)

Perturbing one input at a time with the sampling noise held fixed:

| changed | max abs delta in the action chunk |
| --- | --- |
| lagged state frames only (present bit-identical) | 7.32e-2 |
| whole state | 1.52e-1 |
| sampling noise | 1.40e-1 |
| image | 2.80e-2 |
| prompt (lillet <-> gin) | 2.17e-2 |
| nothing | 0 |

Latency: **90.9 ms** median per inference on one A100, after a one-off JIT compile of roughly 20 s.
Warm up with a dummy request before enabling the controller.

## Things that will bite

**Match the training orientation of the wrist camera; on this rig that means rotating the live feed
180 degrees.** Confirmed on the robot on 2026-08-24, and visible in `reference_wrist.png`, a frame
taken straight from the training set: the bottle label reads upright and the black gripper fingers
sit along the **bottom** edge. `reference_exterior.png` is the matching exterior view (arm entering
from the left, jigger on the table, blinds behind).

Compare your live feed against those two images before trusting any behaviour. If the gripper
fingers appear at the top and the label is upside down, rotate by 180 degrees. An earlier revision
of this document said the opposite -- that was inferred from the previous custom_droid rig rather
than checked against this dataset, and it was wrong.

**The policy leans on proprioception far more than on vision.** Changing the state moves the output
5x more than changing the image does, and the lagged frames alone move it 2.6x more than the image.
Perturbation magnitudes across modalities are not strictly comparable, but the direction is clear.
The risk this points at is a policy that extrapolates its own trajectory instead of looking at the
scene. Worth probing directly on the robot: move the bottle or the jigger between runs and see
whether the motion adapts or replays.

**Offline accuracy is weak in absolute terms.** Held-out action MAE is 0.0588 rad/s against a mean
action magnitude of 0.0716, i.e. only ~18% better than a policy that outputs zero. A larger model
does no better: an otherwise identical run with the action expert fully trained (900.6M trainable
vs 494.8M) plateaus at the same 0.0588, so this is a data limit rather than a capacity limit. MAE
also punishes picking a different but equally valid trajectory, and the previous custom_droid model
looked similarly unimpressive offline yet behaved correctly on the robot. Treat real-robot success
as the deciding measure.

**`exterior_image_2_left` is dead in this dataset** (all black, verified) and the model never sees
it. Only two cameras matter.

**Output is float64.** Cast if the controller assumes float32.

**Sharing a GPU with another process** requires `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.25` or similar, or JAX will take the whole card.

## Model card

pi0.5 with `discrete_state_input=True`: proprioception is discretised into 256 bins and carried as
text in the prompt, 24 dims = [q, grip] at t, t-45 and t-75. LoRA r32/alpha32 on the VLM and on the
action expert; SigLIP and the action/time projections fully trained; 494.8M trainable, 2.936B
frozen. Actions are joint velocities, so no delta transform. Horizon 16.
