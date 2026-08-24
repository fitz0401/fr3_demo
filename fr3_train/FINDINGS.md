> Running record of the FR3 post-training work, kept in commit order. It starts with the first
> dataset (`custom_droid`, teapot into a cup) and continues into `franka_pour_wine`, so the early
> sections describe a rig and a recipe that later sections revise. Where a conclusion was later
> overturned the correction is stated in place rather than by editing history.

# B0 custom_droid Post-Training Manifest

Objective: `pi05_base + fitz0401/custom_droid -> pi_custom_droid`, upstream OpenPI JAX
training only, using the same hybrid recipe that passed the Cocktail tiny-batch test.

## Fixed Sources

- OpenPI upstream commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac` (+ local B0 patches)
- Base checkpoint: `gs://openpi-assets/checkpoints/pi05_base/params` (cached under `$OPENPI_DATA_HOME`)
- Dataset: `huggingface.co/datasets/fitz0401/custom_droid` @ `0cdc4e7db89b48f1cd124c5fd40db070512898e4`
- Local dataset root: `/leonardo_work/EUHPC_D35_005/datasets/lerobot/fitz0401/custom_droid`
- No JAX to PyTorch conversion.

## Dataset Facts (from `meta/info.json` + parquet inspection)

| field | value |
| --- | --- |
| LeRobot codebase version | v2.1 |
| robot | fr3 |
| episodes / frames | 20 / 15,367 (min 504, max 1,354 frames per episode) |
| fps | 15 |
| language task | `pour the teapot into the cup` (single task) |
| cameras | `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left`, PNG 180x320x3 |
| proprio | `joint_position` (7) + `gripper_position` (scalar) -> 8-dim state |
| actions | 8-dim: 7 joint velocities in ~[-0.35, 0.35] + gripper in [0, 1] |
| absolute vs delta | joint **velocity** actions, so no delta transform is applied |
| on-disk size | 2.4 GB (images inline in parquet, no videos) |

Two data caveats found and handled at download time:

1. `exterior_image_2_left` is an all-black placeholder in every frame. The pi0.5 branch of
   `DroidInputs` only consumes `exterior_image_1_left` (-> `base_0_rgb`) and
   `wrist_image_left` (-> `left_wrist_0_rgb`), and masks `right_wrist_0_rgb`, so the blank
   camera never reaches the model.
2. `meta/tasks.jsonl` task 1 carried a stray terminal escape (`"...cup\x1b[D"`), which would
   have produced a second, different prompt for episode 1 (842 frames). Stripped in place;
   originals kept as `meta/tasks.jsonl.orig` and `meta/episodes.jsonl.orig`.

3. 27.2% of frames carry an exactly zero joint-velocity action (1.6% leading idle,
   6.9% trailing idle, the rest mid-episode teleop pauses). Left as-is, but it is the
   first thing to look at if the policy learns to stall.

A further issue is an environment mismatch rather than a data defect: the parquet footers were
written by `datasets>=4.0`, whose `List` feature type the pinned `datasets==3.6.0` cannot
parse (`ValueError: Feature type 'List' not found`). `examples/custom_droid/fix_lerobot_v4_features.py`
rewrites only the `huggingface` schema metadata (`List` -> the structurally identical
`Sequence`); column data is untouched. Re-run it after any fresh download.

## Config

`pi05_custom_droid_hybrid` (in `src/openpi/training/config.py`), reusing the Cocktail recipe:

- `Pi0Config(pi05=True, action_dim=32, action_horizon=16, discrete_state_input=False)`
- SigLIP vision encoder: trainable
- VLM/Gemma backbone: LoRA `gemma_2b_lora_r32`, rank 32, alpha 32
- Action expert: LoRA `gemma_300m_lora`, rank 32, alpha 32
- state/action/time projections: trainable
- ~495M trainable / ~2.94B frozen parameters
- optimizer AdamW, grad clip 1.0; cosine warmup 1,000 steps to 5e-5 then constant
- `ema_decay=None`, global batch 64, `fsdp_devices=1` (4-way data parallel)
- data: `LeRobotDROIDDataConfig` (upstream), `prompt_from_task=True`
- norm stats: computed from this dataset into
  `assets/pi05_custom_droid_hybrid/fitz0401/custom_droid/norm_stats.json` (quantile norm)

Alternative not taken: `pi05_droid` weights + the original DROID norm stats
(upstream `pi05_droid_finetune`). That is the domain-matched starting point for
DROID-schema data; this run deliberately keeps the B0 line `pi05_base -> D`.

Training length: 15,000 steps at batch 64 = ~62 epochs over 15,367 frames; checkpoints
every 2,500 steps so an earlier checkpoint can be picked if it overfits.

## Run Order (Leonardo, 1 node x 4 A100 64GB)

```bash
# one batch through the input pipeline (login node is fine, CPU only)
JAX_PLATFORMS=cpu bash scripts/launch_leonardo_4gpu_custom_droid.sh preflight

# norm stats + 20-step probe + the 15k-step run, all in one allocation
sbatch scripts/sbatch_leonardo_custom_droid.sbatch

# resume after a walltime kill
sbatch --export=ALL,RESUME=true scripts/sbatch_leonardo_custom_droid.sbatch full
```

Compute nodes have no outbound network, so the dataset, `pi05_base` params and the uv
environment must all be on disk first; the launch script sets `HF_HUB_OFFLINE=1`.

OOM fallback order: `FSDP_DEVICES=1` -> `2` -> `4`.

## Gates

Probe (20 steps) must show, as the Cocktail tiny test did:

- loss finite and decreasing
- `grad_norm_siglip`, `grad_norm_vlm_lora`, `grad_norm_action_expert_lora`,
  `grad_norm_projection` all nonzero
- `trainable_parameters.json` written with the expected LoRA targets/ranks
- checkpoint save works

Reference throughput from the Cocktail tiny run at the same topology: 1.62 s/step,
~39.5 samples/s, so 15k steps is roughly 7 h.

## Validation and Checkpoint Retention (added 2026-08-19)

Upstream OpenPI trains without any validation loop, and the 15k run above therefore has no
held-out signal: all 20 episodes were used for training. Three pieces were added.

### 1. Offline evaluation of saved checkpoints

`scripts/eval_offline.py` restores each checkpoint's params and runs forward passes only:

| metric | meaning |
| --- | --- |
| `val_loss` | flow-matching denoising loss, `train=False`, **fixed rng** (seed + batch index) so different checkpoints see identical noise/timestep draws |
| `action_mse_phys`, `action_mae_phys` | error of actions actually sampled by the flow solver (10 denoise steps) vs the demonstration, in rad/s |
| `gripper_mae` | same, for the gripper dimension in [0, 1] |
| `mae_per_joint`, `mae_per_horizon` | per-dimension and per-horizon-step breakdown |
| `pred_stall_frac` vs `gt_stall_frac` | fraction of timesteps with joint velocity below 1e-3. Targets the known data caveat: 27.2% of frames carry an exactly-zero action, so a policy that learns to freeze can still score a respectable loss. `pred >> gt` is that failure. |

Run it with `sbatch scripts/sbatch_leonardo_eval.sbatch` (1 node x 4 A100, ~5 min per checkpoint).

**Interpretation caveat.** Evaluating the `b0_custom_droid_full_15k` checkpoints on any episode of
this dataset is *in-sample*, because that run trained on all 20. In-sample loss falls monotonically,
so it measures fit, not generalization, and cannot be used to select a best checkpoint. Those
numbers are a sanity check on whether the policy learned sane behaviour, nothing more.

### 2. In-loop validation

`TrainConfig` gained `val_episodes`, `eval_interval`, `eval_num_batches` and `eval_seed`. When set,
`scripts/train.py` builds a second data loader on the held-out episodes and runs `eval_step`
(forward only) every `eval_interval` steps and always immediately before a checkpoint save, so
every checkpoint carries a metric. The split is by **whole episodes**: at 15 fps neighbouring frames
are near-duplicates, so a frame-level split would leak validation data into training. The validation
indices are permuted with a fixed seed (`data_loader.VAL_SHUFFLE_SEED`), because in episode order
the first N batches would all come from the first held-out episode.

### 3. Best-checkpoint retention

With `keep_best_checkpoint=True`, `checkpoints.prune_to_final_and_best` runs after every save and
keeps exactly two checkpoints: **the most recent** (so a crashed run resumes where it stopped rather
than rolling back) and **the best `val_loss`**. Rankings live in `eval_history.json` in the
checkpoint directory -- a plain JSON sidecar rather than orbax metadata, so it survives resumes and
can be inspected by hand. orbax's own `max_to_keep`/`keep_period` are disabled in this mode to keep
the two retention policies from racing.

### Config

`pi05_custom_droid_hybrid_val` is the config for any *new* custom_droid run. It is identical to
`pi05_custom_droid_hybrid` except for: `val_episodes=(4, 12, 17)` (990 + 691 + 630 = 2,311 of 15,367
frames, 15.0%, leaving 17 episodes for training), `eval_interval=500`, `keep_best_checkpoint=True`,
and `keep_period=None`. Norm stats are deliberately reused from `pi05_custom_droid_hybrid` via
`AssetsConfig` rather than recomputed, so the two runs stay comparable; the resulting leakage is
limited to the quantiles of a single-task dataset shifting by ~15% of their support.

`pi05_custom_droid_hybrid` itself is left untouched, so the finished 15k run remains reproducible.

## Results of the 15k Run (jobs 52917533 / 52934475, 2026-08-19)

Evaluated on episodes 4/12/17, 16 batches x 64 = 1,024 windows, 10 denoise steps. **In-sample**
(this run trained on all 20 episodes), so these measure fit, not generalization.

| step | val_loss | MAE rad/s | gripper MAE | creep p50 | creep p90 | creep_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 5000 | 0.01370 | 0.02937 | 0.02167 | 0.00517 | 0.1813 | 0.0257 |
| 10000 | 0.00917 | 0.01998 | **0.01627** | 0.00275 | 0.1374 | 0.0134 |
| 14999 | **0.00753** | **0.01600** | 0.01823 | 0.00290 | 0.1081 | 0.0136 |

Joint MAE at 14999 is 0.0160 rad/s against a mean action magnitude of 0.0610 rad/s (26%). Error is
flat across the chunk (0.0148 at h=0 to 0.0174 at h=15 -- no long-horizon collapse) and largest on
the two wrist joints (0.0248, 0.0198) and smallest on joint 1 (0.0074). The whole eval takes ~7 min
for three checkpoints on one 4-GPU node.

Retained checkpoints: **14999** (final, best val_loss/MAE) and **10000** (best gripper MAE).
5000 was deleted. Metrics are recorded in `eval_history.json` in the checkpoint directory.

### The zero-action frames are genuine pauses, not bad labels

Investigated because 27.2% of frames carry an exactly-zero joint-velocity action, which raised the
risk of a policy that learns to freeze. Measuring the arm's *actual* velocity from `joint_position`
differences settles it:

* On non-zero-command frames the robot tracks the command almost exactly (corr 0.894, median
  actual/commanded 0.997), so the state derivative is a valid check on the labels.
* On zero-command frames the median actual speed is **0.0002 rad/s** against 0.2049 rad/s on
  normal frames -- the arm really is stationary. Only 2.9% of zero frames show motion above the
  25th percentile of normal speed.
* Cause: teleoperation is **velocity**-controlled, so every operator hesitation is logged as an
  exact zero rather than as a near-duplicate position. Placement: 25.3% of the zeros are trailing
  dead air (~3.5 s per episode after the task ends), 5.9% leading, 68.7% mid-episode. Within an
  episode the zeros peak at the end (25.8% of them in the last decile) and in the 20-40% approach
  and alignment phase (23.5%). The median pause is only 3 frames (0.2 s).
* Not gripper actuation: the gripper state changes in only 1.1% of zero-command frames.
* The one real defect is small: 188 runs of 1-2 frames (279 frames, 1.8% of the dataset) where the
  arm keeps moving at normal speed through a zero -- these look like dropped teleop samples.
* The model predicts 16-step chunks, which dilutes the problem: only **10.5%** of chunks are
  entirely zero (vs 27.2% of frames), and 65% of those come from the trailing dead air.

**The trained policy shows neither failure mode.** It does not freeze (predicted motion magnitude
0.0611 vs 0.0610 ground truth) and it does hold still when the demonstration does (median predicted
speed during a demonstrated hold is 0.0029 rad/s = 1.4% of normal motion). The residual is a p90
tail of 0.108 rad/s on held steps that narrows steadily with training (0.181 -> 0.137 -> 0.108),
plausibly the pause boundaries inside a chunk -- not verified.

`pred_stall_frac` on its own is misleading here and should not be read as a stall measure: a
continuous-output flow-matching policy essentially never emits an exact zero, so the metric is near
zero by construction. `creep_ratio` is the one to read.

Consequence: **no data surgery is justified**. Cutting the trailing dead air would remove 65% of the
all-zero chunks and is the cheapest change available if a stall problem ever does appear, but
nothing in the current numbers calls for it.

## Real-robot Result and Recipe Changes for the Next Run (2026-08-20)

The 14999 checkpoint was deployed on the FR3 via the JAX websocket server and **behaves like the
demonstrations**, so the hybrid recipe itself is validated. Three changes were identified on the
robot; none of them are fixable in deployment, all require a re-run.

### 1. Feed proprioception

`discrete_state_input=False` leaves this pi0.5 model with no state pathway at all (verified in job
53195955: changing the joint configuration produced bit-identical actions). Set
`discrete_state_input=True` next time.

### 2. Prompt alignment

Same switch, second effect: with `discrete_state_input=True` the prompt becomes the pi0.5 format
`"Task: <text>, State: <256 bins>;\nAction: "`, which is what `pi05_base` was pretrained on. The
current run used the pi0 format `"<text>\n"`, so it was fine-tuning off the pretraining
distribution for no benefit. Items 1 and 2 are therefore one config change, not two.

### 3. Rotate the wrist images 180 degrees to match the DROID convention

Our wrist camera is mounted 180 degrees rotated relative to the DROID dataset's. The current run
trained on the images as collected, which is self-consistent -- training and the validated
deployment both use the unrotated feed -- but it means the wrist view never matched the orientation
the pretrained model saw, so none of that pretrained wrist-view structure was usable.

Next run: rotate `wrist_image_left` by 180 degrees in the dataset so it agrees with DROID. Then the
deployment must rotate the live feed too; the invariant is that training and inference apply the
same transform.

This also raises the base-checkpoint question the manifest deferred earlier: with DROID-aligned
wrist images, `pi05_droid` becomes the natural starting point instead of `pi05_base`. Note that
`pi05_droid` defaults to `action_horizon=15` and `discrete_state_input=True` -- the latter is
change 1 above, so the three changes converge on "use the standard DROID recipe".

### 4. Give the policy one frame of history

The policy has **no memory of any kind**. `Observation.images` is `dict[str, Float["*b h w c"]]` --
one frame per camera slot, no time axis -- and `delta_timestamps` is applied only to
`action_sequence_keys`, so the loader stacks *future actions* and never *past observations*. Each
`infer` call is independent: the repeat test in job 53195955 returned bit-identical actions for a
repeated input, which is only possible if nothing is carried across calls.

Consequence: the policy cannot distinguish "carrying the teapot towards the cup" from "carrying it
back after pouring" unless the single frame contains a visual cue (liquid level, teapot tilt).

**Plan: put the previous wrist frame into the unused third camera slot.** `right_wrist_0_rgb` is
currently an all-zero placeholder with `image_mask=False`, because this robot has two cameras. Fill
it with `wrist_image_left` at t-k. The slot is a wrist slot in pretraining, so a wrist view is much
closer to the pretrained distribution than an exterior view would be.

Implementation, roughly:

* `data_loader.create_torch_dataset`: extend `delta_timestamps` to image keys, e.g.
  `{"wrist_image_left": [-k/fps, 0.0]}`. LeRobot handles negative offsets natively and clamps
  out-of-range indices to the episode boundary (`_get_query_indices`), so the first k frames repeat
  frame 0 and are flagged via `wrist_image_left_is_pad`.
* a custom `DroidInputs` variant: index 1 -> `left_wrist_0_rgb`, index 0 -> `right_wrist_0_rgb`,
  both masks True. Keep the mask True even for padded frames and have the client duplicate the
  current frame when t < k, so training and inference behave identically.
* deployment contract changes: the client must send a new key (e.g.
  `observation/wrist_image_left_prev`) and maintain its own one-frame buffer at exactly the same lag
  k. Keeping the buffer client-side leaves the server stateless.

Decided: **k = 15 frames**, which is 1.0 s at the current 15 fps. Defined in frames rather than
seconds so that it cannot drift silently if the capture rate changes; if a future dataset uses a
different fps, re-derive k to keep the lag at roughly one second. Note that 15 frames is also
about one action chunk (16 frames, 1.07 s), so the model sees where the wrist was at the start of
the previous chunk.

Remaining caveat: a one-second-old wrist view encodes motion *direction*, which resolves the
towards/away ambiguity while the arm is moving. It does **not** resolve "has the pour already
happened" while the arm is stationary -- that cue lives in the exterior view (liquid in the cup),
not in the wrist close-up. If that failure shows up on the robot, the fix is phase conditioning in
the prompt, not a longer lag.

**Cost.** Enabling the third slot adds 256 image tokens to the prefix. Inference was 98 ms with two
slots; this needs re-measuring.

**Conflict to resolve before the next data collection.** The free slot exists only because
`exterior_image_2_left` was a dead, all-black camera in this dataset. If the new capture has a
working second exterior camera, all three slots are taken by live views and there is no room for
the history frame. That is a choice that has to be made at collection time:

* three live views (base, second exterior, wrist), no history, or
* two live views plus a one-second-old wrist frame.

A second exterior view showing the cup would attack the phase ambiguity directly, which the history
frame does not. Worth deciding deliberately rather than inheriting it from whatever the rig does.

#### Revision: history goes into the state, not into a camera slot

Superseding the plan above. The history frame does not need a camera slot at all -- lagged
**proprioception** in the state vector carries the same disambiguating signal, leaves all three
camera slots for live views, and adds no image tokens.

One measurement decides the form. pi0.5 digitises the state into **256 bins**, and joint positions
span 1.2-4.2 rad while one frame of motion is 0.0006-0.003 rad, so at a one-frame lag the median
displacement is **0.20 bins** and **78% of components fall below a single bin** -- `q_t` and
`q_{t-1}` tokenise to identical digits and the signal is destroyed. A one-frame lag is useless
here; the lag has to be long enough to clear quantisation.

Sweep over lag (median episode is 696 frames = 46 s):

| lag | median displacement | signal vanishes | frames clamped at episode start |
| --- | --- | --- | --- |
| 1 s (k=15) | 3.8 bins | 9.7% | 2.0% |
| 2 s (k=30) | 6.9 bins | 4.7% | 3.9% |
| 3 s (k=45) | 9.8 bins | 2.2% | 5.9% |
| 5 s (k=75) | 15.0 bins | 0.9% | 9.8% |
| 10 s (k=150) | 24.6 bins | 0.1% | 19.5% |

A long lag was expected to alias -- an out-and-back motion returning `q_{t-k}` to `q_t` -- but that
does not happen within 10 s on this task, so signal strength increases monotonically and the only
cost is episode-start clamping.

**Decided form**, two lags so short-range direction and longer-range phase are both covered:

```
state = [q_t(7), grip_t(1), q_{t-15}(7), grip_{t-15}(1), q_{t-45}(7), grip_{t-45}(1)]   # 24 dims
```

Measured token cost with the real PaliGemma tokenizer (prompt `"pour the teapot into the cup"`,
`max_token_len=200`):

| state dims | contents | tokens | headroom |
| --- | --- | --- | --- |
| 8 | current proprio (today) | 45 | 155 |
| 16 | current + 1 lag | 72 | 128 |
| 24 | current + 2 lags (chosen) | 100 | 100 |
| 32 | current + 3 lags | 130 | 70 |

Each extra snapshot costs about 28 tokens, so 24 dims uses exactly half the budget. Norm stats must
be recomputed for the widened state.

Note that 32 is *not* a hard ceiling for pi0.5: `pad_to_dim` only ever pads upwards and returns the
array unchanged when it is already at or above the target, and pi0.5 never reads the state array at
all (it is consumed by the tokenizer before padding). The real ceiling is the token budget, roughly
six snapshots. Staying at or below 32 keeps compatibility with the pi0 code path and with
convention. The flip side of that permissiveness: exceeding the intended width fails silently
rather than raising, so the width check has to be ours.

Prefer lagged proprioception over the previous *action*: proprio is a measurement rather than a
command (commands can be clipped or overridden by the controller), and it is the same modality the
pretrained state tokens already carry. It does not eliminate the copycat risk -- `(q_t - q_{t-k})`
still correlates with the action being predicted -- so consider dropping the lagged entries for a
fraction of training samples, and after training verify by zeroing them and measuring how much the
output moves.

#### Two implementation traps for the widened state

**Do not let the lagged copies displace the current one.** LeRobot returns a *stacked* array when a
key appears in `delta_timestamps`, so `{"joint_position": [-45/fps, -15/fps, 0.0]}` yields shape
`(3, 7)` and `joint_position` stops being the plain current reading everywhere downstream. The
transform has to flatten it deliberately, current first:

```
state = [q_t(7), grip_t(1), q_{t-15}(7), grip_{t-15}(1), q_{t-45}(7), grip_{t-45}(1)]
```

Current in the leading 8 dims keeps the same positional semantics the pretrained state tokens have.
An easy way to get this wrong is to index `[0]` out of the stack, which is the *oldest* frame, not
the current one.

**Share one set of norm stats across the three blocks.** If `compute_norm_stats.py` estimates 24
independent quantiles, the three snapshots of the same physical variable get three slightly
different `q01`/`q99` (sampling noise), so identical joint angles at t and t-45 can land in
different bins and manufacture apparent motion. Compute the stats on the 8 base dimensions and tile
them three times instead. The whole point of the design is that the *difference* between blocks is
readable in bin space, and independent normalisation quietly corrupts exactly that.

## Operational notes from the B1 launch (2026-08-22)

**Probe throughput does not predict sustained throughput.** The 20-step probe reported 1.657 s/step,
but the real run settled at 2.464 s/step with the same config: the probe's handful of steps were
served entirely from the prefetch queue built during startup and never reached steady state. Run a
probe long enough to drain the queue -- a couple of hundred steps -- before trusting its rate for a
walltime estimate.

**GPU utilisation at 5-second granularity hides data starvation.** `nvidia-smi -l 5` reported 95%
average utilisation while the loader was in fact the bottleneck. The reliable signal was the
variance of `step_time`: 2.20-2.66 s while starved, and 1.6306-1.6312 s once fed. Raising
`num_workers` from the default 2 to 8 (each batch decodes 128 PNGs; the node has 32 CPUs) took the
run from 20.5 h to 13.6 h for 30k steps. Worker count does not affect the sample order, which comes
from the seeded sampler, so this is free.

## B1: how long to actually train (2026-08-22)

The 30k run (job 53441114) was stopped at 19k steps once its validation curve made the answer
obvious. Every recorded point after the first was worse than the first:

| step | val_loss | epochs |
| --- | --- | --- |
| 2500 | 0.04248 | 3.0 |
| 5000 | 0.05003 | 6.1 |
| 10000 | 0.05870 | 12.2 |
| 17500 | 0.06694 | 21.3 |

Since that run only recorded at its 2,500-step save interval, its earliest point was already past
the minimum. A short scan (job 53575246, 4k steps, eval and save every 250, 16 eval batches) located
it:

| step | val_loss | vs best |
| --- | --- | --- |
| 250 | 0.05022 | +30.9% |
| 500 | 0.04411 | +15.0% |
| 750 | 0.04237 | +10.5% |
| **1000** | **0.03836** | best |
| 1250 | 0.03915 | +2.1% |
| 1500 | 0.03929 | +2.4% |

**The optimum is around 1,000 steps, which is 1.22 epochs** (1000 x 64 / 52,555 train frames). The
30k run was 30x past it; B0's 15k steps on the smaller dataset were 62.5 epochs, roughly 50x past,
which nobody could see at the time because that run had no held-out split.

Read the minimum as a plateau, not a point: 1000/1250/1500 differ by 0.0008-0.0009 while the
standard error of the mean is std/sqrt(16) ~ 0.0011, so those three are statistically
indistinguishable. What is unambiguous is the 24% drop from 250 to 1000 and the steady climb after.

Two caveats before treating 1,000 steps as the recipe:

* `val_loss` is the flow-matching denoising loss, not task performance. Whether action-space error
  (MAE, creep) bottoms out at the same place still has to be checked with `eval_offline.py`.
* `keep_best_checkpoint` retains only the best and the latest, so the intermediate weights are
  already deleted and an action-metric curve across steps cannot be reconstructed from this run.
  Disable it for a run whose purpose is to compare metrics across steps.

## B1 A/B: LoRA vs full action expert (2026-08-22)

Two runs on franka_pour_wine, identical except for the action expert. Vision stays fully trainable
in both -- cocktail making is not a common manipulation scenario, so the visual domain is unlikely
to be well covered by pretraining.

| | A: LoRA expert | B: full expert |
| --- | --- | --- |
| trainable | 494.8M | 900.6M |
| best action MAE | **0.05773** @ 23k | 0.05886 @ 9k (still running) |
| vs zero baseline | 19.4% | 17.8% |
| steps to reach ~0.059 | ~17k | ~6.5k |

Doubling the trainable parameters bought **no accuracy**, only faster convergence: B reaches in
6,500 steps what A needs 17,000 for, then hits the same wall. If that holds to 30k, the bottleneck
is data (36 training episodes), not capacity, and the next move is more demonstrations rather than
more architecture changes.

Both plateau at roughly 19% better than a constant-zero predictor, which is weak in absolute terms.
Offline MAE is however a pessimistic measure for multimodal demonstrations -- several trajectories
can be equally valid and MAE punishes picking a different one -- and B0 looked similarly unimpressive
offline yet behaved correctly on the robot. Real-robot success rate remains the deciding measure.

### Two mechanism bugs this run exposed

* **Eval points and save points are different things.** With `eval_interval=500` and
  `save_interval=2500`, four of every five minima have no weights behind them: A's best evaluation
  was step 23,000 (0.05773) but the best *available* checkpoint is 17,500 (0.05878). Because
  `keep_best_checkpoint` prunes to two, `save_interval` can simply be set equal to `eval_interval`
  -- it costs write I/O, not disk.
* The end-of-run log reported "best at step 23000" while retaining {17500, 29999}, i.e. it named a
  checkpoint that was never written. Fixed to report the best *retained* step, and to say so
  explicitly when the best evaluated point was not saved.
