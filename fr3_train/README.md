# fr3_train

Training-side code and contract for the FR3 pouring policies, so that the client side can check
what the model actually expects instead of inferring it.

Everything here is a delta on top of [openpi](https://github.com/Physical-Intelligence/openpi) at
commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`; there is no standalone training library.

## If you are writing the client, read these three

| file | what it answers |
| --- | --- |
| `DEPLOY.md` | the request/response contract, in full |
| `reference_wrist.png`, `reference_exterior.png` | camera orientation, taken straight from the training set -- compare your live feed against them |
| `serve_wine.py` | the server that enforces the contract; the transform in it is the definition |

The three things most likely to be got wrong:

1. **State is three stacked frames, not one.** `observation/joint_position` is `(3, 7)` and
   `observation/gripper_position` is `(3,)`, ordered **current first**, then t-45 and t-75 frames
   (3 s and 5 s back at 15 fps). Index 0 is the present; reading it as a lag trains and runs fine
   while feeding the model a stale state. For unavailable negative history, repeat the episode's
   startup frame, matching LeRobot's first-frame clamping.
2. **The prompt is mandatory.** The two tasks share a scene and differ only in the instruction, so a
   default would quietly pour the wrong bottle. Requests without one are rejected.
3. **Actions are joint velocities**, `(16, 8)` float64: seven joint rates in rad/s plus a gripper
   position in [0, 1]. Not positions, not deltas.

## If you are reproducing the training

```bash
git clone https://github.com/Physical-Intelligence/openpi.git && cd openpi
git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
git apply /path/to/fr3_train/training/openpi_fr3.patch
uv sync
```

The patch adds, on top of upstream:

* `pi05_wine_hybrid` and `pi05_wine_ae_full` train configs, plus the `custom_droid` ones that came
  before them
* lagged proprioception in the state (`state_history_lags`), threaded through the data loader's
  `delta_timestamps` and a `DroidInputsWithStateHistory` transform
* in-loop validation on a held-out episode split, with action-space metrics and best-checkpoint
  retention (`openpi/training/evaluation.py`, changes to `scripts/train.py`)
* `scripts/eval_offline.py` for scoring saved checkpoints and `scripts/verify_export.py` for
  checking a deployment package before it ships
* the `gemma_2b_lora_r32` variant and an FFN LoRA scaling fix

`deploy.patch` is the small subset of that -- 42 lines across `gemma.py` and `lora.py` -- that a
machine only running inference needs.

## Results and dead ends

`FINDINGS.md` is the running record: what was measured, what it meant, and which of the plausible
ideas turned out not to work. Worth reading before repeating an experiment. The short version for
this dataset (40 episodes, 57,564 frames, two tasks):

* Validation loss and action-space error **move in opposite directions** over training. Ranking
  checkpoints by the flow-matching denoising loss picks the worst one. Rank by action MAE.
* Held-out action MAE bottoms out around 0.0588 rad/s, about 18% better than a constant-zero
  predictor. Doubling the trainable parameters (full action expert instead of LoRA) does not move
  that number at all, so the limit is data rather than capacity.
* pi0.5 has no memory of its own: one frame per camera, no observation history, nothing carried
  between calls. The lagged state exists to supply it.
