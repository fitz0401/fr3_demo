# Task: carry the 1 kHz force stream and the scale readings into the LeRobot dataset on the Hub

For the agent on the robot computer. Read the whole file before changing code. The raw episodes live only on this
machine, so the conversion, the checks and the upload all happen here.

## Why

Since 8358454 the recorder writes two things that `fr3_demo/convert_lerobot.py` never reads:

- `raw_force_*` arrays in each episode's `trajectory.npz`: every arm sample, about 860 Hz, on its own timeline.
- `weights` in each episode's `metadata.json`: four operator scale readings, plus `poured_g` and `spilled_g`.

The converter still writes only the 15 Hz columns (and `joint_torque`), so
`fitz0401/franka_pour_lillet_blue_cup_2` on the Hub has neither. Training needs both:

- **The force stream.** The training side gravity-compensates every raw sample and builds per-frame force windows
  for the policy.
- **The scale readings.** They are the ground truth for "how many grams did the arm's force estimate say left the
  bottle" and for pour accuracy.

The converter only has to copy the data out faithfully. Windowing, gravity compensation and calibration happen on
the training side; do not add them here.

## What to change in `fr3_demo/convert_lerobot.py`

Write two extra outputs into the dataset root, `Path(dataset.root)`, next to `data/` and `meta/`.

### 1. `raw_force/episode_{episode_index:06d}.npz`, one per converted episode

`episode_index` is the LeRobot index: the position of the episode in `find_episodes()` order, which is the order
`convert()` already loops in. Write the file right after `dataset.save_episode()` for that episode.

Copy these keys from `trajectory.npz` unchanged (same dtype, same values):

| key | shape | meaning |
|---|---|---|
| `raw_force_joint_torque` | [K, 7] | tau_J of every arm sample |
| `raw_force_joint_position` | [K, 7] | qpos at the same instants |
| `raw_force_robot_timestamp` | [K] f64 | controller clock |
| `raw_force_request_send_timestamp` | [K] f64 | host monotonic |
| `raw_force_response_receive_timestamp` | [K] f64 | host monotonic |
| `robot_timestamp` | [N] f64 | one per 15 Hz frame: the controller time the frame's state was interpolated at |
| `host_monotonic_timestamp` | [N] f64 | one per 15 Hz frame |

Add these keys:

| key | value |
|---|---|
| `source_episode` | the raw episode directory name, as a 0-d string array (`np.asarray(name)`) |
| `robot_clock_scale` | from the episode's `sync_report.json` → `clock_estimates.robot_controller.scale` |
| `robot_clock_offset` | from the same place, `.offset` |

Rules:

- **Frame order.** Row `i` of the per-frame arrays must be frame `i` of that episode's parquet. The converter does not
  trim or reorder frames today; keep it that way.
- **Missing raw force.** If an episode has no `raw_force_*` arrays (recorded before 8358454), write no file for it.
  Never fill with zeros. List those episodes in the summary.
- **Gating.** Do not make the parquet conversion depend on raw force. The LeRobot dataset itself stays exactly as
  it is now.

### 2. `meta/episode_weights.jsonl`

Write one line per converted episode, for every episode, in episode-index order:

```json
{"episode_index": 3, "source_episode": "<raw dir>", "bottle_before_g": 812.0, "cup_before_g": 201.5, "bottle_after_g": 690.5, "cup_after_g": 318.0, "poured_g": 116.5, "spilled_g": 5.0, "unit": "g", "source": "operator_scale"}
```

Take the fields from `metadata["weights"]`. Write missing readings, or a missing `weights` block, as `null`. Do not
invent values. If readings were noted down but never entered, back-fill them first with
`fr3-annotate --weights` (it already exists).

### 3. A per-episode check, printed and saved to `meta/raw_force_report.json`

For each episode with raw force, report:

- **Alignment.** For each joint `j`, compute
  `np.interp(robot_timestamp, raw_force_robot_timestamp, raw_force_joint_torque[:, j])` and compare it with the
  parquet `joint_torque[:, j]`. Report the max abs difference over the episode; it should be about 1e-4 Nm. The
  recorder interpolated each frame linearly between the same two samples, so anything above about 1e-2 Nm means a
  wrong episode/file pairing or a different time base. Treat that as an error and stop.
  - Duplicate controller timestamps can occur (the sampler can poll the same controller state twice). Keep the first
    occurrence when interpolating for this check, but still save the arrays unchanged.
- **Raw rate.** `(K - 1) / (t[-1] - t[0])`, plus the largest gap between consecutive samples in ms.
- **Weights.** Whether `weights` is present, and `poured_g` / `spilled_g`.

### 4. Pin `datasets`

Add `"datasets==3.6.0"` to `pyproject.toml` dependencies. Datasets 4.x writes a `List` feature type that the
training stack cannot read. Every FR3 dataset converted so far had to be repaired on the training side for this
reason.

## Tests

Add a unit test next to `tests/test_force_and_weight.py`. Mock `LeRobotDataset` the way other tests avoid the real
dependency, or keep the new logic in small helper functions and test those. The test should cover:

- **The written npz.** Build a raw episode with `RawEpisodeWriter` plus `set_raw_force(...)` and convert it. Check the
  file has the keys and shapes above and the values are unchanged.
- **The alignment check.** It passes on that episode, and it fails if the raw arrays of one episode are paired with
  another episode's parquet.
- **Missing raw force.** An episode without raw force gets no npz and is listed.
- **Missing weights.** An episode without `weights` gets a jsonl line with nulls.

## Convert and upload

1. **Check the existing recordings.** Find out which raw episodes behind `franka_pour_lillet_blue_cup_2` contain
   `raw_force_*` and `weights`. That Hub upload (2026-10-16 16:39) predates commit 8358454 (16:45), so they may not.
2. **If they do, convert to a new repo id:** `fitz0401/franka_pour_lillet_blue_cup_2_force`.
   - Do not overwrite `franka_pour_lillet_blue_cup_2`; models have already been trained on it.
   - Use `push_to_hub`. It uploads the whole root folder except images and videos, so `raw_force/` and the new
     `meta/*.json*` files should go up with it. Open the Hub file list and confirm they are there.
3. **If they don't** (or only some do), stop after the code change and report which episodes have what. The
   remaining episodes would need re-recording with the current collector.
4. **Push.** Commit the converter change on this branch (`data/raw-force-hf`) and push it.

## Report back

Report these items in your final message, or in a file `docs/RAW_FORCE_HF_REPORT.md` on this branch:

- the Hub repo id;
- the episode count with and without raw force, and with and without weights;
- per episode: raw rate, largest gap, and alignment max diff (the `meta/raw_force_report.json` content is enough);
- anything that did not match this document.
