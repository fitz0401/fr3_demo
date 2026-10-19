# Report: raw 1 kHz force and scale readings in the Hub dataset

Answers `docs/RAW_FORCE_HF.md`. Collection-side work only; `fr3_pi05/runner.py` was not touched.

## Status

The dataset is **converted and verified locally but not yet uploaded**. The operator is reviewing two
episodes with large unaccounted liquid loss before it is published, so the Hub repo id below is reserved,
not live. Everything else in this report is measured from the finished local dataset.

- Local dataset: `/home/data/u0161364/lerobot/fitz0401/franka_pour_lillet_blue_cup_2_force` (7.3 GB)
- Intended Hub repo id: `fitz0401/franka_pour_lillet_blue_cup_2_force`
- `fitz0401/franka_pour_lillet_blue_cup_2` was **not** overwritten.

## Episode counts

- episodes converted: **22**, total frames **29,033**
- with raw force: **22/22**; without: none
- with weights: **22/22** (all four readings present on every episode)
- raw samples total: **1,669,604** = 57.5x the 15 Hz frame count

All 22 raw episodes behind `franka_pour_lillet_blue_cup_2` carry both `raw_force_*` and `weights`.

The task document expected they might not, reasoning that the 16:39 Hub upload predates commit 8358454
(16:45). That inference does not hold: the recorder ran from an uncommitted working tree that day, so the
feature was live from 12:46 even though it was committed at 16:45. Commit time is not when code went live.
Verified per episode rather than by timestamp.

## Per episode

| ep | source | raw samples | rate (Hz) | largest gap (ms) | frames | skipped | alignment (Nm) | poured (g) | spilled (g) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | episode_000000 | 90,368 | 860.9 | 18 | 1574 | 0 | 1.91e-06 | 31.0 | -1.0 |
| 1 | episode_000001 | 81,144 | 862.2 | 22 | 1411 | 0 | 1.91e-06 | 68.0 | 0.0 |
| 2 | episode_000002 | 94,932 | 860.9 | 26 | 1653 | 1 | 1.90e-06 | 69.0 | 5.0 |
| 3 | episode_000003 | 87,293 | 864.2 | 19 | 1514 | 0 | 1.91e-06 | 22.0 | 7.0 |
| 4 | episode_000004 | 79,077 | 858.1 | 27 | 1381 | 1 | 1.91e-06 | 43.0 | -1.0 |
| 5 | episode_000005 | 83,280 | 864.5 | 31 | 1444 | 0 | 1.92e-06 | 57.0 | 1.0 |
| 6 | episode_000006 | 98,047 | 862.6 | 24 | 1704 | 1 | 1.91e-06 | 128.0 | 2.0 |
| 7 | episode_000007 | 56,014 | 864.2 | 20 | 971 | 0 | 1.91e-06 | 44.0 | 6.0 |
| 8 | episode_000008 | 72,060 | 864.3 | 22 | 1250 | 1 | 1.89e-06 | 56.0 | 1.0 |
| 9 | episode_000009 | 71,375 | 857.9 | 26 | 1247 | 1 | 1.91e-06 | 83.0 | 0.0 |
| 10 | episode_000010 | 64,367 | 856.2 | 24 | 1127 | 0 | 1.91e-06 | 65.0 | 2.0 |
| 11 | episode_000011 | 68,706 | 860.2 | 25 | 1197 | 0 | 1.91e-06 | 0.0 | 0.0 |
| 12 | episode_000012 | 85,093 | 859.4 | 24 | 1484 | 0 | 1.91e-06 | 52.0 | 2.0 |
| 13 | episode_000013 | 61,876 | 861.1 | 27 | 1077 | 0 | 1.92e-06 | 88.0 | -1.0 |
| 14 | episode_000014 | 73,672 | 865.5 | 22 | 1276 | 1 | 1.90e-06 | 66.0 | -1.0 |
| 15 | episode_000015 | 62,177 | 852.2 | 29 | 1094 | 1 | 1.91e-06 | 26.0 | 61.0 |
| 16 | episode_000016 | 96,358 | 866.0 | 34 | 1668 | 0 | 1.91e-06 | 6.0 | 0.0 |
| 17 | episode_000017 | 86,067 | 871.8 | 21 | 1480 | 1 | 1.91e-06 | 48.0 | 0.0 |
| 18 | episode_000018 | 69,463 | 866.7 | 26 | 1201 | 0 | 1.91e-06 | 75.0 | 17.0 |
| 19 | episode_000019 | 70,847 | 862.3 | 22 | 1231 | 1 | 1.91e-06 | 42.0 | 2.0 |
| 20 | episode_000020 | 56,443 | 859.4 | 20 | 984 | 0 | 1.90e-06 | 54.0 | 3.0 |
| 21 | episode_000021 | 60,945 | 858.0 | 25 | 1065 | 1 | 1.91e-06 | 63.0 | 63.0 |

Rate **852.2-871.8 Hz** (mean 861.7), largest single gap
**34 ms**, zero duplicate controller timestamps across all
1,669,604 samples. Alignment error **1.89e-06 to 1.92e-06 Nm**.

## Deviations from the task document

**1. `datasets` is not pinned to 3.6.0 (section 4). The parquet footer is made portable instead.**

The incompatibility was reproduced first: `datasets` 3.6.0 rejects our parquet with
`ValueError: Feature type 'List' not found`. It is purely the `_type` string in the footer -- 4.x renamed
the fixed-length list feature from `Sequence` to `List`. Both decode to the identical
`fixed_size_list<float>[7]`, and 5.0.1 still *reads* `Sequence`, so conversion now rewrites the footer tag
to `Sequence`. Measured: values and encoded image bytes bit-identical across the rewrite, and the result
loads on **both** 3.6.0 and 5.0.1 with images decoding to (320, 180).

This is strictly better than the pin: no environment is held at a particular `datasets` release, and the
training side needs no repair step. Pinning 3.6.0 here would also have been a two-major-version downgrade
of a working environment.

**2. The alignment check skips frames that fall outside the raw stream.**

As specified, the check failed on episode 2 at 0.1119 Nm. It was one frame of 1653 -- frame 0. In
**10 of 22 episodes** the first frame's `robot_timestamp` lands 0.1-1.0 ms *before* the first retained raw
sample, so `np.interp` clamps rather than interpolates and the check was measuring the clamp. Excluding
frames outside the raw span gives 1.9e-06 Nm uniformly on all 22. Those frames are counted per episode as
`frames_outside_raw_span` rather than silently dropped.

**3. The footer rewrite is atomic.**

An in-place `pq.write_table` destroyed `episode_000000.parquet` on the first attempt: it truncates the
destination before validating its arguments, so being rejected over a compression name
(`UNCOMPRESSED` must be written as `none`) left no file at all, while `info.json` still claimed 22
episodes. The rewrite now stages to a `.tmp` sibling and replaces only on success, and that incomplete
dataset was discarded and rebuilt from scratch. Regression-tested.

## Data quality for review

These are faithful recordings, not pipeline faults. Derived values are never clamped, so a residual stays
visible.

- **Episode 11 -- no pour, no spill (reviewed and corrected).** The arm grasped the bottle and executed a
  pour, but only 1 g left it. Peak tilt while holding the bottle was 68.7 deg against 81.4 deg on a
  successful pour, with the bottle already worked down to 739 g, so the liquid never reached the spout;
  the operator confirmed this from video. The -4 g cup reading is scale error, which the arithmetic had
  turned into `poured_g` -4.0 and a phantom `spilled_g` 5.0.

  `poured_g` and `spilled_g` are now both 0.0, carrying `outcome_source: manual_review`, an
  `outcome_reason`, and the superseded `measured_poured_g` / `measured_spilled_g`. **The four scale
  readings are unchanged.** A corrected episode is never presented as a clean measurement, and
  `fr3_demo.recording.override_episode_outcome` is the only path that writes these fields.

- **Episodes 15 and 21 -- large unaccounted loss, accepted as real spills.** 87 g left the bottle and 26 g
  reached the cup (ep 15); 126 g left and 63 g arrived (ep 21). Episode 15 is corroborated by the weight
  chain: episode 16 inherits `bottle_before = 681` and balances exactly (-6 bottle, +6 cup), so
  `bottle_after` was not a misreading. Episode 21 is the session's last, so it has no following row to
  cross-check. Reviewed and kept as recorded.

- **Four `spilled_g = -1.0`** values (ep 0, 4, 13, 14) are scale resolution: the two independent
  measurements agreeing to within 1 g.

- Expected discontinuities: fresh bottles at episodes 4, 8, 12, 18; the cup was emptied between
  episodes 13 and 14.

### The force stream cannot be used to weigh the liquid

Worth stating because it is tempting to try. Calibrating joint torque against bottle mass over
pose-matched frames pooled across all 22 episodes (183k matched pairs, masses spanning 299 g) gives a
sensitivity of about 0.02 Nm per 100 g against 0.09-0.19 Nm of scatter at matched pose -- a mass
resolution of several hundred grams. Teleoperation never revisits a pose (the closest an episode comes to
repeating its own configuration is 0.145 rad), so the payload term cannot be separated from the arm's
configuration-dependent gravity and friction load without a dynamics model and payload identification,
neither of which exists here. The scale readings are the ground truth for grams; the 860 Hz stream is
contact/force data for learning the pour, and corroborates nothing about mass in either direction.

## Reproduce

```bash
fr3-convert \
  --data-dir data/raw/session_20261016_124600_lillet_blue_cup_2 \
  --repo-id fitz0401/franka_pour_lillet_blue_cup_2_force \
  --output-root /home/data/u0161364/lerobot
```

Add `--push-to-hub` to upload. Output goes to `/home/data` because `/home` had 8.7 GB free against a
7.3 GB dataset; no caches had to be cleared.

