# APO vs SkillOpt Face-off: Method and Results

**English** | [中文](apo-faceoff.zh.md)

This document records the head-to-head comparison between the prompt tuned by
the old agent-lightning/APO project and the skill tuned by this SkillOpt port,
including how the shared held-out test split was built and enlarged, the exact
numbers, and the conclusions. Date of the runs: 2026-07-15.

## 1. What is compared

| Contestant | File | Origin |
| --- | --- | --- |
| baseline | `video2frames_env/skills/initial.md` | the untuned instruction prompt (byte-identical to the old project's `baseline_prompt.txt` — verified with `diff`) |
| APO best | `data/best_prompt.txt` | best prompt from the agent-lightning/APO run |
| SkillOpt best | `outputs/skillopt_video2frames_gpt-5.4_20260715_141809/best_skill.md` | best skill from this project's training run (20 steps, 80 train / 100 val) |

Comparability guarantees:

- **Same scoring**: the soft/hard formulas are the lossless port of the APO
  reward ([reward-design.md](reward-design.md)).
- **Same target model** (`gpt-4.1-mini`) and **same judge**
  (`env.judge_model`), fixed across all runs.
- **Uncontaminated test split**: every test task is unseen by *both*
  optimizers — excluded from this project's train/val *and* from the old APO
  project's train/val.

## 2. Growing the test split (30 → 100)

Round 1 ran on the original 30-task test split; the observed gap (APO soft
+0.027 ± 0.018 SE) was within noise, so the split was grown to 100 tasks with:

```bash
python prepare_data.py --grow-test 100 --probe-content-filter --probe-workers 8 \
    --exclude <old-apo-project>/data/train.jsonl <old-apo-project>/data/val.jsonl
```

`--grow-test` appends newly sampled tasks to `data/test.jsonl` until it holds
the target count. Train/val and the existing test rows are untouched
(byte-identical prefix), new candidates are stratified-sampled from the source
pool minus every video used by any split or `--exclude` file, and each one is
probed against the content filter. The candidate pool after all exclusions was
5,582 of 5,847 source videos.

Note: the whole dataset is extremely courier-sparse (13 positives in 5,847
source records, 0.2%), so all splits carry 0 courier positives; the
`courier_match` reward component effectively measures "does the model refrain
from false positives".

## 3. Results on the full 100-task test split

`evaluation.test_env_num` caps eval size (it was 30 in the config); the full
runs used `--cfg-options evaluation.test_env_num=100`.

| Skill | hard | soft | scene_match | courier_match | judge_score |
| --- | --- | --- | --- | --- | --- |
| baseline | **0.59** | 0.7879 | 0.94 | 0.98 | 0.6732 |
| SkillOpt best | 0.55 | 0.7846 | **0.95** | 0.98 | 0.6643 |
| APO best | 0.57 | **0.8056** | 0.94 | **1.00** | **0.6960** |

Paired per-task analysis (same tasks, differences per task, n=100):

| Comparison | soft diff ± SE | t | wins/losses/ties |
| --- | --- | --- | --- |
| APO − baseline | +0.0177 ± 0.0129 | 1.37 (n.s.) | 43/36/21 |
| SkillOpt − baseline | −0.0033 ± 0.0124 | −0.27 (flat) | 37/42/21 |
| APO − SkillOpt | +0.0210 ± 0.0114 | 1.84 (p ≈ 0.07) | 44/36/20 |

For reference, round 1 on the 30-task split: baseline 0.7749, SkillOpt 0.7830,
APO 0.8104 (soft). Re-running the same 30 tasks during round 2 moved each
skill by ±0.01–0.02 — single-eval variance alone is that large, which is why
the 30-task round could not separate the contestants.

## 4. Conclusions

1. **SkillOpt's gain did not generalize.** On the 70 fresh tasks its best
   skill scores level with the baseline (0.7957 vs 0.7921 soft); the +0.008
   seen on the old 30-task split was gate overfitting to the val split plus
   noise.
2. **APO leads SkillOpt** by +0.021 soft (t = 1.84, marginal), and its edge
   over the baseline shrank to a non-significant +0.018 on the larger split.
3. **The whole gap lives in `judge_score`** (text quality against ground
   truth): scene/courier exact-match components are near-saturated and almost
   identical across contestants.
4. **`hard` is highest for the baseline** (0.59): both tuned prompts slightly
   raise or hold average quality while pushing a few borderline tasks below
   the 0.8 bar — another reason `hard` is not used for gating.
5. Practical upshot: with `gpt-4.1-mini` as the target, this task's headroom
   over the baseline prompt is small (≤ +0.02 soft); measuring it reliably
   needs ≥ 100 paired test tasks, and claiming it needs a better optimization
   run, not a better measurement.

## 5. Reproducing

```bash
# grow the test split (idempotent once at 100)
python prepare_data.py --grow-test 100 --probe-content-filter \
    --exclude <old-apo>/data/train.jsonl <old-apo>/data/val.jsonl

# evaluate the three contestants
python eval.py --config configs/video2frames/default.yaml \
    --skill video2frames_env/skills/initial.md --split valid_unseen \
    --cfg-options evaluation.test_env_num=100 env.out_root=outputs/faceoff100_baseline_n100
python eval.py --config configs/video2frames/default.yaml \
    --skill outputs/<run>/best_skill.md --split valid_unseen \
    --cfg-options evaluation.test_env_num=100 env.out_root=outputs/faceoff100_skillopt_n100
python eval.py --config configs/video2frames/default.yaml \
    --skill data/best_prompt.txt --split valid_unseen \
    --cfg-options evaluation.test_env_num=100 env.out_root=outputs/faceoff100_apo_n100
```

Each output directory holds `eval_summary.json`, per-task `results.jsonl`, and
raw predictions; the paired analysis is a straight per-task join of the three
`results.jsonl` files on task id.
