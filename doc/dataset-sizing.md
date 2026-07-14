# Choosing Dataset Sizes for SkillOpt

**English** | [中文](dataset-sizing.zh.md)

How large do the `train` / `val` / `test` splits need to be? Large enough that
the score differences you care about are visible above the evaluation noise —
and no larger. This document gives a concrete procedure for this project:
estimate the noise from runs you already have, derive the required sample
sizes, and scale up in stages instead of guessing.

## 1. Why size matters: gate noise

SkillOpt accepts or rejects each skill edit by its **average soft score on the
val split** (the selection gate, `evaluation.gate_metric: soft` over
`evaluation.sel_env_num` tasks). Every average over `n` tasks carries a
standard error

```
SE = σ / √n
```

where `σ` is the per-task soft-score standard deviation. When the edited skill
is compared against the current best, the difference of two such averages is
only trustworthy when it exceeds roughly `2.8 × SE` (a two-sample z-test at
95% confidence, `√2 × 1.96 ≈ 2.8`). Anything smaller is coin-flipping — the
gate will accept or reject edits based on noise, and the reported best score
will not reproduce.

To detect a true difference of size `δ` between two skills, you need

```
n ≈ 2 × (1.96 × σ / δ)²    per skill, on the val split
```

With a typical `σ ≈ 0.20` for this project's hybrid score:

| val size `n` | SE | smallest reliably detectable gap (≈2.8×SE) |
| --- | --- | --- |
| 24 (default) | ~0.041 | > 0.11 |
| 50 | ~0.028 | > 0.08 |
| 100 | ~0.020 | > 0.055 |
| 200 | ~0.014 | > 0.04 |

Interpretation for this project: with the default `val` of 24, a skill edit
that improves the true soft score by 0.05 is **invisible** to the gate. Expect
useful prompt-tuning gains in the 0.03–0.10 range, so a val split of
**64–100** is a realistic working size.

## 2. Estimate σ from runs you already have (free)

Do not guess `σ` — the per-task scores are already on disk:

- `outputs/eval_.../` (written by `eval.py`) contains one `soft` score per
  task in the run's `results.jsonl`, plus the aggregate in
  `eval_summary.json`.
- `outputs/<train run>/history.json` records the gate score per step;
  the per-task results of each step's selection eval live under
  `outputs/<train run>/steps/`.

Compute the standard deviation of those per-task soft scores; that is your
`σ`. Plug it into the formula above with the `δ` you want to detect, and you
have your val size. Re-estimate after any change to the scoring function or
the target model — `σ` is a property of the (model, score, data) combination,
not a constant.

## 3. Different splits need different sizes

The three splits play different roles, so they scale differently:

- **`val` — scale this first.** It drives the accept/reject gate inside
  SkillOpt; its noise directly causes wrong gate decisions. Target the `n`
  from the formula (typically 64–100 here). Keep `evaluation.sel_env_num`
  equal to the full val size so every candidate skill is scored on the same
  tasks.
- **`test` — second priority.** The final baseline-vs-tuned comparison is
  **paired**: run `eval.py` with both skills on the same split, so use the
  standard deviation of the per-task score *differences* (`σ_d`, usually much
  smaller than `σ`) in a one-sample version of the formula
  (`n ≈ (1.96 × σ_d / δ)²`). Around 100 tasks is usually enough for a
  credible final claim. Match `evaluation.test_env_num` to the split size.
- **`train` — usually fine as is.** Each step only samples
  `train.batch_size` (default 8) tasks, and each reflection analyst sees
  `gradient.minibatch_size` (default 4) of them; a pool of 40 already
  provides variety. To improve the reflection signal, increase
  `gradient.minibatch_size` or `train.num_epochs` before adding train data.

## 4. Staged scaling: coarse screening + large-set re-scoring

Gating every step on a large val split is the expensive part. Rollout cost
per run is roughly:

```
rollouts ≈ sel_env_num                          (baseline gate eval)
         + steps × (batch_size + sel_env_num)   (train + gate per step)
         + 2 × test_env_num                     (baseline + final test eval)

steps = num_epochs × ceil(train_size / (batch_size × accumulation))
```

each rollout being one multimodal call + one judge call. The standard remedy
is a racing ladder — cheap screening for the many, expensive scoring for the
few:

1. **Screen** — train with a small-to-medium val (e.g. 24–64). Small samples
   are enough to reject clearly bad edits; only near-ties are decided by
   noise.
2. **Re-score** — after training, take `outputs/<run>/best_skill.md` (plus
   any promising intermediate skill from `outputs/<run>/skills/`) and the
   baseline, and re-evaluate each on a larger held-out re-scoring split
   (100–200 tasks, sampled from the full pool, disjoint from
   train/val/test):

   ```bash
   .venv/bin/python eval.py --config configs/video2frames/default.yaml \
       --skill video2frames_env/skills/initial.md --split valid_unseen
   .venv/bin/python eval.py --config configs/video2frames/default.yaml \
       --skill outputs/<run>/best_skill.md --split valid_unseen
   ```

   Pick the final winner by these scores, not the small-val gate scores.
3. **Escalate only on evidence** — if the re-scored best-vs-baseline gap is
   smaller than `2 × SE` of the re-scoring set, the effect is not
   established. First search harder (more epochs, larger
   `optimizer.learning_rate` edit budget, better reflection minibatches);
   only grow the datasets when a promising-but-unconfirmed gap needs a
   tighter confidence interval.

This ladder needs no changes to SkillOpt itself — `prepare_data.py` sizes are
CLI flags and `eval.py` accepts any skill file and split.

## 5. Sampling techniques

- **Keep `val`/`test` stratified-random** (already the default:
  `prepare_data.py` stratifies by dataset family × courier label with a fixed
  seed). These splits must represent the deployment distribution; never bias
  them.
- **Bias `train` toward hard examples if anything.** The reflection step
  learns from failures (`gradient.failure_only` even restricts it to them),
  so low-score tasks carry the most information. Score a candidate pool with
  the initial skill first, then over-sample the low-score tasks into `train`.
  This usually beats simply adding more (mostly easy) train data.
- **Always sample with `--probe-content-filter`** so content-filter-blocked
  videos (uniform score 0, pure noise) never enter any split; probe results
  are cached per video in `data/content_filter_cache.json`, so growing the
  splits later re-probes only the new videos.

## 6. Step-by-step playbook: growing data and search budget together

The dataset sizes and the training hyperparameters are one budget — grow them
in lockstep, one stage at a time, and let each stage's numbers decide the
next move.

**Stage 0 — smoke (once per environment).** Verify the loop, not the science.

```bash
.venv/bin/python prepare_data.py --train-size 2 --val-size 2 --test-size 2 --probe-content-filter
.venv/bin/python train.py --config configs/video2frames/default.yaml \
    --cfg-options train.num_epochs=1 train.batch_size=2 env.limit=4 env.workers=1 \
    evaluation.sel_env_num=4 evaluation.test_env_num=4 env.out_root=outputs/smoke_epoch
```

Move on when: run completes, `outputs/smoke_epoch/history.json` has one
record per step.

**Stage 1 — pilot: measure the noise.** Default sizes, default config.

```bash
.venv/bin/python prepare_data.py --train-size 40 --val-size 24 --test-size 30 --seed 42 --probe-content-filter
.venv/bin/python train.py --config configs/video2frames/default.yaml
.venv/bin/python eval.py --config configs/video2frames/default.yaml \
    --skill video2frames_env/skills/initial.md --split valid_unseen   # per-task scores -> sigma
```

Read out of this stage: `σ` from the baseline eval's per-task results, and
the step-by-step gate scores in `outputs/<run>/history.json`. Decision rule:

- Accepted and rejected edits all land within `2.8 × σ/√24` of the current
  best → val is too small to gate; go to Stage 2.
- The gate cleanly separates good edits → you may already confirm the best
  skill on test and stop.

**Stage 2 — widen the search, then sharpen the ruler.** Change one axis at a
time so you can attribute the improvement:

1. *More exploration, same data* — raise `train.num_epochs` (more passes of
   reflect-and-edit; edits compound because each step starts from the
   current best) or `optimizer.learning_rate` (a larger per-step edit
   budget; `optimizer.lr_scheduler: cosine` decays it over the run).
   Prefer more epochs first: gated sequential edits compound, a bigger edit
   budget only makes each single step bolder.
2. *Better critiques* — raise `gradient.minibatch_size` to 8 so each
   reflection analyst sees more failure examples (cheap: train rollouts
   only).
3. *Sharper selection* — regrow val to the size Section 1 prescribes and
   match the config:

   ```bash
   .venv/bin/python prepare_data.py --train-size 40 --val-size 64 --test-size 100 --seed 42 --probe-content-filter
   .venv/bin/python train.py --config configs/video2frames/default.yaml \
       --cfg-options evaluation.sel_env_num=64 evaluation.test_env_num=100
   ```

   The probe cache makes regrowing cheap: only the newly added videos are
   probed. Note that regrowing with a different size re-deals all splits —
   train/val/test stay disjoint, but individual tasks may move between
   splits, so re-run the baseline eval afterwards.

Decision rule: keep escalating exploration while late steps still improve the
best (`history.json` shows gate acceptances in late epochs). If the last
epoch never improves, more epochs are wasted money — stop growing the budget.

**Stage 3 — confirm on held-out data.** Re-score the finalists (Section 4,
step 2) on test / a large re-scoring split:

```bash
.venv/bin/python eval.py --config configs/video2frames/default.yaml \
    --skill video2frames_env/skills/initial.md --split valid_unseen
.venv/bin/python eval.py --config configs/video2frames/default.yaml \
    --skill outputs/<run>/best_skill.md --split valid_unseen
```

Ship the tuned skill only if the paired gap exceeds `2 × SE` of the test set.
Otherwise return to Stage 2 with the cheapest untried lever.

Which knob for which symptom:

| Symptom (from history.json / eval) | Knob | Direction |
| --- | --- | --- |
| edits accepted/rejected within noise | `evaluation.sel_env_num` + val split size | grow val |
| skill barely changes step to step | `optimizer.learning_rate` | larger edit budget |
| best improves every epoch | `train.num_epochs` | add epochs |
| late epochs never improve | `train.num_epochs` | stop growing |
| analyses repeat the same complaint | `gradient.minibatch_size`, harder train tasks | richer failures |
| final gap plausible but unconfirmed | test split size + `evaluation.test_env_num` | grow test |

## 7. Recommended defaults for this project

| Split | Pilot (smoke) | Working size | When to grow further |
| --- | --- | --- | --- |
| train | 4 | 40 (keep) | only if reflection minibatches look repetitive |
| val | 2 | 64–100 (from σ) | gate decisions unstable across runs |
| test | 2 | ~100 | final paired gap has `p ≈ 0.05`, need tighter CI |

Procedure in one line: estimate `σ` from an eval run's per-task scores, size
val with `n ≈ 2(1.96σ/δ)²` for the gap `δ` you care about, keep train small
but hard, screen with small val, confirm on a large held-out set.
