# Reflection Trajectories: the `conversation.json` Contract

**English** | [中文](reflection-trajectories.zh.md)

SkillOpt's optimization step only happens if the reflection stage can read a
trajectory for each rollout. That requirement is an *implicit* contract — it is
not part of the `EnvAdapter` interface — and missing it does not raise any
error: training silently degrades into repeated evaluation of the initial
skill. This document records the incident where exactly that happened, the
root cause, the fix, and how to detect the failure mode.

## 1. The incident (2026-07-15)

A full training run (`outputs/skillopt_video2frames_gpt-4.1_20260715_094345`,
4 epochs × 5 steps, ~1.49M prompt tokens) produced a `best_skill.md` that was
byte-identical to the initial skill. `summary.json` told the story:

```
"total_steps": 20,
"total_accepts": 0,
"total_rejects": 0,
"total_skips": 20,          <- every step was "skip_no_patches"
"best_origin": "initial_skill",
"token_summary": { "rollout": ... }   <- no analyst/optimizer calls at all
```

Three evaluations of the same initial prompt on the 30-item test split scored
`soft` = 0.769 / 0.754 / 0.778 — the spread (~±0.015) is pure judge noise, not
optimization progress.

## 2. Root cause

The reflection pipeline (`skillopt.gradient.reflect`) builds analyst input via
`fmt_minibatch_trajectories`, which reads
`predictions/<task_id>/conversation.json` for every rollout result and
**silently skips** any task where the file is missing:

```python
conv_path = os.path.join(prediction_dir, tid, "conversation.json")
if not os.path.exists(conv_path):
    continue
```

Our rollout wrote only `predictions/<id>/rollout.json`. So:

1. every trajectory was skipped → the formatted trajectory text was empty;
2. `run_error_analyst_minibatch` returns `None` on empty input *before*
   calling the optimizer model;
3. zero patches → the trainer logs `skip_no_patches` and moves on.

Each step record showed `reflect_s: 0.0` and an empty `patches/` directory.

### Why the port missed it

- **The requirement is not in the adapter contract.** The abstract
  `EnvAdapter.rollout()` docstring only requires returning dicts with
  `id`/`hard`/`soft`. `conversation.json` is a convention implemented
  independently by every built-in env (alfworld, searchqa, docvqa, …) and
  documented only in `gradient/reflect.py` helper docstrings.
- **The failure is silent by design.** Missing files are skipped, the analyst
  returns `None`, and `skip_no_patches` is a legal step outcome — no error, no
  warning.
- **Neither the test suite nor the smoke criteria caught it.** Offline tests
  cover rollout/scoring/dataloading; reflection runs inside skillopt with real
  model calls. The smoke acceptance ("`history.json` has one record per step,
  `best_skill.md` exists") passes even when every step skips.
- Corroborating: the adapter implemented `build_reference_text` expecting
  skillopt to attach it via `attach_reference_context`, but that method has
  **zero callers** in the pinned 0.2.0 wheel — the port was written against
  the adapter API surface without tracing the reflect-stage data flow.

## 3. The fix

`video2frames_env/rollout.py` now writes, for **every** task (success, content
filter, error — failures are exactly what the error analyst needs):

- `predictions/<id>/rollout.json` — scores + solution, as before;
- `predictions/<id>/conversation.json` — the trajectory the analyst reads:

```json
[
  {"role": "user",      "content": "<task description> (frame images omitted from this trace)"},
  {"role": "assistant", "content": "<model response or '(no response)'>"},
  {"role": "system",    "content": "scene_match=.. courier_match=.. judge_score=.. soft=.. hard=..\nfail_reason: .."}
]
```

Each result dict also carries `reference_text` (the ground-truth JSON), which
`fmt_minibatch_trajectories` renders as a "Hidden Reference" header — the
analyst can compare output vs. expectation without the reference ever entering
the rollout prompt.

Frame images cannot be shown to the (text-only) analyst; the per-dimension
scores plus the reference JSON are the substitute signal.

## 4. Step outcomes: accept / reject / skip

Every training step ends in exactly one of three outcomes, recorded as the
`action` field in `history.json` (per step), aggregated in `summary.json`
(`total_accepts` / `total_rejects` / `total_skips`, `epoch_stats`), with full
detail in `steps/step_NNNN/step_record.json`:

- **accept** — the analysts produced patches, the merged edit was applied, and
  the edited skill scored **at least as well** on the validation gate
  (`valid_seen`, metric `soft`) as the current skill. The skill is updated;
  if it also beats the best-so-far, `best_skill.md` is refreshed.
- **reject** — patches were produced and merged, but the edited skill scored
  **worse** on the gate. The edit is discarded and the previous skill is kept
  (the rejection is summarized into the step buffer so later analysts can
  avoid repeating it).
- **skip_no_patches** — the reflection stage produced no patches at all, so
  there was nothing to gate. Occasional skips are normal (e.g. a batch with
  nothing actionable); *all* steps skipping means reflection is broken — see
  below.

A healthy run therefore shows a mix of accepts and rejects; rejects are not
failures, they are the gate doing its job.

## 5. How to detect the silent-skip failure mode

After any run, check `summary.json`:

| Signal | Healthy | Broken |
| --- | --- | --- |
| `total_accepts + total_rejects` | > 0 | 0 (all `skips`) |
| `best_origin` | `step_N` | `initial_skill` |
| `token_summary` | has analyst/optimizer entries | `rollout` only |
| per-step `reflect_s` | seconds | `0.0` |
| `steps/step_*/patches/` | patch JSON files | empty |

A smoke run is only meaningful if it shows `failure_patches > 0` in the step
logs and at least one `accept`/`reject` in `history.json`.

## 6. Lesson for future env ports

When porting an environment onto SkillOpt, satisfying the `EnvAdapter`
abstract methods is necessary but not sufficient. Trace the full step
pipeline (rollout → reflect → merge → gate) once with a real or mocked run and
verify each stage consumes what the previous one produced — in particular that
`predictions/<id>/conversation.json` exists and is non-empty.
