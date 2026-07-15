# Score Design and Open Questions for the Customer

**English** | [中文](reward-design.zh.md)

The soft score is the optimization target: SkillOpt edits the skill in whatever
direction the selection gate scores higher. If the score encodes the wrong
priorities, the optimizer will optimize the wrong thing — precisely. This
document records how the current score is defined, why, which parts are
assumptions that only the customer can confirm, and how to run that
conversation before a large-scale training run.

## 1. Current definition

Implemented in `video2frames_env/evaluator.py` (weights at the top, scoring in
`compute_scores`, judge in `judge_text_fields`):

```
soft = 0.2 × exact match of scene_type          (case-insensitive)
     + 0.2 × exact match of is_courier_action   (tolerates "true"/"false" strings)
     + 0.6 × LLM-judge semantic score over english_detail / brief / title

hard = 1 if soft >= 0.8 else 0                  (env.hard_threshold, default 0.8)
```

Two hard zero rules:

- Output that is not a valid JSON object → `0` (`fail_reason="invalid_json"`).
- Request rejected by the Azure content safety filter → `0`
  (`fail_reason="content_filter"`; the rejection depends only on the input
  frames, identical for every candidate skill).

The judge (configured via `env.judge_model` in the YAML config,
`temperature=0`, structured
output with `reason` + `score`) is instructed to check whether the generated
text "describes the same subjects and actions" as the ground truth, wording may
differ, be critical, partial credit allowed. It returns **one combined 0–1
score** for all three text fields.

This formula is byte-compatible with the old APO project's reward, so scores
from both optimizers are directly comparable.

## 2. Why it is designed this way

1. **Split by field nature.** Of the five output fields, `scene_type`
   (indoor/outdoor) and `is_courier_action` (bool) are objectively checkable —
   exact match is free, noise-free, and unambiguous. The three free-text fields
   can never match exactly, so semantic comparison by an LLM judge is the only
   practical grader.
2. **The optimizer needs a continuous signal.** With exact matches alone the
   score would take only five discrete values; the reflection analysts would
   have almost nothing to critique. The judge's partial credit distinguishes
   "slightly off description" from "completely wrong", which is what the
   failure analysis feeds on. `soft` drives the gate (`evaluation.gate_metric:
   soft`); `hard` plays two roles: during reflection, rollouts with
   `hard == 0` are the *failure* bucket the error analyst mines for patches
   (`env.hard_threshold` therefore controls how aggressively tasks are
   treated as failures), and after training it is the shippable-quality
   headline number. Being binary, `hard` is deliberately not used for
   gating — it cannot see incremental improvements that `soft` can.
3. **Weights follow content share.** The three text fields are the bulk of the
   output (and the part a prompt can influence most), hence `0.6`; the two
   classification fields get `0.2` each.
4. **The zero rules remove non-prompt noise.** Invalid JSON means the format
   contract is broken (downstream cannot consume the output — punish hard).
   Content-filter rejections are independent of the candidate skill, so scoring
   them 0 (and excluding those videos at sampling time via the probe cache)
   keeps them from polluting comparisons.

## 3. How the reward drives optimization (the RL analogy)

SkillOpt's training loop is reinforcement learning transplanted into text
space. The reward defined above is consumed at these points:

| RL concept | SkillOpt equivalent |
| --- | --- |
| policy | the skill text (the prompt) itself |
| reward | `soft` (continuous) + `hard` (success signal) |
| gradient | the analysts' written critiques + patches ("textual gradients") |
| step size | `gradient.edit_budget` (max edits per patch), `optimizer.learning_rate` |
| trust region / conservative update | the validation gate — an edited skill must score ≥ current on `valid_seen` or the edit is rejected and rolled back |
| replay / avoiding repeated bad actions | the step buffer — rejected edits are summarized and shown to later analysts |

The loop learns from **both** positive and negative examples. Each step,
`hard` splits the rollout batch into two buckets
(`skillopt/gradient/reflect.py`):

- **failures** (`hard == 0`) → the *error analyst*: cross-trajectory root-cause
  analysis, proposes corrective patches;
- **successes** (`hard == 1`) → the *success analyst*: extracts behavior
  patterns common across multiple successful trajectories and not yet covered
  by the skill, so effective behavior gets reinforced rather than lost.

Both analysts run when `gradient.failure_only: false` (this project's
default). The key difference from numeric RL: credit assignment is not
estimated from many samples — the optimizer LLM reads the trajectories and
writes the update direction directly. One batch yields a structured edit; the
cost is that update quality depends entirely on the optimizer model's
judgment.

Two practical consequences:

1. **`env.hard_threshold` is a static config knob** (default 0.8), not
   adaptive — it is the failure/success cut line, so it controls how
   aggressively tasks are mined as failures. Tune it manually after
   inspecting the `soft` distribution; SkillOpt never adjusts it. (SkillOpt
   itself allows a continuous `hard` 0.0–1.0, "smoothed reward"; this env
   emits the binary form.)
2. **Spend model capability where the leverage is.** The optimizer performs
   the hardest reasoning in the loop (multi-trajectory credit assignment) and
   is called only a few times per step — upgrading it is cheap and
   high-leverage; use the strongest model available. The judge is a
   constrained rubric-grading task called once per rollout (high volume);
   consistency matters more than raw capability — and it must stay **fixed**
   across any runs you compare, because changing the judge changes the score
   scale.

## 4. What only the customer can answer

These are assumptions baked into the score. Getting them wrong means the
optimizer optimizes a precisely wrong target, so confirm them **before** the
large run:

| # | Question | Why it matters | If the answer differs |
| --- | --- | --- | --- |
| 1 | Is `is_courier_action` the business-critical signal (this looks like a courier/delivery detection product)? Are false positives and false negatives equally bad? | At weight 0.2 a skill edit that fixes courier detection gains little score; the optimizer will prioritize text quality instead. Misclassification costs are usually asymmetric. | Raise `COURIER_WEIGHT`; replace symmetric exact match with asymmetric scoring (e.g. missed courier costs more than a false alarm). |
| 2 | Which text field is actually consumed downstream — `brief` (user-facing?), `english_detail` (search/archive?), `title`? | The judge currently emits one combined score; a skill that improves the important field while degrading an unimportant one scores flat. | Split the judge into per-field scores with separate weights. |
| 3 | How was the ground truth produced — human annotation or model-generated (the dataset name suggests SFT distillation)? Known quality issues? | The judge grades *against the GT*. Noisy GT both caps the achievable score and can steer tuning toward reproducing GT artifacts. | Clean or re-annotate a subset for val/test; or instruct the judge to tolerate specific GT quirks. |
| 4 | What improvement is worth shipping (e.g. +0.05 average soft score, or +X pp courier accuracy)? | This is the effect size `δ` in [dataset-sizing.md](dataset-sizing.md) — it determines how large val/test must be and when to stop tuning. | Resize the splits with the sizing formula before the run. |
| 5 | Is downstream parsing strict JSON, or tolerant (e.g. strips markdown fences)? | We currently score any non-JSON output 0 — the harshest possible penalty. (`parse_model_output` already strips one fenced block.) | Relax the parser / partial credit for recoverable outputs. |
| 6 | Can the customer hand-score 10–20 sample outputs? | Calibrates the LLM judge. If judge scores do not correlate with human judgment, the judge rubric must be fixed *before* tuning — it is the examiner of the whole system. | Iterate on the judge prompt / model until correlation is acceptable. |

Questions 1–4 should be settled before spending on a full run; 5–6 are cheap to
check in parallel.

## 5. Suggested next steps

1. **Send the customer a short brief** (this document works): the score
   formula, the six questions above, plus 2–3 concrete scored examples from an
   eval run (`outputs/<eval run>/predictions/<id>/rollout.json`) so the
   discussion is grounded in real outputs rather than abstractions.
2. **Run the pilot in parallel** (Stage 1 of
   [dataset-sizing.md](dataset-sizing.md), default 40/24/30 splits) — it
   measures score noise σ and produces the example outputs for step 1, and
   nothing in it is wasted even if the weights change later.
3. **Fold the answers back in.** Weight changes are three constants
   (`SCENE_WEIGHT` / `COURIER_WEIGHT` / `JUDGE_WEIGHT` in
   `video2frames_env/evaluator.py`); per-field judging or asymmetric courier
   scoring are small, local edits to `judge_text_fields` / `compute_scores`
   with matching unit tests in `tests/test_evaluator.py`.
4. **Only then run the full training ladder** (Stage 2+). Changing the score
   after a big run means paying for the run again — the score conversation is
   the cheapest insurance in the whole project.
