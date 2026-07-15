# video2frames-prompt-tuning-skillOpt

Prompt tuning for a video-frame-analysis agent using
[SkillOpt](https://github.com/microsoft/SkillOpt) — a text-space optimizer with
validation-gated, bounded skill edits. This is a port of the agent-lightning /
APO-based `video2frames-prompt-tuning` project to the SkillOpt training loop;
the scoring is byte-compatible so results are directly comparable.

The task: given N frames sampled from a short video (4s apart, delivered as
Azure Blob SAS URLs), the target model must produce a structured JSON
description (`english_detail`, `brief`, `title`, `scene_type`,
`is_courier_action`). The tuned text ("skill") is the instruction prompt that
precedes the frames.

## How it maps to SkillOpt

| SkillOpt concept | This project |
| --- | --- |
| skill (tuned text) | the instruction prompt, seeded from `video2frames_env/skills/initial.md` |
| env adapter | `Video2FramesAdapter` (`video2frames_env/adapter.py`), registered as `video2frames` |
| rollout | one multimodal call per task: skill text + `<frame n \| Xs>` labels interleaved with frame images (`video2frames_env/rollout.py`) |
| hard / soft score | `hard = int(soft >= 0.8)`; `soft = 0.2·scene_match + 0.2·courier_match + 0.6·judge_score` — identical to the old APO reward (`video2frames_env/evaluator.py`) |
| selection gate | `val` split (`valid_seen`), metric `soft` |
| held-out test | `test` split (`valid_unseen`) |

## Included Files

| File | Role |
| --- | --- |
| `train.py` | Entry point: loads `.env`, installs missing SkillOpt prompt files, registers the adapter, delegates to `scripts.train` (SkillOpt CLI). |
| `eval.py` | Eval-only entry point: run any skill file on any split via `scripts.eval_only`. |
| `install_prompts.py` | Backfills `skillopt/prompts/*.md` reflection prompts missing from the skillopt 0.2.0 wheel (pinned upstream commit). |
| `configs/video2frames/default.yaml` | Structured SkillOpt config (model, train, gradient, optimizer, evaluation, env sections). |
| `video2frames_env/adapter.py` | `Video2FramesAdapter(EnvAdapter)` — wires dataloader and rollout together. |
| `video2frames_env/dataloader.py` | `FrameDataLoader(SplitDataLoader)` over `data/splits/{train,val,test}`. |
| `video2frames_env/rollout.py` | Multimodal rollout with `results.jsonl` resume and content-filter short-circuit. |
| `video2frames_env/evaluator.py` | JSON parsing, exact-match + LLM-judge scoring (`hard`/`soft`). |
| `video2frames_env/tasks.py` | Task schema (`FrameTask`), data paths, model name helpers. |
| `video2frames_env/skills/initial.md` | Initial skill — byte-identical to the old project's `baseline_prompt.txt`. |
| `prepare_data.py` | Builds stratified train/val/test splits from `original_data/` and mirrors them into `data/splits/`. |
| `probe_content_filter.py` | Probes which videos the Azure content filter blocks; cached so blocked tasks score 0 without burning requests. |
| `blob_utils.py` | `.env` loading and Azure Blob SAS URL construction. |
| `doc/dataset-sizing.md` | How to size the splits relative to evaluation noise ([中文](doc/dataset-sizing.zh.md)). |
| `doc/reward-design.md` | Scoring rationale and customer questions ([中文](doc/reward-design.zh.md)). |
| `doc/reflection-trajectories.md` | The `conversation.json` contract reflection depends on, and the silent-skip incident it caused ([中文](doc/reflection-trajectories.zh.md)). |
| `tests/` | Offline test suite — all network calls mocked. |

`data/`, `original_data/`, `logs/`, `outputs/` are never committed (only
`.gitkeep`); copy data in locally.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in real credentials
```

`.env` must provide the Azure OpenAI endpoint/key/deployment and the Blob
connection settings (see `.env.example`). `train.py`/`eval.py` load it before
any skillopt import and default the target/optimizer deployment to
`AZURE_OPENAI_DEPLOYMENT`.

## Data preparation

Copy the source dataset into `original_data/` (e.g.
`qwen_0318_swift_task.json`), then:

```bash
python prepare_data.py --train-size 40 --val-size 24 --test-size 30 --seed 42 --probe-content-filter
```

This writes `data/{train,val,test}.jsonl` plus the SkillOpt-style mirror
`data/splits/{train,val,test}/items.json`. If splits already exist and only the
mirror is needed: `python prepare_data.py --mirror-only`.

## Training

```bash
python train.py --config configs/video2frames/default.yaml
```

Outputs land in `outputs/skillopt_video2frames_<optimizer>_<timestamp>/`
(override with `--cfg-options env.out_root=...`): `history.json` (per-step
record), `best_skill.md`, `summary.json`, per-step rollout artifacts.

Any config key can be overridden from the CLI, e.g.:

```bash
python train.py --config configs/video2frames/default.yaml \
    --cfg-options train.num_epochs=2 train.batch_size=4 optimizer.learning_rate=4
```

## Evaluating a skill

```bash
python eval.py --config configs/video2frames/default.yaml \
    --skill outputs/<run>/best_skill.md --split valid_unseen
```

`--split` accepts `train` / `valid_seen` (val) / `valid_unseen` (test) / `all`;
results go to `outputs/eval_.../eval_summary.json`.

## Smoke test

Offline (no network, ~1s):

```bash
.venv/bin/python -m pytest -q
```

Online (a few dollars of API calls, ~2 min): limit everything to 4 items and
run one epoch, then eval the resulting best skill:

```bash
python train.py --config configs/video2frames/default.yaml \
    --cfg-options train.num_epochs=1 train.batch_size=2 env.limit=4 env.workers=1 \
    evaluation.sel_env_num=4 evaluation.test_env_num=4 env.out_root=outputs/smoke_epoch

python eval.py --config configs/video2frames/default.yaml \
    --skill outputs/smoke_epoch/best_skill.md --split valid_unseen \
    --cfg-options env.limit=2 env.workers=1 env.out_root=outputs/smoke_eval
```

Expect `outputs/smoke_epoch/history.json` with one record per step and a final
test summary printed at the end. Also verify the optimizer actually engaged:
step logs must show `failure_patches > 0` and `summary.json` must have
`total_accepts + total_rejects > 0` — if every step is `skip_no_patches`, the
reflection stage received no trajectories (see
`doc/reflection-trajectories.md`) and the run only re-evaluated the initial
skill.

## Comparing against the old APO project

- Epoch-0 baseline `soft` on test should match the old project's baseline
  reward (the scoring port is lossless — verified on shared task IDs).
- Compare `outputs/<run>/best_skill.md` against the old
  `results/best_prompt.txt` to see whether both optimizers learn the same
  rules.
