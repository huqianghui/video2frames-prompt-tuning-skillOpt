# custom_prompt/

Project-specific overrides for SkillOpt's reflection prompts.

Drop a file here with the same name as one in `skillopt_prompts/` (e.g.
`analyst_error.md`) and it replaces the vendored default: `train.py` syncs
this directory into `site-packages/skillopt/prompts/` on every run, so edits
take effect on the next training run without reinstalling anything.

With the default `skill_update_mode=patch`, the prompts actually used are:

- `analyst_error.md` / `analyst_success.md` — how the optimizer analyzes
  failed/successful rollouts and proposes patches (≈ APO's gradient prompt)
- `merge_failure.md` / `merge_success.md` / `merge_final.md` — how patches
  are merged into one skill update (≈ APO's apply-edit prompt)
- `ranking.md` — candidate ranking

The `_rewrite` / `_full_rewrite` variants belong to the other update modes.
