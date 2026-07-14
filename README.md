# video2frames-prompt-tuning-skillOpt

Prompt tuning for a video-frame-analysis agent using
[SkillOpt](https://github.com/microsoft/SkillOpt) — a text-space optimizer with
validation-gated, bounded skill edits. This is a port of the
agent-lightning/APO-based `video2frames-prompt-tuning` project to the SkillOpt
training loop.

Status: work in progress — features land commit by commit.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in real credentials
```

Data (`original_data/`, `data/`) and run artifacts (`logs/`, `outputs/`) are
never committed; only the empty directory structure is tracked.
