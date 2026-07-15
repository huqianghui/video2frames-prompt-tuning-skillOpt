# Env Architecture: Why These Components, and What a Customer Implements

**English** | [中文](env-architecture.zh.md)

The old APO project on agent-lightning needed one rollout function plus a
reward emitted through tracer labels. This project has a whole package —
`video2frames_env/` with `tasks.py`, `dataloader.py`, `rollout.py`,
`evaluator.py`, `adapter.py` — and thin `train.py`/`eval.py` entry points
that just register the adapter. This document explains why the shape changed,
what stayed the same, and which parts a customer actually has to write when
porting to a new task.

## 1. Both frameworks ask for the same three things

Any prompt/skill optimizer needs the customer to provide exactly three
pieces of logic: **data** (what a task looks like), **execution** (how to
turn skill + task into a model output), and **reward** (how good the output
is). The frameworks differ only in the shape of the slots:

| Customer logic | agent-lightning (old APO) | SkillOpt (this repo) |
| --- | --- | --- |
| Data: task schema and feeding | read files yourself, post task dicts to the server | `tasks.py` (schema) + `dataloader.py` (splits/batches) |
| Execution: run one task | a rollout function (workers poll for tasks) | `rollout.py` (frame URLs + target model call) |
| Reward: score the output | compute reward, emit via tracer labels | `evaluator.py` (scene/courier hard checks + judge soft score) |
| Wiring: how the framework sees input/output/reward | tracer collects triplets from labels implicitly | `adapter.py` implements the `EnvAdapter` interface explicitly |
| Entry point | framework CLI | `train.py` / `eval.py` (~40 lines: register + delegate) |

## 2. Why an "env", and why train.py registers-then-delegates

"Env" is RL vocabulary: the **trainer is the generic optimization loop, the
env is the task-specific world**. SkillOpt's trainer implements rollout →
reflect → merge → gate eval, resume, token accounting, and gate decisions
once; everything about *this particular task* (data, execution, scoring)
lives behind the `EnvAdapter` interface. Same idea as gym's `env.step()`:
the algorithm does not know whether it is playing Atari or optimizing a
courier-video prompt. Switching to a new customer task means swapping the
env — the trainer changes by zero lines.

The thin `train.py` wrapper exists because skillopt is an **installed
third-party package**: its CLI (`scripts.train`) ships a registry of
built-in envs and knows nothing about ours. The entry point is the
composition root — the only place that knows both sides:

```python
skillopt_train._ENV_REGISTRY["video2frames"] = Video2FramesAdapter  # register
skillopt_train.main()                                               # delegate
```

That one registration line is what lets us use the upstream trainer with
zero modification and zero fork: resume, gating, and token accounting come
for free, and upgrading skillopt never touches our code. It is the same
pattern as pytest plugins or Django apps: a generic engine plus business
modules registered into it.

## 3. Small-implicit vs large-explicit contracts

**agent-lightning: a small, implicit contract.** You write one function and
tag rewards; a tracer collects the (input, output, reward) triplets behind
the scenes. Minimal code to get started, but the data flow is invisible —
when a reward silently fails to be collected there is little to inspect, and
the server-client architecture brings the OS-dependent concurrency issues
documented in [concurrency.md](concurrency.md).

**SkillOpt: a large, explicit contract.** Five files instead of one, but
each has a single responsibility and is mostly pure functions, so all of it
is unit-testable offline — this repo's test suite (76 tests, zero network
calls) is a direct payoff of that structure. SkillOpt also adds one
requirement agent-lightning has no equivalent of: the **reflection
trajectory contract** — every task must persist a `conversation.json` so the
optimizer can read full input/output/feedback during reflection (see
[reflection-trajectories.md](reflection-trajectories.md)). That is the price
of replacing a scoring black box with textual gradients.

Note that "prompt became skill" is only a rename: the skill
(`video2frames_env/skills/initial.md`, byte-identical to the old
`baseline_prompt.txt`) is still the system prompt sent to the target model.
The prompts that drive the optimizer itself are separate meta-prompts
(`skillopt_prompts/`, overridable via `custom_prompt/`).

## 4. What a customer actually writes, by design weight

1. **Reward design (`evaluator.py`) — the only part that needs real
   design.** The optimizer climbs whatever the reward measures; a wrong
   reward optimizes the wrong thing. This is why
   [reward-design.md](reward-design.md) is framed as customer-confirmation
   questions. Rewards port verbatim between frameworks — this repo's soft
   formula is copied unchanged from the APO project.
2. **Target execution (`rollout.py`, ~50 core lines):** how skill + task
   data become one model call. Business-specific but mechanical.
3. **Data ingestion (`tasks.py` + `dataloader.py`):** convert customer data
   into a task list. Purely mechanical.
4. **Glue (`adapter.py`, `train.py`):** near-boilerplate; copy and rename
   for a new task.

## 5. Bottom line

The effort is roughly equal on both frameworks, because items 1–3 are the
business itself — no framework writes them for you. The difference is the
shape of layer 4: agent-lightning hides it behind tracer conventions,
SkillOpt spells it out as files. For a project handed to a customer for
long-term maintenance, the explicit contract wins: testable, debuggable,
and identical behavior across platforms.
