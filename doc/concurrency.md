# Concurrency Model, Knobs, and Sizing

**English** | [中文](concurrency.zh.md)

Everything in this project runs concurrent work with **single-process thread
pools** (`ThreadPoolExecutor`). There is no `multiprocessing`, no fork/spawn,
no client-server split — which means behavior is **identical on macOS and
Linux**. This document records why, lists every concurrency knob, and gives
sizing advice.

## 1. Why threads (and why agent-lightning chose processes)

The old APO project ran on agent-lightning, whose concurrency behaved
differently on macOS vs Linux. That is not an implementation bug but a
consequence of its architecture: agent-lightning is a general agent-RL
framework and uses a server-client, multi-process design because it must

1. **decouple rollouts from a GPU trainer** — workers poll tasks over HTTP and
   can run on other machines; threads cannot leave the process;
2. **run arbitrary user agent code** — possibly CPU-bound (GIL contention) or
   thread-unsafe, so process isolation is the only safe container;
3. **kill hung rollouts** — Python threads cannot be force-killed, processes
   can.

The price is OS-dependent semantics: Linux defaults `multiprocessing` to
`fork`, macOS to `spawn` (fork is unsafe next to the Objective-C runtime), and
macOS ships a low default file-descriptor limit. Hence "works on Linux, flaky
on macOS".

This project needs none of those three properties: it is a single machine, a
fixed code path (Azure OpenAI + Blob HTTP calls, pure I/O wait — the GIL is
irrelevant), and hangs come from API calls that timeouts + retries already
bound. Threads are the lightest tool that fits, with zero cross-platform
divergence.

## 2. The concurrency knobs

| Knob | Where | What it parallelizes | Default |
| --- | --- | --- | --- |
| `env.workers` (YAML) | `video2frames_env/rollout.py` `run_batch` | rollouts within one batch/eval: each worker does one target call + one judge call per task | 4 (currently 12) |
| `gradient.analyst_workers` (YAML) | skillopt `gradient/reflect.py` | error/success analyst minibatch calls to the optimizer model within one step | 4 |
| `--probe-workers` (CLI) | `prepare_data.py` `resolve_frames` | frame-blob listing + content-filter probes during data prep | 8 |

Notes:

- The pipeline stages of a training step run **sequentially** (rollout →
  reflect → merge → gate eval), so the knobs do not stack: peak in-flight
  requests is roughly `max(env.workers, analyst_workers)`, not the product.
- Gate evals and test evals reuse `env.workers`.
- `--probe-workers` resolves candidates in order-preserving chunks, so the
  selected split is **byte-identical to a serial run** (blocked-ness is a
  property of the video, not of probing order).
- Probe results are cached per task id in `data/content_filter_cache.json`
  (flushed after every probe, thread-safe); re-runs skip everything already
  probed.

## 3. Retry layers (what saves you from transient failures)

| Call | Retry | Behavior |
| --- | --- | --- |
| target + optimizer calls | skillopt `_chat_impl`, `retries=3` | retries **all** exceptions, exponential sleep, **silent** (no log lines) |
| judge call | `evaluator.py judge_text_fields`, 3 attempts | retries 429/5xx/network with backoff, logs each attempt; raises afterwards (task then scores 0 with `fail_reason`) |
| content-filter probe | `probe_content_filter.py probe_task`, 3 attempts | retries Azure's transient 400 "Timed out while downloading image"; persistent timeout ⇒ video excluded as unusable |
| Blob listing/download | Azure Storage SDK built-in policy | automatic |

The silent skillopt retry has one observable symptom worth knowing: a step
that sits quietly for many minutes (e.g. a `reflect_s` of 1200 s with only
~1 k completion tokens) is almost always the optimizer deployment being
throttled (429) behind that silent loop — check the Azure portal metrics, not
the process.

## 4. Sizing recommendations

1. **The real ceiling is Azure quota, not the OS or the pool size.** Each
   rollout sends every frame image (~28 k prompt tokens per task); raising
   `env.workers` beyond what the target deployment's TPM allows just converts
   parallelism into 429 retries. Size against the deployment quota:
   start at 8–12 for `gpt-4.1-mini` at default quota, drop back if you see
   throttling stalls.
2. **`analyst_workers` rarely needs tuning.** A step produces only a handful
   of minibatches (`ceil(failures/M) + ceil(successes/M)`), so 4 workers
   already saturate; the optimizer model's latency dominates.
3. **`--probe-workers` 8 is a good default.** Probes are `max_tokens=1`
   requests but carry full frame payloads; the same TPM logic as rollouts
   applies. The probe cache makes interrupted runs cheap to resume, so err on
   the side of fewer workers.
4. **Judge volume scales with eval size, not workers.** Moving
   `sel_env_num` from 24 to 100 quadruples judge calls per gate eval; if the
   judge deployment throttles, gate evals slow down uniformly (they share
   `env.workers`).
5. If a run must survive throttling storms, resume is always available:
   re-run with `--cfg-options env.out_root=<existing run dir>` — completed
   steps, rollout results, and finished minibatch patches are all reused.
