# 并发模型、配置项与规模建议

[English](concurrency.md) | **中文**

本项目中所有并发工作都使用**单进程线程池**（`ThreadPoolExecutor`）。没有
`multiprocessing`，没有 fork/spawn，也没有 client-server 拆分——这意味着
在 macOS 和 Linux 上的行为**完全一致**。本文档记录选择线程的原因、列出
所有并发配置项，并给出规模建议。

## 1. 为什么用线程（以及 agent-lightning 为什么选进程）

旧的 APO 项目跑在 agent-lightning 上，其并发行为在 macOS 与 Linux 上不
一致。这不是实现上的 bug，而是其架构的必然结果：agent-lightning 是通用
的 agent-RL 框架，采用 server-client 多进程设计，因为它必须

1. **把 rollout 与 GPU 训练器解耦** —— worker 通过 HTTP 拉取任务，可以跑
   在其他机器上；线程无法离开进程；
2. **运行任意用户 agent 代码** —— 可能是 CPU 密集（GIL 竞争）或线程不安
   全的，进程隔离是唯一安全的容器；
3. **杀掉挂死的 rollout** —— Python 线程无法被强制终止，进程可以。

代价是操作系统相关的语义：Linux 的 `multiprocessing` 默认 `fork`，macOS
默认 `spawn`（fork 在 Objective-C 运行时旁边不安全），且 macOS 的文件描
述符默认上限很低。于是出现"Linux 上正常，macOS 上不稳定"。

本项目不需要上述三个特性：单机运行、代码路径固定（Azure OpenAI + Blob
的 HTTP 调用，纯 I/O 等待——GIL 无关紧要）、挂死都来自已经被超时+重试
兜住的 API 调用。线程是满足需求的最轻量工具，且零跨平台差异。

## 2. 并发配置项

| 配置项 | 位置 | 并行的内容 | 默认值 |
| --- | --- | --- | --- |
| `env.workers`（YAML） | `video2frames_env/rollout.py` `run_batch` | 一个 batch/eval 内的 rollout：每个 worker 对每个任务做一次目标调用 + 一次 judge 调用 | 4（当前设为 12） |
| `gradient.analyst_workers`（YAML） | skillopt `gradient/reflect.py` | 一个 step 内向优化器模型发起的错误/成功分析 minibatch 调用 | 4 |
| `--probe-workers`（CLI） | `prepare_data.py` `resolve_frames` | 数据准备时的帧 blob 列举 + 内容过滤探测 | 8 |

说明：

- 训练 step 的各个流水线阶段**顺序执行**（rollout → reflect → merge →
  gate eval），因此各配置项不会叠加：峰值并发请求约为
  `max(env.workers, analyst_workers)`，而不是它们的乘积。
- Gate eval 和 test eval 复用 `env.workers`。
- `--probe-workers` 按保序分块解析候选，因此选出的切分与串行运行
  **逐字节一致**（是否被屏蔽是视频本身的属性，与探测顺序无关）。
- 探测结果按 task id 缓存在 `data/content_filter_cache.json`（每次探测后
  立即落盘，线程安全）；重跑会跳过所有已探测过的条目。

## 3. 重试层（帮你扛住瞬时故障的机制）

| 调用 | 重试 | 行为 |
| --- | --- | --- |
| 目标 + 优化器调用 | skillopt `_chat_impl`，`retries=3` | 重试**所有**异常，指数退避，**静默**（无日志） |
| judge 调用 | `evaluator.py judge_text_fields`，3 次尝试 | 重试 429/5xx/网络错误并退避，每次尝试都打日志；耗尽后抛出（该任务计 0 分并记录 `fail_reason`） |
| 内容过滤探测 | `probe_content_filter.py probe_task`，3 次尝试 | 重试 Azure 的瞬时 400 "Timed out while downloading image"；持续超时 ⇒ 该视频按不可用排除 |
| Blob 列举/下载 | Azure Storage SDK 内置策略 | 自动 |

skillopt 的静默重试有一个值得记住的可观察症状：某个 step 安静地卡很多
分钟（例如 `reflect_s` 达 1200 秒而 completion token 只有约 1k），就是
优化器调用被困在那个静默循环里。如果优化器配额充足（reflect 只有 3–4
个并发纯文本调用，远低于任何真实 TPM 上限），原因通常**不是**你自己的
配额，而是以下二者之一：

1. **区域容量限流** —— 标准（按量付费）部署在区域算力紧张时即使没用完
   配额也会收到 429，对最新模型尤其常见。查部署的 Azure 指标里的
   "Throttled Requests"：利用率很低却有 429 就是容量问题，换 Global
   Standard / Data Zone 部署类型或换区域可解；
2. **推理模型的生成耗时** —— reasoning token 会被生成但不显示在
   completion 里；一个复杂的 analyst minibatch 本身就可能合理地跑几分
   钟，叠加一次超时+重试就翻倍。

无论哪种：去查 Azure 门户指标，而不是查进程。

## 4. 规模建议

每个流水线阶段访问的是不同的部署，所以每个配置项要对着它实际加压的那个
部署的配额来定：

| 阶段 | 加压的部署 | 负载特征 |
| --- | --- | --- |
| rollout / gate eval / test eval（`env.workers`） | target（`model.target`）+ judge（`env.judge_model`） | target：每任务 ~2.8 万 prompt token（帧图片）；judge：每任务 ~1–2k 纯文本 token |
| reflect / merge（`analyst_workers`） | optimizer（`model.optimizer`） | 3–4 个并发纯文本调用，每个几万 token |
| 数据准备探测（`--probe-workers`） | 探测模型（`--probe-model`） | 完整帧负载，`max_tokens=1` |

### 实例核算（来自一组真实部署的配额）

| 部署 | 配额（TPM / RPM） | 实测负载 | 结论 |
| --- | --- | --- | --- |
| gpt-4.1-mini（target） | 5M / 5k | gate eval 在 72.7 秒内跑完 100 任务 × ~2.8 万 token ≈ **2.3M TPM** | 当前瓶颈——还有约一倍 worker 余量 |
| gpt-5.4-mini（judge） | 3M / 3k | 同一次 eval 期间 ~0.3M TPM | 只用 ~10%，永远不是约束 |
| gpt-5.4（optimizer） | 2.5M / 25k | 每次 reflect 几万 token | 配额无关紧要；卡顿来自容量 429 或推理耗时（见 §3） |

据此定规模：100 任务 / 12 并发 ≈ 9 波 × ~8 秒 ≈ 每次 gate eval 73 秒。
把 `env.workers` 提到 24 后约 5 波 ≈ 40 秒，吞吐 ~4.2M TPM——仍在 5M
配额之内。再往上（例如 32）就会越过配额，把收益变回 429 重试。RPM 在
这里从来不是瓶颈（~85 请求/分钟 vs 上限 5000）。

1. **真正的上限是 Azure 配额，不是操作系统或线程池大小。**每个 rollout
   会发送全部帧图片（每任务约 2.8 万 prompt token）；把 `env.workers`
   提到目标部署 TPM 承受不了的程度，只会把并行度变成 429 重试。按部署
   配额来定：`gpt-4.1-mini` 默认配额下从 8–12 起步，看到限流卡顿就回调。
2. **`analyst_workers` 很少需要调。**一个 step 只产生少量 minibatch
   （`ceil(failures/M) + ceil(successes/M)`），4 个 worker 已经饱和；瓶颈
   在优化器模型的延迟。
3. **`--probe-workers` 8 是不错的默认值。**探测虽是 `max_tokens=1` 请求，
   但携带完整帧负载；与 rollout 相同的 TPM 逻辑同样适用。探测缓存让中断
   的运行几乎零成本恢复，所以宁可少开几个 worker。
4. **judge 调用量随 eval 规模增长，而不是随 worker 数。**把
   `sel_env_num` 从 24 提到 100 会让每次 gate eval 的 judge 调用翻四倍；
   若 judge 部署被限流，gate eval 会整体变慢（它们共用 `env.workers`）。
5. 如果一次运行必须扛过限流风暴，随时可以断点续跑：用
   `--cfg-options env.out_root=<已有运行目录>` 重新运行——已完成的
   step、rollout 结果和已完成的 minibatch patch 都会被复用。
