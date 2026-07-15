# 反思轨迹：`conversation.json` 契约

[English](reflection-trajectories.md) | **中文**

SkillOpt 的优化步骤只有在反思阶段能读到每条 rollout 的轨迹时才会发生。这个要求是一个
**隐式**契约——它不在 `EnvAdapter` 接口里——而且缺失时不报任何错误：训练会静默退化成
对初始 skill 的反复评估。本文档记录了这个问题实际发生的一次事故、根因、修复方式，以及
如何检测这种失败模式。

## 1. 事故（2026-07-15）

一次完整训练（`outputs/skillopt_video2frames_gpt-4.1_20260715_094345`，
4 epochs × 5 steps，约 149 万 prompt tokens）产出的 `best_skill.md` 与初始 skill
逐字节相同。`summary.json` 说明了一切：

```
"total_steps": 20,
"total_accepts": 0,
"total_rejects": 0,
"total_skips": 20,          <- 每一步都是 "skip_no_patches"
"best_origin": "initial_skill",
"token_summary": { "rollout": ... }   <- 完全没有 analyst/优化器调用
```

同一个初始 prompt 在 30 条 test 集上的三次评估 `soft` = 0.769 / 0.754 / 0.778 ——
±0.015 的散布纯粹是 judge 噪声，不是优化进展。

## 2. 根因

反思管线（`skillopt.gradient.reflect`）通过 `fmt_minibatch_trajectories` 构造
analyst 输入，它读取每条 rollout 结果对应的
`predictions/<task_id>/conversation.json`，文件不存在时**静默跳过**：

```python
conv_path = os.path.join(prediction_dir, tid, "conversation.json")
if not os.path.exists(conv_path):
    continue
```

而本项目的 rollout 只写了 `predictions/<id>/rollout.json`。于是：

1. 所有轨迹被跳过 → 拼出的轨迹文本为空；
2. `run_error_analyst_minibatch` 对空输入直接返回 `None`，**根本不会**调用优化器模型；
3. 零 patch → trainer 记录 `skip_no_patches` 继续下一步。

每个 step 记录里 `reflect_s: 0.0`、`patches/` 目录为空，都印证了这一点。

### 为什么移植时会漏掉

- **这个要求不在 adapter 契约里。** 抽象方法 `EnvAdapter.rollout()` 的文档只要求返回
  带 `id`/`hard`/`soft` 的字典列表。`conversation.json` 是各内置 env（alfworld、
  searchqa、docvqa……）各自实现的约定，只记载在 `gradient/reflect.py` 辅助函数的
  docstring 里。
- **失败是设计上静默的。** 文件缺失被跳过、analyst 返回 `None`、`skip_no_patches`
  是合法的步骤结果——没有报错，没有警告。
- **测试和冒烟标准都验不出来。** 离线测试覆盖 rollout/打分/数据加载；反思阶段在
  skillopt 内部运行且需要真实模型调用。冒烟验收标准（"`history.json` 每步一条记录、
  存在 `best_skill.md`"）在全部 skip 时照样满足。
- 佐证：adapter 实现了 `build_reference_text`，期望 skillopt 通过
  `attach_reference_context` 挂载参考答案，但锁定的 0.2.0 wheel 里该方法**零调用**——
  说明移植是对着 adapter API 表面写的，没有追踪反思阶段的数据流。

## 3. 修复

`video2frames_env/rollout.py` 现在对**每个**任务（成功、content filter、报错——
失败轨迹恰恰是 error analyst 最需要的）都写出：

- `predictions/<id>/rollout.json` —— 评分 + 参考答案，同以前；
- `predictions/<id>/conversation.json` —— analyst 读取的轨迹：

```json
[
  {"role": "user",      "content": "<任务描述> (frame images omitted from this trace)"},
  {"role": "assistant", "content": "<模型输出，或 '(no response)'>"},
  {"role": "system",    "content": "scene_match=.. courier_match=.. judge_score=.. soft=.. hard=..\nfail_reason: .."}
]
```

每条结果字典还附带 `reference_text`（ground-truth JSON），
`fmt_minibatch_trajectories` 会以 "Hidden Reference" 标题呈现——analyst 可以对照
输出与期望，而参考答案永远不会进入 rollout prompt。

帧图片无法展示给（纯文本的）analyst；分维度评分加参考 JSON 是替代信号。

## 4. 步骤结果：accept / reject / skip

每个训练步骤以三种结果之一结束，记录在 `history.json` 每步的 `action` 字段，
汇总在 `summary.json`（`total_accepts` / `total_rejects` / `total_skips`、
`epoch_stats`），单步完整详情在 `steps/step_NNNN/step_record.json`：

- **accept** —— analyst 产出了 patch，合并后的编辑被应用，且编辑后的 skill 在验证
  门控（`valid_seen` 集，指标 `soft`）上得分**不低于**当前 skill。skill 被更新；
  如果同时超过历史最优，`best_skill.md` 也会刷新。
- **reject** —— 产出并合并了 patch，但编辑后的 skill 在门控上得分**更差**。编辑被
  丢弃，保留原 skill（被拒的编辑会摘要进 step buffer，供后续 analyst 避免重复）。
- **skip_no_patches** —— 反思阶段完全没有产出 patch，没有东西可以门控。偶尔的 skip
  是正常的（比如某个 batch 没有可改进点）；*所有*步骤都 skip 说明反思阶段坏了——
  见下节。

因此健康的训练是 accept 和 reject 混合出现的；reject 不是失败，恰恰是门控在起作用。

## 5. 如何检测静默 skip 失败模式

每次训练后检查 `summary.json`：

| 信号 | 健康 | 异常 |
| --- | --- | --- |
| `total_accepts + total_rejects` | > 0 | 0（全部 `skips`） |
| `best_origin` | `step_N` | `initial_skill` |
| `token_summary` | 有 analyst/优化器条目 | 只有 `rollout` |
| 每步 `reflect_s` | 数秒 | `0.0` |
| `steps/step_*/patches/` | 有 patch JSON 文件 | 空 |

冒烟只有在步骤日志出现 `failure_patches > 0`、且 `history.json` 里至少有一次
`accept`/`reject` 时才算通过。

## 6. 给后续 env 移植的教训

把环境移植到 SkillOpt 上时，实现完 `EnvAdapter` 的抽象方法只是必要条件而非充分条件。
用一次真实或 mock 的运行把完整步骤管线（rollout → reflect → merge → gate）追一遍，
确认每个阶段都消费到了上一阶段的产物——特别是
`predictions/<id>/conversation.json` 存在且非空。
