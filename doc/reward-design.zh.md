# 评分设计与需要客户确认的问题

[English](reward-design.md) | **中文**

soft 分数是优化目标：选择门控给哪个方向打分更高，SkillOpt 就朝哪个方向
修改 skill。如果分数编码了错误的优先级，优化器就会"精确地"优化错误的目
标。本文记录当前分数的定义、设计原因、哪些部分是只有客户才能确认的假
设，以及在大规模训练前如何完成这场对话。

## 1. 当前定义

实现于 `video2frames_env/evaluator.py`（权重在文件顶部，打分在
`compute_scores`，judge 在 `judge_text_fields`）：

```
soft = 0.2 × scene_type 精确匹配            （大小写不敏感）
     + 0.2 × is_courier_action 精确匹配     （容忍 "true"/"false" 字符串）
     + 0.6 × LLM judge 对 english_detail / brief / title 的语义打分

hard = 1 if soft >= 0.8 else 0              （env.hard_threshold，默认 0.8）
```

两条硬性零分规则：

- 输出不是合法 JSON 对象 → `0`（`fail_reason="invalid_json"`）。
- 请求被 Azure 内容安全过滤器拒绝 → `0`
  （`fail_reason="content_filter"`；拒绝只取决于输入帧，对每个候选
  skill 都相同）。

judge（由 YAML 配置的 `env.judge_model` 指定，`temperature=0`，结构化输出
含 `reason` + `score`）被指示检查生成文本是否"描述了与 ground truth 相
同的主体和动作"，措辞可以不同，要求严格，允许部分得分。它对三个文本字
段返回**一个合并的 0–1 分数**。

该公式与旧 APO 项目的 reward 逐字节兼容，两个优化器的分数可以直接对比。

## 2. 为什么这样设计

1. **按字段性质拆分。** 五个输出字段中，`scene_type`（indoor/outdoor）
   和 `is_courier_action`（bool）可以客观检查——精确匹配零成本、无噪
   声、无歧义。三个自由文本字段永远不可能精确匹配，LLM judge 的语义比
   较是唯一可行的打分方式。
2. **优化器需要连续信号。** 只用精确匹配，分数只有五个离散取值；反思
   分析器几乎没有可批判的素材。judge 的部分得分能区分"描述略有偏差"和
   "完全错误"，这正是失败分析所依赖的。`soft` 驱动门控
   （`evaluation.gate_metric: soft`）；`hard` 有两个作用：反思阶段
   `hard == 0` 的 rollout 进入 error analyst 挖掘 patch 的"失败"桶
   （因此 `env.hard_threshold` 实际控制失败挖掘的口径），训练之后它是
   "可交付质量"的头条指标。`hard` 是二元的，刻意不用于门控——它看不到
   `soft` 能看到的渐进式改进。
3. **权重跟随内容占比。** 三个文本字段是输出的主体（也是提示词最能影
   响的部分），所以给 `0.6`；两个分类字段各 `0.2`。
4. **零分规则剔除非提示词噪声。** 非法 JSON 意味着格式契约被破坏（下
   游无法消费输出——重罚）。内容过滤器的拒绝与候选 skill 无关，打 0 分
   （并在采样时通过 probe 缓存排除这些视频）可避免污染对比。

## 3. reward 如何驱动优化（与强化学习的类比）

SkillOpt 的训练循环是把强化学习搬进了"文本空间"，上文定义的 reward 在
以下位置被消费：

| RL 概念 | SkillOpt 对应物 |
| --- | --- |
| policy | skill 文本（prompt）本身 |
| reward | `soft`（连续）+ `hard`（成败信号） |
| 梯度 | analyst 产出的文字批评 + patch（"textual gradient"） |
| 步长 | `gradient.edit_budget`（每个 patch 最多几条编辑）、`optimizer.learning_rate` |
| 信任域 / 保守更新 | 验证门控——编辑后的 skill 在 `valid_seen` 上不劣于当前才 accept，否则 reject 回滚 |
| 经验回放 / 避免重复坏动作 | step buffer——被 reject 的编辑会摘要给后续 analyst |

这个循环**同时从正、负两类例子学习**。每一步 `hard` 把 rollout batch
分成两个桶（`skillopt/gradient/reflect.py`）：

- **failures**（`hard == 0`）→ *error analyst*：跨轨迹找共性根因，提出
  修正性 patch；
- **successes**（`hard == 1`）→ *success analyst*：提炼在多条成功轨迹中
  反复出现、且 skill 尚未覆盖的行为模式，把有效做法固化下来。

`gradient.failure_only: false`（本项目默认）时两个 analyst 都启用。与
数值 RL 的关键差异：credit assignment 不靠大量采样估计，而是由 optimizer
模型直接读轨迹、写出更新方向——一个 batch 就产出结构化编辑；代价是更新
质量完全取决于 optimizer 模型的判断力。

两个实践结论：

1. **`env.hard_threshold` 是静态配置项**（默认 0.8），不会自适应——它就
   是失败/成功的分界线，控制失败挖掘的口径。需要人工观察 `soft` 分布后
   手动调整；SkillOpt 不会替你调。（SkillOpt 本身允许连续的 `hard`
   0.0–1.0 即 "smoothed reward"；本环境按契约输出二元值。）
2. **把模型能力花在杠杆最大的地方。** optimizer 承担整个循环中最难的推
   理（跨轨迹 credit assignment），且每步只调用几次——升级它便宜且杠杆
   高，应使用可用的最强模型。judge 是受约束的对照打分任务，每条 rollout
   调用一次（量大）；一致性比智力更重要——而且在所有要互相比较的 run
   之间必须**保持不变**，因为换 judge 会改变分数量纲。

## 4. 只有客户能回答的问题

以下是烘焙进分数里的假设。答错意味着优化器会精确地优化一个错误目标，
所以务必在大规模训练**之前**确认：

| # | 问题 | 为什么重要 | 如果答案不同 |
| --- | --- | --- | --- |
| 1 | `is_courier_action` 是否是业务关键信号（这看起来是快递/配送检测产品）？漏报和误报是否同样糟糕？ | 权重 0.2 下，修复快递检测的 skill 修改得分收益很小；优化器会优先改文本质量。误分类代价通常是不对称的。 | 提高 `COURIER_WEIGHT`；把对称精确匹配换成不对称打分（例如漏检快递比误报代价更高）。 |
| 2 | 下游实际消费的是哪个文本字段——`brief`（面向用户？）、`english_detail`（搜索/归档？）还是 `title`？ | judge 目前输出一个合并分数；一个 skill 改善重要字段却损害不重要字段时分数持平。 | 把 judge 拆成按字段打分并分别加权。 |
| 3 | ground truth 是怎么产生的——人工标注还是模型生成（数据集名称暗示是 SFT 蒸馏）？已知质量问题？ | judge 是*对照 GT* 打分的。有噪声的 GT 既压低可达分数上限，也可能把调优引向复现 GT 的伪影。 | 清洗或重新标注 val/test 子集；或指示 judge 容忍特定的 GT 缺陷。 |
| 4 | 多大的提升才值得上线（例如平均 soft +0.05，或快递准确率 +X pp）？ | 这就是 [dataset-sizing.md](dataset-sizing.zh.md) 中的效应量 `δ`——它决定 val/test 需要多大、何时停止调优。 | 训练前用容量公式重新确定切分大小。 |
| 5 | 下游解析是严格 JSON 还是容错的（例如会剥掉 markdown 代码围栏）？ | 我们目前对任何非 JSON 输出打 0 分——最严厉的惩罚。（`parse_model_output` 已经会剥一层围栏代码块。） | 放宽解析器 / 对可恢复输出给部分得分。 |
| 6 | 客户能否人工评 10–20 条样例输出？ | 用于校准 LLM judge。如果 judge 分数与人工判断不相关，必须在调优*之前*修好 judge rubric——它是整个系统的主考官。 | 迭代 judge 提示词/模型直到相关性可接受。 |

问题 1–4 应在花钱做完整训练前敲定；5–6 可以并行低成本验证。

## 5. 建议的下一步

1. **给客户发一份简报**（本文即可）：分数公式、上面六个问题，外加 2–3
   个来自 eval 运行的真实打分样例
   （`outputs/<eval run>/predictions/<id>/rollout.json`），让讨论落在真
   实输出而不是抽象概念上。
2. **并行跑 pilot**（[dataset-sizing.md](dataset-sizing.zh.md) 的
   Stage 1，默认 40/24/30 切分）——它测出分数噪声 σ 并产出第 1 步需要
   的样例输出，即使权重后来改变也毫不浪费。
3. **把答案折回来。** 权重改动就是三个常量
   （`video2frames_env/evaluator.py` 里的 `SCENE_WEIGHT` /
   `COURIER_WEIGHT` / `JUDGE_WEIGHT`）；按字段打分或不对称快递打分是对
   `judge_text_fields` / `compute_scores` 的小型局部修改，配套单测在
   `tests/test_evaluator.py`。
4. **然后才跑完整训练阶梯**（Stage 2+）。大规模训练后再改分数意味着重
   新付一遍训练成本——分数对话是整个项目里最便宜的保险。
