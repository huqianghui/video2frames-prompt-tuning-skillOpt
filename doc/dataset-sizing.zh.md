# SkillOpt 数据集规模选择

[English](dataset-sizing.md) | **中文**

`train` / `val` / `test` 三个切分需要多大？大到你关心的分数差异能从评
估噪声中显现出来——但不必更大。本文给出本项目的具体操作流程：用已有运
行估计噪声，推导所需样本量，然后分阶段扩容而不是拍脑袋。

## 1. 为什么规模重要：门控噪声

SkillOpt 通过 skill 编辑在 **val 切分上的平均 soft 分数**来接受或拒绝
每次编辑（选择门控，`evaluation.gate_metric: soft`，作用于
`evaluation.sel_env_num` 条任务）。任何 `n` 条任务上的平均值都带有标准误

```
SE = σ / √n
```

其中 `σ` 是单任务 soft 分数的标准差。把编辑后的 skill 和当前最优比较
时，两个平均值之差只有超过约 `2.8 × SE` 才可信（95% 置信度的双样本
z 检验，`√2 × 1.96 ≈ 2.8`）。小于这个值就是抛硬币——门控会基于噪声接受
或拒绝编辑，报告的最优分数也无法复现。

要检测两个 skill 之间大小为 `δ` 的真实差异，需要

```
n ≈ 2 × (1.96 × σ / δ)²    每个 skill，在 val 切分上
```

以本项目混合分数的典型值 `σ ≈ 0.20` 计：

| val 大小 `n` | SE | 可靠检测的最小差距（≈2.8×SE） |
| --- | --- | --- |
| 24（默认） | ~0.041 | > 0.11 |
| 50 | ~0.028 | > 0.08 |
| 100 | ~0.020 | > 0.055 |
| 200 | ~0.014 | > 0.04 |

对本项目的解读：默认 val 为 24 时，一次真实提升 soft 0.05 的 skill 编
辑对门控**不可见**。提示词调优的有效收益通常在 0.03–0.10 区间，所以
val 切分 **64–100** 是现实的工作规模。

## 2. 从已有运行免费估计 σ

不要猜 `σ`——单任务分数已经在磁盘上：

- `outputs/eval_.../`（`eval.py` 写出）在该运行的 `results.jsonl` 里有
  每条任务一个 `soft` 分数，汇总在 `eval_summary.json`。
- `outputs/<训练运行>/history.json` 记录每步的门控分数；每步选择评估的
  单任务结果在 `outputs/<训练运行>/steps/` 下。

计算这些单任务 soft 分数的标准差，就是你的 `σ`。代入上面的公式和你想
检测的 `δ`，即得 val 大小。评分函数或目标模型任何变更后都要重新估
计——`σ` 是（模型、评分、数据）组合的属性，不是常数。

## 3. 不同切分需要不同大小

三个切分角色不同，扩容策略也不同：

- **`val` —— 优先扩这个。** 它驱动 SkillOpt 内部的接受/拒绝门控；它的
  噪声直接导致错误的门控决策。目标是公式给出的 `n`（本项目通常
  64–100）。让 `evaluation.sel_env_num` 等于完整 val 大小，保证每个候
  选 skill 在同一批任务上打分。
- **`test` —— 第二优先。** 最终的 baseline vs tuned 对比是**配对
  的**：用 `eval.py` 在同一切分上跑两个 skill，因此用单任务分数*差值*
  的标准差（`σ_d`，通常远小于 `σ`）代入单样本版公式
  （`n ≈ (1.96 × σ_d / δ)²`）。100 条左右通常足以支撑可信的最终结论。
  让 `evaluation.test_env_num` 匹配切分大小。
- **`train` —— 通常保持现状即可。** 每步只采样 `train.batch_size`（默
  认 8）条任务，每个反思分析器只看其中 `gradient.minibatch_size`（默
  认 4）条；40 条的池子已经足够多样。想提升反思信号，先增大
  `gradient.minibatch_size` 或 `train.num_epochs`，再考虑加训练数据。

## 4. 分阶段扩容：粗筛 + 大集重评

每步都在大 val 上做门控是最贵的部分。单次运行的 rollout 成本约为：

```
rollouts ≈ sel_env_num                          （baseline 门控评估）
         + steps × (batch_size + sel_env_num)   （每步训练 + 门控）
         + 2 × test_env_num                     （baseline + 最终 test 评估）

steps = num_epochs × ceil(train_size / (batch_size × accumulation))
```

每个 rollout 是一次多模态调用 + 一次 judge 调用。标准对策是竞速阶
梯——对多数候选便宜地粗筛，对少数入围者昂贵地精评：

1. **粗筛** —— 用中小规模 val（如 24–64）训练。小样本足以拒绝明显糟糕
   的编辑；只有势均力敌的比较才由噪声决定。
2. **重评** —— 训练结束后，取 `outputs/<run>/best_skill.md`（外加
   `outputs/<run>/skills/` 里有希望的中间 skill）和 baseline，在更大的
   持留重评切分（100–200 条，从全量池采样，与 train/val/test 不相交）
   上分别重新评估：

   ```bash
   .venv/bin/python eval.py --config configs/video2frames/default.yaml \
       --skill video2frames_env/skills/initial.md --split valid_unseen
   .venv/bin/python eval.py --config configs/video2frames/default.yaml \
       --skill outputs/<run>/best_skill.md --split valid_unseen
   ```

   以这些分数、而不是小 val 门控分数来定最终赢家。
3. **只凭证据升级** —— 如果重评的 best vs baseline 差距小于重评集的
   `2 × SE`，效果不成立。先加大搜索（更多 epoch、更大的
   `optimizer.learning_rate` 编辑预算、更好的反思 minibatch）；只有当
   一个"有希望但未确认"的差距需要更窄的置信区间时才扩数据。

这套阶梯不需要改 SkillOpt 本身——`prepare_data.py` 的大小是 CLI 参数，
`eval.py` 接受任意 skill 文件和切分。

## 5. 采样技巧

- **`val`/`test` 保持分层随机**（已是默认：`prepare_data.py` 按数据集
  家族 × 快递标签分层，固定种子）。这两个切分必须代表部署分布；绝不
  要偏置它们。
- **要偏置就偏置 `train` 向困难样本。** 反思步骤从失败中学习
  （`gradient.failure_only` 甚至可以限定只看失败），低分任务携带的信
  息最多。先用初始 skill 给候选池打分，再把低分任务过采样进 `train`。
  这通常胜过单纯增加（大多简单的）训练数据。
- **采样时始终带 `--probe-content-filter`**，让被内容过滤器拦截的视频
  （统一 0 分、纯噪声）永远不进入任何切分；probe 结果按视频缓存在
  `data/content_filter_cache.json`，之后扩容只需探测新增视频。

## 6. 分步操作手册：数据和搜索预算同步增长

数据集大小和训练超参数是同一份预算——同步、逐阶段地增长，让每个阶段的
数字决定下一步。

**Stage 0 —— 冒烟（每个环境一次）。** 验证流程，不验证科学。

```bash
.venv/bin/python prepare_data.py --train-size 2 --val-size 2 --test-size 2 --probe-content-filter
.venv/bin/python train.py --config configs/video2frames/default.yaml \
    --cfg-options train.num_epochs=1 train.batch_size=2 env.limit=4 env.workers=1 \
    evaluation.sel_env_num=4 evaluation.test_env_num=4 env.out_root=outputs/smoke_epoch
```

通过标准：运行完成，`outputs/smoke_epoch/history.json` 每步一条记录。

**Stage 1 —— pilot：测量噪声。** 默认大小，默认配置。

```bash
.venv/bin/python prepare_data.py --train-size 40 --val-size 24 --test-size 30 --seed 42 --probe-content-filter
.venv/bin/python train.py --config configs/video2frames/default.yaml
.venv/bin/python eval.py --config configs/video2frames/default.yaml \
    --skill video2frames_env/skills/initial.md --split valid_unseen   # 单任务分数 -> sigma
```

本阶段读出：baseline 评估单任务结果中的 `σ`，以及
`outputs/<run>/history.json` 里逐步的门控分数。决策规则：

- 接受与拒绝的编辑分数都落在当前最优的 `2.8 × σ/√24` 之内 → val 太小
  无法门控；进入 Stage 2。
- 门控能干净地区分好的编辑 → 可以直接在 test 上确认最优 skill 并收工。

**Stage 2 —— 先拓宽搜索，再磨尖尺子。** 一次只动一个轴，才能归因提升：

1. *同样数据，更多探索* —— 提高 `train.num_epochs`（更多轮反思与编
   辑；每步从当前最优出发，编辑是复利的）或 `optimizer.learning_rate`
   （更大的单步编辑预算；`optimizer.lr_scheduler: cosine` 会随运行衰
   减它）。优先加 epoch：门控下的顺序编辑会复利，更大的编辑预算只是
   让单步更大胆。
2. *更好的批判* —— 把 `gradient.minibatch_size` 提到 8，让每个反思分
   析器看到更多失败样例（便宜：只有训练 rollout）。
3. *更锐利的选择* —— 把 val 扩到第 1 节给出的规模并匹配配置：

   ```bash
   .venv/bin/python prepare_data.py --train-size 40 --val-size 64 --test-size 100 --seed 42 --probe-content-filter
   .venv/bin/python train.py --config configs/video2frames/default.yaml \
       --cfg-options evaluation.sel_env_num=64 evaluation.test_env_num=100
   ```

   probe 缓存让扩容很便宜：只探测新增视频。注意不同大小重新生成会重
   新分配所有切分——train/val/test 仍互不相交，但个别任务可能在切分间
   移动，之后要重跑 baseline 评估。

决策规则：只要后期步骤仍在刷新最优（`history.json` 显示后期 epoch 仍
有门控接受），就继续加大探索。如果最后一个 epoch 从未改进，再加 epoch
就是浪费钱——停止增长预算。

**Stage 3 —— 在持留集上确认。** 把入围者（第 4 节第 2 步）在 test /
大重评切分上重评：

```bash
.venv/bin/python eval.py --config configs/video2frames/default.yaml \
    --skill video2frames_env/skills/initial.md --split valid_unseen
.venv/bin/python eval.py --config configs/video2frames/default.yaml \
    --skill outputs/<run>/best_skill.md --split valid_unseen
```

只有配对差距超过 test 集的 `2 × SE` 才上线调优后的 skill。否则带着最
便宜的未试杠杆回到 Stage 2。

症状对旋钮：

| 症状（来自 history.json / eval） | 旋钮 | 方向 |
| --- | --- | --- |
| 编辑的接受/拒绝在噪声之内 | `evaluation.sel_env_num` + val 切分大小 | 扩 val |
| skill 每步几乎不变 | `optimizer.learning_rate` | 加大编辑预算 |
| 每个 epoch 最优都在提升 | `train.num_epochs` | 加 epoch |
| 后期 epoch 从不提升 | `train.num_epochs` | 停止增长 |
| 反思分析重复同样的抱怨 | `gradient.minibatch_size`、更难的训练任务 | 更丰富的失败 |
| 最终差距貌似成立但未确认 | test 切分大小 + `evaluation.test_env_num` | 扩 test |

## 7. 本项目的推荐默认值

| 切分 | Pilot（冒烟） | 工作规模 | 何时继续扩 |
| --- | --- | --- | --- |
| train | 4 | 40（保持） | 仅当反思 minibatch 看起来重复 |
| val | 2 | 64–100（由 σ 定） | 门控决策在多次运行间不稳定 |
| test | 2 | ~100 | 最终配对差距 `p ≈ 0.05`，需要更窄置信区间 |

一句话流程：从 eval 运行的单任务分数估计 `σ`，用 `n ≈ 2(1.96σ/δ)²` 按
你关心的差距 `δ` 定 val 大小，train 保持小而难，用小 val 粗筛，用大持
留集确认。
