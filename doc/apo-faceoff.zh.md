# APO vs SkillOpt 对决：方法与结果

[English](apo-faceoff.md) | **中文**

本文档记录旧 agent-lightning/APO 项目调出的 prompt 与本 SkillOpt 移植项目
调出的 skill 之间的正面对比，包括共享 held-out 测试集的构建与扩容方式、
完整数据表格和结论。运行日期：2026-07-15。

## 1. 对比对象

| 参赛者 | 文件 | 来源 |
| --- | --- | --- |
| baseline | `video2frames_env/skills/initial.md` | 未调优的初始指令 prompt（与旧项目 `baseline_prompt.txt` 逐字节一致——已用 `diff` 验证） |
| APO best | `data/best_prompt.txt` | agent-lightning/APO 训练跑出的最优 prompt |
| SkillOpt best | `outputs/skillopt_video2frames_gpt-5.4_20260715_141809/best_skill.md` | 本项目训练跑出的最优 skill（20 步，80 train / 100 val） |

可比性保障：

- **相同打分**：soft/hard 公式是 APO reward 的无损移植
  （[reward-design.zh.md](reward-design.zh.md)）。
- **相同目标模型**（`gpt-4.1-mini`）和**相同 judge**（`env.judge_model`），
  所有运行固定不变。
- **无污染的测试集**：每个测试任务对*两个*优化器都是未见过的——既排除
  了本项目的 train/val，也排除了旧 APO 项目的 train/val。

## 2. 扩容测试集（30 → 100）

第一轮在原始 30 任务测试集上运行；观察到的差距（APO soft +0.027 ± 0.018
SE）在噪声范围内，因此把测试集扩到 100 个任务：

```bash
python prepare_data.py --grow-test 100 --probe-content-filter --probe-workers 8 \
    --exclude <old-apo-project>/data/train.jsonl <old-apo-project>/data/val.jsonl
```

`--grow-test` 向 `data/test.jsonl` 追加新采样的任务，直到达到目标数量。
train/val 和已有的测试行不受影响（前缀逐字节一致）；新候选从源数据池中
减去所有切分及 `--exclude` 文件使用过的视频后做分层采样，并逐个探测内容
过滤。所有排除之后的候选池为 5,847 个源视频中的 5,582 个。

注意：整个数据集的 courier 正例极其稀疏（5,847 条源记录里只有 13 个正
例，0.2%），因此所有切分的 courier 正例都是 0；`courier_match` 这个
reward 分量实际度量的是"模型是否避免误报"。

## 3. 100 任务完整测试集上的结果

`evaluation.test_env_num` 会限制 eval 规模（配置里当时是 30）；完整运行
使用了 `--cfg-options evaluation.test_env_num=100`。

| Skill | hard | soft | scene_match | courier_match | judge_score |
| --- | --- | --- | --- | --- | --- |
| baseline | **0.59** | 0.7879 | 0.94 | 0.98 | 0.6732 |
| SkillOpt best | 0.55 | 0.7846 | **0.95** | 0.98 | 0.6643 |
| APO best | 0.57 | **0.8056** | 0.94 | **1.00** | **0.6960** |

逐任务配对分析（相同任务、逐任务求差，n=100）：

| 对比 | soft 差值 ± SE | t | 胜/负/平 |
| --- | --- | --- | --- |
| APO − baseline | +0.0177 ± 0.0129 | 1.37（不显著） | 43/36/21 |
| SkillOpt − baseline | −0.0033 ± 0.0124 | −0.27（持平） | 37/42/21 |
| APO − SkillOpt | +0.0210 ± 0.0114 | 1.84（p ≈ 0.07） | 44/36/20 |

作为参考，第一轮 30 任务集的结果：baseline 0.7749、SkillOpt 0.7830、
APO 0.8104（soft）。第二轮重跑同样 30 个任务时，每个 skill 移动了
±0.01–0.02——单次 eval 的方差就有这么大，这正是 30 任务那一轮无法区分
参赛者的原因。

## 4. 结论

1. **SkillOpt 的增益没有泛化。**在 70 个全新任务上，其最优 skill 与
   baseline 打平（soft 0.7957 vs 0.7921）；旧 30 任务集上看到的 +0.008
   是 gate 对 val 集的过拟合加噪声。
2. **APO 领先 SkillOpt** +0.021 soft（t = 1.84，边缘显著），而它对
   baseline 的优势在更大的测试集上缩小到不显著的 +0.018。
3. **全部差距都在 `judge_score`**（对照 ground truth 的文本质量）：
   scene/courier 精确匹配分量接近饱和，各参赛者几乎一致。
4. **`hard` 以 baseline 最高**（0.59）：两个调优后的 prompt 都在略微抬升
   或保持平均质量的同时，把少数边缘任务压到 0.8 阈值以下——这也是
   `hard` 不用于 gating 的又一个理由。
5. 实践意义：以 `gpt-4.1-mini` 为目标模型时，此任务相对 baseline prompt
   的提升空间很小（soft ≤ +0.02）；要可靠地测出它需要 ≥ 100 个配对测试
   任务，而要真正拿到它需要的是更好的优化运行，不是更好的测量。

## 5. 复现

```bash
# 扩容测试集（达到 100 后幂等）
python prepare_data.py --grow-test 100 --probe-content-filter \
    --exclude <old-apo>/data/train.jsonl <old-apo>/data/val.jsonl

# 评测三个参赛者
python eval.py --config configs/video2frames/default.yaml \
    --skill video2frames_env/skills/initial.md --split valid_unseen \
    --cfg-options evaluation.test_env_num=100 env.out_root=outputs/faceoff100_baseline_n100
python eval.py --config configs/video2frames/default.yaml \
    --skill outputs/<run>/best_skill.md --split valid_unseen \
    --cfg-options evaluation.test_env_num=100 env.out_root=outputs/faceoff100_skillopt_n100
python eval.py --config configs/video2frames/default.yaml \
    --skill data/best_prompt.txt --split valid_unseen \
    --cfg-options evaluation.test_env_num=100 env.out_root=outputs/faceoff100_apo_n100
```

每个输出目录包含 `eval_summary.json`、逐任务的 `results.jsonl` 和原始预
测；配对分析就是把三份 `results.jsonl` 按 task id 直接逐任务连接。
