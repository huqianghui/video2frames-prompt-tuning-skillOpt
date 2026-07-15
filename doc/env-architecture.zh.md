# Env 架构：为什么拆成这些组件，客户到底要实现什么

[English](env-architecture.md) | **中文**

旧的 APO 项目在 agent-lightning 上只需要一个 rollout 函数，外加通过
tracer 标签上报 reward。本项目却有一整个包——`video2frames_env/` 下的
`tasks.py`、`dataloader.py`、`rollout.py`、`evaluator.py`、`adapter.py`——
以及只做注册的薄入口 `train.py`/`eval.py`。本文解释形态为什么变了、什么
没变，以及移植到新任务时客户真正要写哪些部分。

## 1. 两个框架要的东西本质相同

任何 prompt/skill 优化器都需要客户提供且只需要提供三样逻辑：**数据**
（任务长什么样）、**执行**（skill + 任务怎么变成模型输出）、**打分**
（输出好不好）。框架的差别只在插槽的形状：

| 客户逻辑 | agent-lightning（旧 APO） | SkillOpt（本仓库） |
| --- | --- | --- |
| 数据：任务 schema 与供给 | 自己读文件、把 task dict 发给 server | `tasks.py`（schema）+ `dataloader.py`（split/分批） |
| 执行：跑一个任务 | 一个 rollout 函数（worker 轮询领任务） | `rollout.py`（帧 URL + 目标模型调用） |
| 打分：给输出评分 | 自己算 reward，通过 tracer 标签上报 | `evaluator.py`（scene/courier 硬指标 + judge 软分） |
| 接线：框架如何拿到 input/output/reward | tracer 从标签里隐式收集三元组 | `adapter.py` 显式实现 `EnvAdapter` 接口 |
| 入口 | 框架 CLI | `train.py` / `eval.py`（约 40 行：注册 + 委托） |

## 2. 为什么要有 "env"，train.py 为什么是"注册再委托"

"env" 是从 RL 借来的词：**trainer 是通用优化循环，env 是任务专属的世
界**。SkillOpt 的 trainer 把 rollout → reflect → merge → gate eval、断点
续跑、token 统计、gate 判定实现了一次；所有属于*这个具体任务*的东西
（数据、执行、打分）都在 `EnvAdapter` 接口后面。这和 gym 的
`env.step()` 是同一个思想：算法不需要知道自己在玩 Atari 还是在优化快递
视频的 prompt。换一个客户任务 = 换 env，trainer 改动为零行。

`train.py` 这层薄包装的存在是因为 skillopt 是**装在 site-packages 里的
第三方包**：它的 CLI（`scripts.train`）自带内置 env 注册表，但不认识我
们的任务。入口文件就是组合根——唯一同时认识两侧的地方：

```python
skillopt_train._ENV_REGISTRY["video2frames"] = Video2FramesAdapter  # 注册
skillopt_train.main()                                               # 委托
```

这一行注册换来的是：上游 trainer **零修改、零 fork**——续跑、gate、
token 记账全部白拿，升级 skillopt 不碰我们的代码。这与 pytest 插件、
Django app 是同一个模式：通用引擎 + 注册进去的业务模块。

## 3. 小而隐式 vs 大而显式的契约

**agent-lightning：契约小、隐式。**写一个函数、打上 reward 标签，tracer
在幕后收集 (input, output, reward) 三元组。上手代码最少，但数据流不可
见——reward 悄悄没被收集到时几乎无从排查，而且 server-client 架构带来
了 [concurrency.md](concurrency.zh.md) 里记录的跨平台并发问题。

**SkillOpt：契约大、显式。**一个文件变五个，但每个职责单一、以纯函数为
主，全部可以离线单测——本仓库的测试套件（76 个测试、零网络调用）就是
这个结构的直接收益。SkillOpt 还多一条 agent-lightning 没有的要求：**反
思轨迹契约**——每个任务必须落一份 `conversation.json`，让优化器在反思
阶段读到完整的 input/output/feedback（见
[reflection-trajectories.md](reflection-trajectories.zh.md)）。这是用"文
本梯度"取代评分黑盒的代价。

另外，"prompt 变成了 skill"只是改名：skill
（`video2frames_env/skills/initial.md`，与旧 `baseline_prompt.txt` 逐字
节相同）仍然是发给目标模型的 system prompt。驱动优化器本身的是另一层元
prompt（`skillopt_prompts/`，可用 `custom_prompt/` 覆盖）。

## 4. 客户真正要写的部分（按设计含金量排序）

1. **Reward 设计（`evaluator.py`）——唯一真正需要"设计"的部分。**优化
   器只会朝 reward 度量的方向爬；reward 定义错了就在优化错的东西。这也
   是 [reward-design.md](reward-design.zh.md) 以客户确认问题的形式来写
   的原因。Reward 在框架间可以逐字移植——本仓库的 soft 公式就是从 APO
   项目原样拷来的。
2. **目标调用（`rollout.py`，核心约 50 行）**：skill + 任务数据如何变成
   一次模型调用。业务相关但机械。
3. **数据接入（`tasks.py` + `dataloader.py`）**：把客户数据转成任务列
   表。纯机械。
4. **胶水（`adapter.py`、`train.py`）**：近乎模板，换任务时拷贝改名即可。

## 5. 结论

两个框架的工作量大体相同，因为第 1–3 项就是业务本身——没有框架能替你
写。差别在第 4 层的形状：agent-lightning 把它藏进 tracer 约定，SkillOpt
把它摊开成文件。对要交付给客户长期维护的项目，显式契约更划算：可测试、
可调试、跨平台行为一致。
