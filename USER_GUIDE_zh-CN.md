# M2Bench 使用文档

[English](USER_GUIDE.md) | [简体中文](USER_GUIDE_zh-CN.md) | [README](README_zh-CN.md) | [论文](https://arxiv.org/abs/2605.09633v2)

本文档以当前代码为准。所有命令均假设工作目录为项目根目录，并且已激活 Conda 环境 `Patrolling`。

> **TWLO 与 `masup`。** 论文将我们提出的建模方法称为 Tail Worst-case Latency-Optimizing MDP（TWLO-MDP）。代码实现早于最终命名，因此模块、注册键和 YAML 路径仍使用 `masup`。两者表示同一个 MDP。本文正文统一使用 **TWLO**，仅在必须与代码完全一致时保留 `masup`。

## 目录

- [配置体系](#配置体系)
- [最小运行流程](#最小运行流程)
- [已实现的监测 MDP](#已实现的监测-mdp)
- [已实现的 RL 与 MARL 算法](#已实现的-rl-与-marl-算法)
- [已实现的启发式策略](#已实现的启发式策略)
- [网络与 Policy Mapping](#网络与-policy-mapping)
- [添加地图](#添加地图)
- [添加 MDP](#添加-mdp)
- [添加网络](#添加网络)
- [添加算法](#添加算法)

## 配置体系

M2Bench 使用四类 YAML：

| 配置 | 位置 | 用途 |
|---|---|---|
| 实验配置 | `configs/experiments/<mdp>/` | 环境、算法、网络、训练预算、日志和可选的训练后评估 |
| Sweep 配置 | `configs/sweep/<mdp>/` | W&B 搜索方法、优化指标、参数分布和可选的提前终止规则 |
| 评估配置 | `configs/eval/<mdp>/` | 评估环境、episode 数量、绘图、动画和动作分数日志 |
| 启发式配置 | `configs/heuristic/` | 单个启发式的参数和默认显示选项；运行环境由终端参数提供 |

实验 YAML 会被加载为 `ExperimentConfig`，主要结构如下：

```yaml
algo_name: mappo          # ALGO_REGISTRY 注册键
env_type: masup           # ENV_REGISTRY 注册键；masup 即 TWLO
actor_type: masup_mlp     # 可选的 ACTOR_REGISTRY 注册键
critic_type: masup_mlp    # 可选的 CRITIC_REGISTRY 注册键
q_type: null              # 可选的 Q_NETWORK_REGISTRY 注册键

env:                      # EnvConfig
  graph_path: graphs/grid.json
  num_agents: 3
  init_positions: [1, 26, 50]
  episode_len: 500
  custom_configs: {}

algo: {}                  # 算法参数 dataclass
training: {}              # TrainerConfig 或其子类
actor: {}                 # actor 网络参数
critic: {}                # critic 网络参数
q_network: {}             # Q 网络参数
```

基于 dataclass 的配置段会忽略未知字段，因此拼写错误可能不会报错，而是悄悄使用默认值。创建配置时应以同一 MDP、同一算法家族的现有 YAML 为模板进行核对。

常用训练字段：

| 字段 | 含义 |
|---|---|
| `training.num_envs` | 并行仿真环境数量 |
| `training.num_steps` | 每个环境在一次 iteration 中采集的 transition 数 |
| `training.total_steps` | 全局环境步预算；设置后决定实际 iteration 数 |
| `training.use_subproc` | 是否使用多进程向量环境 |
| `training.minibatch_size`, `update_epochs` | on-policy 更新设置 |
| `training.batch_size`, `warmup_steps`, `buffer_size` | off-policy replay 设置 |
| `algo.shared_policy` | 在算法支持时使用一个共享 policy，否则为每个机器人建立独立网络 |
| `env.custom_configs.truncate_by_time` | 将 `episode_len` 解释为物理仿真时间，而不是决策步数 |

Checkpoint、训练日志和评估结果默认分别写入 `models/`、`runs/` 与 `evaluators/results/`。可通过 `train.py --save-dir`、`train.py --results-dir`、`sweep.py --models-dir`、`sweep.py --results-dir` 以及 evaluator 的输出参数覆盖这些路径。

## 最小运行流程

### 运行单个实验

```bash
python train.py configs/experiments/masup/mappo_masup_grid_a3.yaml
```

尽管路径中仍为 `masup`，该示例实际是 TWLO 上的 MAPPO。训练会在带时间戳的模型目录中保存周期、best 和 final checkpoint。如果实验 YAML 设置了 `eval_config_path`，训练结束后会自动评估当前最佳 checkpoint。

常用路径覆盖方式：

```bash
python train.py configs/experiments/masup/mappo_masup_grid_a3.yaml \
  --eval-config configs/eval/masup/masup_grid_a3.yaml \
  --save-dir /data/checkpoints/grid-a3 \
  --results-dir /data/evaluation
```

仓库中的配置面向完整研究实验。需要检查流程时，应单独复制一份 smoke-test YAML，并减小 `total_steps`、`num_envs` 和 `num_steps`；不能根据启动命令较短就推断训练耗时较短。

### 运行 W&B Sweep

首次使用先登录：

```bash
wandb login
```

在当前进程中创建并执行 sweep：

```bash
python sweep.py \
  --base-config configs/experiments/masup/mappo_masup_grid_a3.yaml \
  --sweep-config configs/sweep/masup/mappo_masup_grid_a3.yaml \
  --count 20 \
  --eval-config configs/eval/masup/masup_grid_a3.yaml \
  --top-n 5
```

Base experiment 提供固定字段。与算法、trainer 或实验顶层字段同名的 sweep 参数会自动覆盖；环境特有参数使用 `custom_configs.<字段名>`。

需要在多个终端或机器上并行 trial 时，先只创建 sweep：

```bash
python sweep.py \
  --base-config configs/experiments/masup/mappo_masup_grid_a3.yaml \
  --sweep-config configs/sweep/masup/mappo_masup_grid_a3.yaml \
  --create-only
```

然后使用输出的 ID，在一个或多个终端启动 agent，并确保 W&B project 一致：

```bash
python sweep.py \
  --base-config configs/experiments/masup/mappo_masup_grid_a3.yaml \
  --sweep-config configs/sweep/masup/mappo_masup_grid_a3.yaml \
  --sweep-id <SWEEP_ID> \
  --project <WANDB_PROJECT> \
  --count 10
```

### 评估学习策略

`evaluators/test.py` 根据 checkpoint 元数据重建 policy。`--model` 必须指向包含 `config.yaml` 和 `policy.pt` 的目录；神经 actor-critic checkpoint 还可能包含 `critic.pt`。

```bash
python evaluators/test.py \
  --model models/mappo-masup-grid/<timestamp>/best \
  --env_config configs/eval/masup/masup_grid_a3.yaml \
  --num_episodes 10 \
  --episode_time 500 \
  --results-dir evaluators/results \
  --no_show
```

常用可选输出：

```bash
python evaluators/test.py \
  --model <CHECKPOINT_DIR> \
  --env_config <EVAL_YAML> \
  --animation \
  --max_frames 400 \
  --log_action_logits \
  --action_logits_csv evaluators/results/action_scores.csv \
  --no_show
```

Evaluator 使用与训练环境相同的 metric tracker 报告 IGI、AGI、IWI 和 WI。设置暂态截断时刻 `T` 后，TWLO 配置还会报告 `wi_fromT`。

### 评估启发式策略

启发式 evaluator 从终端直接接收环境超参数；启发式 YAML 仅保留策略参数和默认显示选项。

```bash
python evaluators/heuristic_evaluator.py island 6 500 \
  --policy HPCC \
  --init-positions 0 10 20 30 40 49 \
  --speeds 1 1 1 1 1 1 \
  --num_episodes 10 \
  --results-dir evaluators/results \
  --no_show
```

位置参数格式为：

```text
heuristic_evaluator.py MAP NUM_AGENTS EPISODE_LEN [options]
```

`MAP` 可以写成 `island`、`island.json` 或显式路径。`--init-positions`、`--speeds` 的长度必须等于 `NUM_AGENTS`，且每个初始节点 ID 都必须真实出现在地图的 `nodes` 数组中。省略初始位置时采用随机初始化。

其他环境选项包括 `--enable-wait`、`--deltaT`、`--truncate-by-steps`、边行驶时间抖动参数，以及可重复使用的通用覆盖参数，例如 `--env edge_time_jitter_frac=0.2`。

在相同地图、机器人数量和时域设置下运行全部启发式：

```bash
python run_all_heuristics.py island 6 500 \
  --init-positions 0 10 20 30 40 49
```

## 已实现的监测 MDP

下表按概念上的 MDP 列出方法，不把同一 MDP 的 API adapter 重复计数。引用链接指向方法来源；仓库实现会将各方法适配到统一的加权图仿真和评估协议中。

| MDP | 注册键 | 时间推进/API | 当前实现 | 参考文献 |
|---|---|---|---|---|
| **TWLO** | `masup` | Event-driven，PettingZoo parallel | 暂态后最坏时延状态与时长加权 reward；代码类名仍为 `MASUPEnv` | [Wang et al., 2026](https://arxiv.org/abs/2605.09633v2) |
| **TWLO 图观测版本** | `masup_gnn` | Event-driven，PettingZoo parallel | TWLO 动力学，加上静态图节点、机器人虚拟节点和边特征 | [Wang et al., 2026](https://arxiv.org/abs/2605.09633v2) |
| BBLA | `bbla` | Fixed-step，PettingZoo parallel | Black-Box Learner Agent 状态与到达节点时延 reward | [Santana et al., 2004](https://ieeexplore.ieee.org/document/1373634) |
| GBLA | `gbla` | Fixed-step，PettingZoo parallel | 在 BBLA 上加入其他机器人对相邻节点的目标意图 | [Santana et al., 2004](https://ieeexplore.ieee.org/document/1373634) |
| Extended-GBLA | `ex_gbla` | Fixed-step，PettingZoo parallel | 按时延排序的邻接节点与考虑协作冲突的 reward | [Lauri and Koukam, 2014](https://link.springer.com/chapter/10.1007/978-3-319-12970-9_18) |
| NEP | `nep` | Event-driven，Gymnasium joint | Node-edge-position 状态及表格 Q-learning 接口 | [Hu and Zhao, 2010](https://ieeexplore.ieee.org/document/5599681) |
| S4R1 | `s4r1` | Fixed-step，PettingZoo parallel | 起点/目标点、邻域时延等状态，用于深度 Q-learning | [Jana et al., 2022](https://doi.org/10.1007/s41315-022-00235-1) |
| BEAU | `beau` | Event-driven，PettingZoo parallel | 用于自回归 MAT 决策的图状态；由原网格/可见域设定适配而来 | [Guo et al., 2023](https://doi.org/10.1109/ICRA48891.2023.10160923) |
| MAGEC | `magec` | Event-driven，PettingZoo parallel | 与 GraphSAGE 对接的监测节点及机器人虚拟节点图特征 | [Goeckner et al., 2024](https://arxiv.org/abs/2403.13093) |
| SUNS | `suns`, `suns_gym` | Event-driven；parallel 或单智能体 Gym adapter | 全图时延/距离特征与 SUN actor/critic | [Ward et al., 2025](https://arxiv.org/abs/2412.11916) |
| OUCS | `oucs` | Fixed-step，PettingZoo parallel | 机器人位置、邻域访问次数、节点优先级和协作 reward | [Palma-Borda et al., 2026](https://doi.org/10.1016/j.engappai.2025.113706) |

`suns_gym` 仅支持单机器人。`masup_gnn` 改变 TWLO 的观测表示，而不改变监测目标。BEAU 保留了 decision-step 数据采集思想，但并非原实现的逐步完全复现；具体差异记录在 `envs/mdps/beau.py`。

## 已实现的 RL 与 MARL 算法

可执行标识以 `configs/registry.py` 为准。

| 注册键 | 类型 | Policy 组织方式 | 参考文献或实现说明 |
|---|---|---|---|
| `a2c` | On-policy actor-critic | 单策略或共享 actor | 基于 [Mnih et al., 2016](https://arxiv.org/abs/1602.01783) 的同步 A2C |
| `maa2c` | MARL actor-critic | 共享 actor、集中式 critic | 本仓库对 A2C 的 CTDE 扩展 |
| `ppo` | On-policy PPO | Actor 与 critic | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) |
| `mappo` | MARL PPO | 共享 actor、集中式 critic | [Yu et al., 2022](https://arxiv.org/abs/2103.01955) |
| `ippo` | Independent PPO | 默认独立参数，也可配置共享 | [de Witt et al., 2020](https://arxiv.org/abs/2011.09533) |
| `vdppo` | Value-decomposition PPO | PPO actor 与分解 Q 函数 | [Palma-Borda et al., 2026](https://doi.org/10.1016/j.engappai.2025.113706) 中的 VDPPO baseline；本仓库使用 QPLEX 风格 mixer |
| `d3qn` | Off-policy value learning | Q 网络 | Double DQN 与可选的 [dueling architecture](https://arxiv.org/abs/1511.06581) |
| `iql` | Independent Q-learning | 每个机器人独立 Q 网络 | [Tan, 1993](https://doi.org/10.1145/168871.168872) |
| `vdn` | Value decomposition | 共享单体 Q 网络与求和 mixer | [Sunehag et al., 2017](https://arxiv.org/abs/1706.05296) |
| `qmix` | 单调 value decomposition | 共享 Q 网络与依赖全局状态的 mixer | [Rashid et al., 2020](https://jmlr.org/papers/v21/20-081.html) |
| `qtable` | 表格 Q-learning | 每个机器人独立 Q-table | [Watkins and Dayan, 1992](https://doi.org/10.1007/BF00992698) |
| `mappo_mat` | 自回归多智能体 PPO | GAT encoder 与 MAT decoder | [Wen et al., 2022](https://arxiv.org/abs/2205.14953) 的 MAT 架构 |
| `happo` | BEAU 兼容注册键 | 与 `mappo_mat` 使用同一个 `MAPPOMATAlgo` | 被 BEAU 实验 YAML 使用，并不是另一套标准 HAPPO 实现 |

On-policy 流程支持 action mask、有效决策 mask、循环网络 chunk、value normalization，以及处理 event-driven 非活动步的 transparent GAE。Off-policy 流程支持普通或序列 replay、target network、循环网络 burn-in，以及配置指定的决策时刻同步。

## 已实现的启发式策略

下列名称可以直接传给 `--policy`。链接给出概念来源。部分控制器是对统一仿真接口的适配或轻量仓库变体，在论文中将其描述为原算法的严格复现前，应先查看对应模块的 docstring。

| CLI 名称 | 策略 | 参考文献或说明 |
|---|---|---|
| `RANDOM` | 均匀随机选择相邻节点 | 无需引用的随机基线 |
| `CR` | Conscientious Reactive | [Portugal and Rocha, 2013](https://doi.org/10.1080/01691864.2013.763722) |
| `HCR` | Heuristic Conscientious Reactive | [Portugal and Rocha, 2013](https://doi.org/10.1080/01691864.2013.763722) Algorithm 2 |
| `CC` | Conscientious Cognitive | 本仓库局部时延-距离变体；相关对比见 [Portugal and Rocha, 2013](https://doi.org/10.1080/01691864.2013.763722) |
| `HPCC` | Heuristic Pathfinder Conscientious Cognitive | [Portugal and Rocha, 2013](https://doi.org/10.1080/01691864.2013.763722) Algorithm 3 |
| `ER` | Expected Idleness / Expected Reactive | [Yan and Zhang, 2016](https://doi.org/10.1177/1729881416663666) |
| `GBS` | Greedy Bayesian Strategy | [Portugal and Rocha, 2013](https://doi.org/10.1016/j.robot.2013.06.011) |
| `SEBS` | State Exchange Bayesian Strategy | [Portugal and Rocha, 2013](https://doi.org/10.1016/j.robot.2013.06.011) |
| `BAPS` | Bayesian Ant Patrolling Strategy | [Chen et al., 2015](https://doi.org/10.5194/isprsannals-II-4-W2-103-2015) |
| `CBLS` | 带独立 tabu list 的 cycle-based 策略 | 轻量仓库变体；Bayesian 巡逻背景可参考 [Portugal 博士论文](https://ap.isr.uc.pt/archive/DPortugal_PhDThesis_2014.pdf) |
| `MSP` | 按配置循环路线执行 | 相关分区方法见 [Portugal and Rocha, 2010](https://ap.isr.uc.pt/archive/PR10_ACM_SAC2010.pdf)；当前实现直接读取预计算路线 |
| `DTAGREEDY` | 分布式任务分配 greedy 变体 | [Farinelli et al., 2017](https://doi.org/10.1007/s10514-016-9579-8) |
| `DTASSI` | 受 sequential single-item 启发的 DTA 变体 | 相关拍卖方法见 [Farinelli et al., 2017](https://doi.org/10.1007/s10514-016-9579-8)；当前代码使用带噪局部分数，并未执行完整拍卖协议 |
| `AHPA` | Adaptive Heuristic Patrolling Agent | [Goeckner et al., 2024](https://arxiv.org/abs/2304.01386) |

## 网络与 Policy Mapping

Actor、critic 与 Q 网络可以独立选择。因此，同一个 MDP 和优化算法可以与 MLP、循环网络、图网络或特定建模方法的 encoder 组合，而不必重写 trainer。

| 注册表 | 可用键 |
|---|---|
| Actor | `mlp`, `rnn`, `sun`, `mpnn`, `sage`, `masup_mlp`, `masup_rnn` |
| Critic | `mlp`, `rnn`, `sun`, `masup_mlp`, `masup_rnn` |
| Q 网络 | `mlp`, `rnn`, `masup_q_mlp`, `masup_q_rnn`, `masup_vdppo_mlp`, `masup_vdppo_rnn` |

历史命名 `masup_*` 表示 TWLO 专用观测 encoder。MAT 使用独立的 `GATEncoder` 与 `MATDecoder` 构建路径。

`ActorPolicy` 与 `ValuePolicy` 定义可复用的“观测到动作”行为；`MultiAgentPolicy` 再将机器人 ID 映射到不同 policy 实例，或映射到同一个共享实例。共享模式会将各机器人 batch 堆叠后一次调用网络，再还原成按 agent ID 索引的输出。由此，算法可以改变参数优化方式，而无需重复实现 action masking、循环状态和推理逻辑。

## 添加地图

### 1. 创建图 JSON

地图表示 `G=(V,E,W,phi)`：

```json
{
  "nodes": [0, 1, 2],
  "edges": [
    {"from": 0, "to": 1, "weight": 1.0},
    {"from": 1, "to": 0, "weight": 1.0},
    {"from": 1, "to": 2, "weight": 2.0},
    {"from": 2, "to": 1, "weight": 2.0}
  ],
  "phi": {"0": 1.0, "1": 2.0, "2": 1.0}
}
```

- `nodes` 必须是整数节点 ID。强烈建议使用连续 ID，因为部分基线状态编码有此假设。
- `weight` 是正的行驶距离，仿真器根据距离和机器人速度得到行驶时间。
- `phi` 必须为每个节点提供正的监测优先级权重。
- Parser 按有向邻接项读取边。无向监测图应像现有地图一样，为每条边写入权重相同的两个方向。
- 使用最短路的 policy 与 MDP 要求图连通。

### 2. 生成可视化坐标

```bash
python utils/graph_layout.py graphs/my_map.json
```

命令会生成归一化坐标文件 `graphs/my_map_coords.json`。可视化以及 BEAU/MAT 图模块会使用该文件；后两者可以自动生成缺失坐标，但提交固定坐标文件更有利于图像复现。

### 3. 添加配置

从同一 MDP 家族复制实验与评估 YAML，至少修改：

- `graph_name` 和环境、网络段中的所有 `graph_path`。
- `num_agents`、`init_positions` 和 `episode_len`。
- TWLO 网络中的 `num_nodes`、`num_agents`、角色信息和截断时刻 `T`。
- 与图规模相关的 batch 或模型参数。

必须根据实际 `nodes` 数组验证初始位置。地图有 50 个节点，并不代表 ID `0` 到 `49` 一定全部存在。

## 添加 MDP

### 1. 选择环境 API

- 同质 PettingZoo parallel 环境继承 `FixedStepEnv` 或 `EventDrivenEnv`。
- 用一个 Gymnasium action 控制 joint 过程时，继承 `JointFixedStepEnv` 或 `JointEventDrivenEnv`。
- 复用 `PatrolWorld` 处理图运动、到达、等待、时延、节点权重和指标记录。

### 2. 实现接口

Parallel 环境需要提供 `observation_space`、`action_space`、`state`、`_build_obs`、`_build_info`、`_compute_rewards`、`_compute_truncations` 和 `_action_to_target`。Joint 环境需要提供 Gymnasium spaces，以及对应的单 action dispatch、observation、info、reward 和 truncation 方法。

异步环境应暴露 `active_mask`，并通过 `action_mask` 让未到达决策时刻的机器人只能执行 no-op，防止算法优化机器人实际没有作出的决策。

### 3. 注册并配置

在 `configs/registry.py` 的 `ENV_REGISTRY` 中添加模块与类，然后创建相应的 experiment/eval YAML。所有 MDP 共享的构造字段放入 `EnvConfig`；特定建模方法的值放入 `env.custom_configs`，并以关键字参数传入环境。

### 4. 验证行为

检查 reset/step 与 space 是否一致、READY 与移动中机器人的 action mask、在 `episode_len` 处的物理时间截断、episode 指标收尾、随机种子确定性，以及学习策略能否通过 evaluator 恢复并运行。

## 添加网络

### 1. 定义网络配置

在 `configs/network_configs.py` 中为结构参数添加 dataclass。即使 actor、critic、Q 网络共享基础字段，也应保留独立的具体类型。

### 2. 实现重建协议

可保存的网络应提供稳定的 `input_dim`、`output_dim` 元数据，并实现：

```python
def get_config_dict(self, input_dim: int, output_dim: int) -> dict: ...

@classmethod
def from_config_dict(cls, cfg: dict): ...
```

保存的字典必须包含唯一的 `type`，以及加载 state dict 前重建模块所需的全部参数。循环网络还应实现 policy wrapper 使用的隐藏状态与 sequence-forward 接口。

### 3. 注册两条构建路径

在 `configs/registry.py` 的 `ACTOR_REGISTRY`、`CRITIC_REGISTRY` 或 `Q_NETWORK_REGISTRY` 添加 YAML 键。若现有 factory 分支无法构造该网络，再有针对性地扩展 `create_actor`、`create_critic` 或 `create_q_network`。

同时必须将类名加入 `utils/model_io._get_class_registry`，否则训练可以保存权重，评估却无法重建网络。

### 4. 测试保存与加载

完成一次实例化或短训练，保存 checkpoint，通过 `load_policy_for_eval` 重新加载，并比较相同 observation 和 action mask 下的确定性输出。

## 添加算法

### 1. 添加参数与实现

在 `configs/algo_configs.py` 中创建算法参数 dataclass。在 `algorithms/rl/`、`algorithms/marl/` 或 `algorithms/tabular/` 下实现算法，通常继承 `BaseAlgorithm`、`ActorCriticOnPolicyAlgo`、`PPOBase` 或 `QLearningOffPolicyAlgo`。

面向 trainer 的实现需要准备 collector 产生的 batch、完成参数更新、返回训练统计，并在需要时维护 target network 或循环状态。

### 2. 注册算法

在 `ALGO_REGISTRY` 中添加：

```python
"my_algo": {
    "module": "algorithms.marl.my_algo",
    "class_name": "MyAlgo",
    "params_class": MyAlgoParams,
    "trainer_type": "on_policy",  # 或 off_policy / 单独的 tabular 路径
    "policy_type": "actor",       # actor、value 或 mat
}
```

`create_algorithm` 会检查构造函数，并从 `train.py` 组装的 context 中传入名称匹配的值。只有算法需要的信息尚不在 context 中时，才需要在那里增加新字段。

### 3. 对接 policy 与 collector 语义

明确算法需要 actor 还是 value policy、独立还是共享参数、是否使用集中式 state、同步 replay index、循环序列和 active mask。若需求与现有算法家族不同，应显式扩展 `_build_policy` 或 `_build_collector`，而不是添加一个没有组件读取的 YAML 字段。

### 4. 添加配置并验证

至少提供一组 experiment/eval YAML，先运行短 collection/update 检查，再验证 checkpoint 重建和完整 evaluator。MARL 算法需要额外检查 action mask 与 agent 顺序；event-driven 算法还要检查非活动 transition 的处理。
