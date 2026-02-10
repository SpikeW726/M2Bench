# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个**多智能体巡逻(MAP)模仿学习框架**，用于研究图环境中的多智能体协调算法。框架支持启发式策略和强化学习算法，采用事件驱动的同步决策机制。

## 核心架构

### 三层分离设计

```
┌─────────────────────────────────────────────────────────────┐
│                      Policy 层                               │
│  /policies/heuritic/      启发式策略 (ER, HPCC)              │
│  /policies/rl/           强化学习策略 (ActorPolicy)          │
│  /policies/marl/         多智能体策略 (MultiAgentPolicy)     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Environment 层                          │
│  EventDrivenEnv          事件驱动同步决策基类                │
│  MASUPEnv                主要环境实现                        │
│  PatrolWorld             物理世界模拟器                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core 层                                 │
│  Graph                    图结构（节点、边、权重、优先级）    │
│  EpisodeMetricsTracker   指标追踪（IGI, AGI, IWI, WI）       │
└─────────────────────────────────────────────────────────────┘
```

### 关键设计模式

**1. 事件驱动决策 (Event-Driven)**

环境不是固定时间步推进，而是推进到"下一个事件"（智能体到达目标或等待完成）。只有状态为 `READY` 的智能体才需要决策。

```python
# envs/mdps/base_envs.py:44-71
# 智能体在边上的不会接收动作，只有 READY 状态的会决策
def step(self, actions: Dict[str, int]):
    # 1. 只为"可以决策的智能体"设置动作
    for agent_str, action_idx in actions.items():
        if self.world.is_ready(agent_id):
            # 设置动作
    # 2. 推进到最近的事件
    result = self.world.tick_to_next_event()
```

**2. 顺序决策协议 (Sequential Decision)**

启发式策略采用顺序决策：Agent 0 → Agent 1 → Agent 2 → ...，每个智能体的意图立即记录，后续智能体可以看到并避免冲突。

```python
# policies/heuritic/er.py:80-105
# 每个 agent 决策后立即更新 er_intention_eta
for agent_str, obs in obs_dict.items():
    result = self._compute_action(agent_id, obs, global_state)
    if result is not None:
        action_idx, target_node, eta = result
        actions[agent_str] = action_idx
        # 立即更新意图，后续 agent 可以看到
        er_intention_eta[target_node] = eta
```

**3. 模块化配置 (YAML Config)**

环境、策略、训练参数全部通过 YAML 配置：

```python
# configs/MASUPEnv.yaml
env_config:
  graph_path: "graphs/simple_TSP_12.json"
  num_agents: 3
  episode_len: 300

custom_config:
  T: 20.0                    # T_time 截断
  role_imformation: "agent-index"  # 观测空间类型
  truncate_by_time: True     # 按时间还是步数截断
```

## 核心类说明

### PatrolWorld (`envs/mdps/patrol_core.py`)

物理世界模拟器，职责：
- 管理图结构和智能体位置
- 处理智能体移动、等待、节点空闲度更新
- 提供时间推进：`tick(dt)` 和 `tick_to_next_event()`

**关键方法**：
- `tick(dt)`: 推进指定时间量，返回 `TickResult`
- `set_move_action(agent_id, target_node)`: 设置移动动作
- `set_wait_action(agent_id)`: 设置等待动作
- `is_ready(agent_id)`: 智能体是否可以决策

### EventDrivenEnv (`envs/mdps/base_envs.py`)

抽象基类，继承自 `PettingZoo.ParallelEnv`。子类必须实现：
- `observation_space(agent)`: 观测空间
- `action_space(agent)`: 动作空间
- `_build_obs(result)`: 构建观测
- `_build_info(result)`: 构建 info（必须包含 `active_mask`）
- `_compute_rewards(result)`: 计算奖励
- `_action_to_target(agent_id, action_idx)`: 动作索引 → 目标节点

### MASUPEnv (`envs/mdps/masup_env.py`)

主要环境实现，特点：
- **T_time 截断**：`obs_timer < T_time` 时 `worst_idleness` 保持为 0
- **两种截断模式**：按时间 (`truncate_by_time=True`) 或按步数
- **观测空间类型**：`agent-index`（one-hot 编码）、`position`（位置信息）或 `decision`（决策导向观测）
- **奖励缩放**：支持 `reward_scale`、`idi_scale`、`contribution_scale` 参数调节奖励权重

### HeuristicBasePolicy (`policies/heuritic/heuristic_base.py`)

启发式策略基类，子类必须实现：
- `compute_actions(obs_dict, global_state)`: 为所有智能体计算动作
- `_compute_action(agent_idx, obs, global_state)`: 单个智能体决策

**观测格式** (`obs_dict`):
```python
{
    'current_node': int,
    'neighbors': List[int],
    'on_edge': bool,
    ...
}
```

**全局状态格式** (`global_state`):
```python
{
    'graph': Graph,
    'agent_positions': Dict[int, int],
    'agents_target_node': Dict[int, int],
    'node_idleness': Dict[int, float],
    'agents_on_edge': Dict[int, bool],
    'current_time': float,
    'node_last_visit': Dict[int, float],
    'er_intention_eta': Dict[int, float],  # ER 算法的意图表
}
```

### ERPolicy (`policies/heuritic/er.py`)

Expected Idleness 启发式策略：
- 对每个邻居 v，计算到达时的预期空闲度 `I_exp`
- 如果其他智能体有意图 (v, eta)，则 `I_exp = t_next - eta`
- 否则 `I_exp = t_next - last_visit_time[v]`
- 选择效用最大的邻居：`U = |I_exp| / travel_time`

### HPCCPolicy (`policies/heuritic/hpcc.py`)

Hierarchical Priority-based Coordination Control 启发式策略：
- 基于优先级的协调控制算法
- 加权效用函数：`U = w_idl * U_idl + w_dist * U_dist + w_conflict * U_conflict`
  - `U_idl`: 空闲度效用
  - `U_dist`: 距离效用（支持最短路径 `sp` 或欧几里得距离 `euclidean`）
  - `U_conflict`: 冲突效用（通过 ETA 机制避免冲突）
- 支持认知协调参数：
  - `conflict_eta_margin`: 允许的 ETA 余量
  - `eta_clip_min`: ETA 最小值裁剪

## 运行命令

```bash
# 启发式策略采样 - 收集专家演示数据（支持并行采样）
python trainers/imitator/heuristic_sampler.py

# MAPPO 训练 - 使用 OnPolicyTrainer（推荐）
python train_mappo.py

# MAPPO 训练 - 使用 main.py（手动训练循环）
python main.py

# 测试训练好的策略
python evaluators/test.py --model models/mappo/imi/final --num_episodes 10

# 测试并保存可视化结果
python evaluators/test.py --model models/mappo/imi/final --num_episodes 10 --save_plot evaluators/results/rl_eval.png --no_show

# 生成 MP4 动画（启发式策略）
python evaluators/heuristic_evaluator.py --policy ER --save_animation evaluators/results/ER_animation.mp4
```

## 完整训练流程

框架支持从启发式策略到强化学习的完整训练流程：

```
1. 启发式采样 → 2. 模仿学习预训练 → 3. RL 微调
```

### 1. 启发式采样 (`trainers/imitator/heuristic_sampler.py`)

使用启发式策略（如 ER）收集专家演示数据，保存为 NPZ/HDF5 格式。

**支持并行采样**：
- 使用 `num_workers` 参数启用多进程并行采集
- 内存优化的分块收集机制，支持大规模数据采集

```python
# 数据格式:
- obs: [N, T, M, Obs_Dim] - Actor 输入
- critic_states: [N, T, M, State_Dim+M] - Centralized Critic 输入
- actions: [N, T, M, 1] - 动作索引
- action_masks: [N, T, M, Act_Dim] - 动作掩码
- rewards: [N, T, M, 1] - 每个智能体的奖励
- padded_mask: [N, T, 1] - 填充掩码 (1=真实, 0=填充)
- returns: [N, T, M, 1] - 累计折扣回报
```

### 2. 模仿学习预训练 (`trainers/imitator/imitation_trainer.py`)

在收集的数据上训练 Actor-Critic 网络（监督学习）。

### 3. RL 微调 (`train_mappo.py` 或 `main.py`)

使用 MAPPO 算法微调预训练网络。MAPPO 特点：
- **双优化器**：Actor 和 Critic 使用独立的优化器和学习率
- **参数共享**：多个智能体共享 Actor 网络
- **Centralized Critic**：Critic 输入包含全局状态 + agent one-hot

## 代码使用示例

```python
from policies.heuritic.er import ERPolicy
from envs.mdps.masup_env import MASUPEnv
import yaml

with open("configs/ER.yaml") as f:
    ER_config = yaml.safe_load(f)
with open("configs/MASUPEnv.yaml") as f:
    env_config = yaml.safe_load(f)
    custom_config = env_config["custom_config"]
    env_config = env_config["env_config"]

num_agents = env_config["num_agents"]
policy = ERPolicy(num_agents, ER_config)
env = MASUPEnv(env_config, **custom_config)

obs, infos = env.reset()
for _ in range(100):
    # 构建 obs_dict 和 global_state
    obs_dict = {...}
    global_state = {...}
    actions = policy.compute_actions(obs_dict, global_state)
    obs, rewards, terms, truncs, infos = env.step(actions)
```

## 图文件格式

图文件位于 `graphs/` 目录，格式为 JSON：

```json
{
  "nodes": [0, 1, 2, ...],
  "edges": [
    {"from": 0, "to": 1, "weight": 4},
    {"from": 1, "to": 2, "weight": 2}
  ],
  "phi": {"0": 1, "1": 2, "2": 1}
}
```

- `nodes`: 节点 ID 列表
- `edges`: 边列表（有向），包含权重（旅行时间）
- `phi`: 节点优先级（用于计算加权空闲度）

## 常见任务

### 添加新的启发式策略

1. 继承 `HeuristicBasePolicy`
2. 实现 `compute_actions()` 和 `_compute_action()`
3. 在 `configs/` 下添加配置文件
4. 更新 `HeuristicSampler` 以支持新策略

### 添加新的图环境

1. 在 `graphs/` 下添加 JSON 文件
2. 更新 `configs/MASUPEnv.yaml` 中的 `graph_path`

### 修改观测空间

编辑 `envs/mdps/masup_env.py` 中的 `_build_obs()` 和 `observation_space()`，注意保持上下界一致。

## 重要约束

1. **动作索引约定**：`action_idx=0` 始终代表"等待"动作（如果 `enable_wait=True`）
2. **info 必须包含 active_mask**：用于标记智能体是否真正需要决策
3. **T_time 语义**：T_time 之前 `worst_idleness_fromT` 保持为 0，用于延迟奖励计算

## 核心模块扩展

### 网络架构 (`networks/`)

- **ActorMLP** (`networks/mlp.py`): Actor 网络，输出层使用 `std=0.01` 初始化保证稳定性
- **CriticMLP** (`networks/mlp.py`): Critic 网络（值函数），输出层使用 `std=1.0`

所有线性层使用正交初始化 (`layer_init`)。

### 训练框架 (`trainers/`)

- **OnPolicyTrainer** (`trainers/rl_trainer.py`): 通用的 On-Policy RL 训练器，支持回调机制
  - `save_checkpoint_fn(iteration)`: 保存检查点
  - `log_extra_fn()`: 记录额外指标（如 idleness）
  - `stop_fn(mean_reward)`: 提前停止条件

### 模型 I/O (`utils/model_io.py`)

采用 HuggingFace 风格的模型保存格式：

```
model_dir/
├── config.yaml    # 网络架构配置
├── policy.pt      # Policy 权重
└── critic.pt      # Critic 权重（可选）
```

```python
# 保存
save_model(save_dir, policy=ma_policy, critic=critic_net,
           actor_config={...}, critic_config={...})

# 加载
actor, critic = load_model(model_dir, device='cuda')
```

### 多智能体策略 (`policies/marl/`)

- **MultiAgentPolicy**: 包装单个 Policy 实现多智能体接口
  - `shared=True`: 所有智能体共享参数
  - 支持 `action_mask` 参数用于无效动作屏蔽

### 数据采集 (`data/`)

- **MACollector** (`data/collector.py`): 多智能体数据采集器
- **CollectResult** (`data/batch.py`): 采集结果数据类，包含 transitions、episode_rewards 等

### 可视化工具 (`utils/vis_utils.py`)

- **create_nx_layout()**: 使用 NetworkX 的 kamada_kawai_layout 算法创建高质量节点布局
- **plot_training_curve()**: 绘制训练曲线（平均空闲度）
- **plot_training_curves()**: 绘制双曲线（平均空闲度和最大延迟）
- **MP4 动画生成**: 支持生成智能体巡逻过程的 MP4 动画

### 图工具 (`utils/graph_utils.py`)

- **Graph**: 图数据结构，包含节点、边、权重和优先级
- 支持最短路径计算和邻居查询

### 评估工具 (`utils/eval_utils.py`)

- **aggregate_episode_metrics()**: 聚合 episode 指标
- **plot_aggregated_metrics()**: 绘制聚合指标图表

### 日志工具 (`utils/log_utils.py`)

- 训练日志记录和 TensorBoard 集成
- 指标追踪和可视化

## 新增功能（近期更新）

### MP4 可视化动画

框架支持生成智能体巡逻过程的 MP4 动画，直观展示策略性能：

```python
from utils.vis_utils import create_animation

# 生成动画
create_animation(
    env=env,
    policy=policy,
    save_path="evaluators/results/patrol_animation.mp4",
    num_episodes=1
)
```

动画特性：
- 使用 NetworkX kamada_kawai_layout 高质量布局
- 节点大小根据优先级动态调整
- 边的视觉长度与权重成正比
- 智能体位置实时更新
- 节点空闲度颜色编码

### 并行采样优化

HeuristicSampler 支持多进程并行数据采集：

```python
# configs/HeuristicSampler.yaml
num_workers: 4  # 使用 4 个进程并行采集
```

并行特性：
- 每个 worker 进程独立环境和策略实例
- 内存优化的分块收集机制
- 支持 HDF5 格式大规模数据存储
- 自动处理进程间数据汇总

### 值归一化

训练过程中支持 Critic 值归一化，提高训练稳定性：
- 模仿学习预训练时启用值归一化
- RL 微调时保持归一化统计量
- 自动更新 running mean/std

