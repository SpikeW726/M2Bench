# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个**多智能体巡逻(MAP)模仿学习框架**，用于研究图环境中的多智能体协调算法。框架支持启发式策略和强化学习算法，采用事件驱动的同步决策机制。

## 核心架构

### 三层分离设计

```
┌─────────────────────────────────────────────────────────────┐
│                      Policy 层                               │
│  /polocies/heuritic/      启发式策略 (ER, HPCC)              │
│  /polocies/rl/           强化学习策略 (A2C, PPO, D3QN)       │
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
# envs/BaseEnvs.py:44-71
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
# polocies/heuritic/ER.py:80-105
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

### PatrolWorld (`envs/PatrolCore.py`)

物理世界模拟器，职责：
- 管理图结构和智能体位置
- 处理智能体移动、等待、节点空闲度更新
- 提供时间推进：`tick(dt)` 和 `tick_to_next_event()`

**关键方法**：
- `tick(dt)`: 推进指定时间量，返回 `TickResult`
- `set_move_action(agent_id, target_node)`: 设置移动动作
- `set_wait_action(agent_id)`: 设置等待动作
- `is_ready(agent_id)`: 智能体是否可以决策

### EventDrivenEnv (`envs/BaseEnvs.py`)

抽象基类，继承自 `PettingZoo.ParallelEnv`。子类必须实现：
- `observation_space(agent)`: 观测空间
- `action_space(agent)`: 动作空间
- `_build_obs(result)`: 构建观测
- `_build_info(result)`: 构建 info（必须包含 `active_mask`）
- `_compute_rewards(result)`: 计算奖励
- `_action_to_target(agent_id, action_idx)`: 动作索引 → 目标节点

### MASUPEnv (`envs/MASUPEnv.py`)

主要环境实现，特点：
- **T_time 截断**：`obs_timer < T_time` 时 `worst_idleness` 保持为 0
- **两种截断模式**：按时间 (`truncate_by_time=True`) 或按步数
- **观测空间类型**：`agent-index`（one-hot 编码）或 `position`（位置信息）

### HeuristicBasePolicy (`polocies/heuritic/HeuristicBase.py`)

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

### ERPolicy (`polocies/heuritic/ER.py`)

Expected Idleness 启发式策略：
- 对每个邻居 v，计算到达时的预期空闲度 `I_exp`
- 如果其他智能体有意图 (v, eta)，则 `I_exp = t_next - eta`
- 否则 `I_exp = t_next - last_visit_time[v]`
- 选择效用最大的邻居：`U = |I_exp| / travel_time`

## 运行命令

```bash
# 启发式策略采样 - 收集专家演示数据
python trainers/imitator/heuristic_sampler.py

# MAPPO 训练 - 使用 OnPolicyTrainer（推荐）
python train_mappo.py

# MAPPO 训练 - 使用 main.py（手动训练循环）
python main.py

# 测试训练好的策略
python evaluators/test.py --model models/mappo/imi/final --num_episodes 10

# 测试并保存可视化结果
python evaluators/test.py --model models/mappo/imi/final --num_episodes 10 --save_plot evaluators/results/rl_eval.png --no_show
```

## 完整训练流程

框架支持从启发式策略到强化学习的完整训练流程：

```
1. 启发式采样 → 2. 模仿学习预训练 → 3. RL 微调
```

### 1. 启发式采样 (`trainers/imitator/heuristic_sampler.py`)

使用启发式策略（如 ER）收集专家演示数据，保存为 NPZ 格式：

```python
# 数据格式:
- obs: [N, T, M, Obs_Dim] - Actor 输入
- critic_states: [N, T, M, State_Dim+M] - Centralized Critic 输入
- actions: [N, T, M, 1] - 动作索引
- action_masks: [N, T, M, Act_Dim] - 动作掩码
- rewards: [N, T, M, 1] - 每个智能体的奖励
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
from polocies.heuritic.ER import ERPolicy
from envs.MASUPEnv import MASUPEnv
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

编辑 `MASUPEnv._build_obs()` 和 `observation_space()`，注意保持上下界一致。

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

### 多智能体策略 (`polocies/marl/`)

- **MultiAgentPolicy**: 包装单个 Policy 实现多智能体接口
  - `shared=True`: 所有智能体共享参数
  - 支持 `action_mask` 参数用于无效动作屏蔽

### 数据采集 (`data/`)

- **MACollector** (`data/collector.py`): 多智能体数据采集器
- **CollectResult** (`data/batch.py`): 采集结果数据类，包含 transitions、episode_rewards 等

