# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Multi-Agent Patrolling (MAP) imitation learning framework** for studying coordination algorithms in graph environments. The framework supports both heuristic and reinforcement learning approaches with an event-driven synchronization mechanism.

## Core Architecture

### Three-Layer Separation Design

```
┌─────────────────────────────────────────────────────────────┐
│                      Policy Layer                            │
│  /policies/heuritic/      Heuristic policies (ER, HPCC)     │
│  /policies/rl/           RL policies (ActorPolicy)         │
│  /policies/marl/         Multi-agent policies              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Environment Layer                         │
│  EventDrivenEnv          Event-driven sync base class       │
│  MASUPEnv                Main environment implementation    │
│  PatrolWorld             Physical world simulator          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                              │
│  Graph                   Graph structure                    │
│  EpisodeMetricsTracker   Metric tracking (IGI, AGI, IWI, WI)│
└─────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

**1. Event-Driven Decision Making**

The environment advances to "next event" (agent arrival or wait completion) rather than fixed timesteps. Only agents in `READY` state need decisions - agents on edges don't receive actions.

**2. Sequential Decision Protocol**

Heuristic policies use sequential decision: Agent 0 → Agent 1 → Agent 2 → ... Each agent's intention is immediately recorded to `er_intention_eta`, so subsequent agents can avoid conflicts.

**3. Modular Configuration System**

YAML-based configuration with registry pattern factory methods. All components (environments, algorithms, networks, trainers) are registered in `configs/registry.py` and created via factory functions.

## Running Commands

### Training

```bash
# Unified training entry point (recommended)
python train.py configs/experiments/masup/mappo_masup_tsp12_imi.yaml
python train.py configs/experiments/masup/mappo_masup_tsp12_scratch.yaml

# With wandb sweep
python sweep.py --base-config configs/experiments/masup/mappo_masup_tsp12_imi.yaml --sweep-config configs/sweep/masup/mappo_masup.yaml
```

### Data Collection

```bash
# Collect expert demonstrations using heuristic policies
python trainers/imitator/heuristic_sampler.py --num_episodes 50000 --num_workers 4
```

### Evaluation

```bash
# Test trained models
python evaluators/test.py --model models/mappo/imi/final --num_episodes 10 --save_plot evaluators/results/rl_eval.png --no_show

# Generate MP4 animation
python evaluators/heuristic_evaluator.py --policy ER --save_animation evaluators/results/ER_animation.mp4
```

## Configuration System

The framework uses a centralized configuration system in `configs/`:

- **Experiments**: `configs/experiments/*.yaml` - Full training configurations
- **Policies**: `configs/policies/*.yaml` - Heuristic policy parameters
- **Networks**: Network architectures via `actor_type` / `critic_type` / `q_type` dispatch
- **Evaluation**: `configs/eval/*.yaml` - Environment configs for evaluation

### Registry Pattern

All components are registered in `configs/registry.py`:
- `ALGO_REGISTRY`: PPO, MAPPO, IPPO, VDPPO, D3QN, IQL, VDN, QMIX
- `ENV_REGISTRY`: MASUP, OUCS, S4R1 environments
- `ACTOR_REGISTRY`: ActorMLP, ActorRNN
- `CRITIC_REGISTRY`: CriticMLP, CriticRNN
- `Q_NETWORK_REGISTRY`: QMLP, QRNN
- `TRAINER_REGISTRY`: On-policy, Off-policy trainers

Factory functions automatically instantiate components from YAML config:
```python
from configs.registry import load_config, create_vec_env, create_actor, create_critic, create_q_network

config = load_config("configs/experiments/mappo_tsp12_imi.yaml")
vec_env = create_vec_env(config.env_type, config.env, num_envs=16)
actor = create_actor(config.actor_type, config.actor, obs_dim, action_dim, device)
critic = create_critic(config.critic_type, config.critic, critic_input_dim, device)
q_net = create_q_network(config.q_type, config.q_network, input_dim, action_dim, device)
```

## Network Architecture

### Overview

网络模块分为三个文件: `networks/mlp.py`, `networks/rnn.py`, `networks/mixing.py`。
Actor / Critic / Q-network 三路独立配置，YAML 中通过 `actor_type` / `critic_type` / `q_type` 分发字段选择对应的网络类和配置类。

### Component-Algorithm Mapping

```
                  actor    critic   q_network   mixer
PPO/IPPO/MAPPO    yes      yes      -           -
VDPPO             yes      -        yes         QPLEXMixer
D3QN              -        -        yes         -
IQL               -        -        yes         -
VDN               -        -        yes         SumMixer (无参数)
QMIX              -        -        yes         QMIXMixer (超网络)
```

### Network Classes

#### MLP Networks (`networks/mlp.py`)

| Class | Signature | Output std | Use Case |
|-------|-----------|------------|----------|
| `ActorMLP` | `(input_dim, hidden_sizes, output_dim)` | 0.01 | PPO/MAPPO/IPPO actor |
| `CriticMLP` | `(input_dim, hidden_sizes, output_dim=1)` | 1.0 | PPO/MAPPO/IPPO critic (value) |
| `QMLP` | `(input_dim, hidden_sizes, output_dim, dueling=False)` | 1.0 | DQN/IQL/VDPPO Q-network |

`QMLP` 的 `dueling=True` 启用 Dueling 架构: `Q = V + A - mean(A)`。

#### RNN Networks (`networks/rnn.py`)

所有 RNN 网络继承 `_BaseRNN`，支持 single-step `forward(obs, hidden)` 和 sequence `forward_sequence(obs_seq, hidden)`。

| Class | Signature | Output std | Use Case |
|-------|-----------|------------|----------|
| `ActorRNN` | `(input_dim, hidden_size, output_dim, num_layers=1, rnn_type="gru")` | 0.01 | RNN actor |
| `CriticRNN` | `(input_dim, hidden_size, output_dim=1, num_layers=1, rnn_type="gru")` | 1.0 | RNN critic (value) |
| `QRNN` | `(input_dim, hidden_size, output_dim, num_layers=1, rnn_type="gru", dueling=False)` | 1.0 | RNN Q-network |

Hidden state 格式: `(recurrent_N, batch, hidden_size)`，其中 `recurrent_N = num_layers * (2 if LSTM else 1)`。
`get_initial_hidden(batch_size, device)` 返回全零初始 hidden state。

#### Mixing Networks (`networks/mixing.py`)

| Class | Signature | Use Case |
|-------|-----------|----------|
| `SumMixer` | `(n_agents, state_dim=0)` | VDN: Q_tot = Σ Q_i (无可学习参数) |
| `QMIXMixer` | `(n_agents, state_dim, embed_dim=32)` | QMIX: 超网络保证单调性 |
| `QPLEXMixer` | `(n_agents, state_dim, embed_dim)` | VDPPO Q-value decomposition |

### Config Dataclasses (`configs/network_configs.py`)

```
NetworkConfig (base, 无字段)
├── MLPConfig (hidden: List[int] = [256, 256])
│   └── QMLPConfig (hidden, dueling: bool = False)
└── RNNConfig (rnn_type: str = "gru", hidden_size: int = 64, num_layers: int = 1)
    └── QRNNConfig (rnn_type, hidden_size, num_layers, dueling: bool = False)
```

YAML 中 `actor_type` / `critic_type` 为 `"mlp"` 时使用 `MLPConfig`，为 `"rnn"` 时使用 `RNNConfig`。
`q_type` 为 `"mlp"` 时使用 `QMLPConfig`，为 `"rnn"` 时使用 `QRNNConfig`（含 `dueling` 字段）。

### YAML Configuration Examples

#### MAPPO (MLP Actor + MLP Critic)

```yaml
algo_name: mappo
actor_type: mlp
critic_type: mlp
actor:
  hidden: [256, 256]
critic:
  hidden: [256, 256]
```

#### MAPPO (RNN Actor + MLP Critic)

```yaml
algo_name: mappo
actor_type: rnn
critic_type: mlp
actor:
  rnn_type: gru
  hidden_size: 64
  num_layers: 1
critic:
  hidden: [256, 256]
```

#### MAPPO (RNN Actor + RNN Critic)

```yaml
algo_name: mappo
actor_type: rnn
critic_type: rnn
actor:
  rnn_type: gru
  hidden_size: 64
  num_layers: 1
critic:
  rnn_type: gru
  hidden_size: 64
  num_layers: 1
```

#### DQN / IQL (MLP Q-network)

```yaml
algo_name: d3qn   # or iql
q_type: mlp
q_network:
  hidden: [256, 256]
  dueling: true
```

#### DQN / IQL (RNN Q-network)

```yaml
algo_name: d3qn   # or iql
q_type: rnn
q_network:
  rnn_type: gru
  hidden_size: 64
  num_layers: 1
  dueling: true
algo:
  seq_len: 20          # RNN 训练序列长度
  max_episodes: 5000   # EpisodeReplayBuffer 容量
```

#### VDN (Q-network, 无 state)

```yaml
algo_name: vdn
q_type: mlp
q_network:
  hidden: [256, 256]
  dueling: false
algo:
  shared_policy: true
  use_double_dqn: true
```

#### QMIX (Q-network, 需要 global state)

```yaml
algo_name: qmix
q_type: mlp
q_network:
  hidden: [256, 256]
  dueling: false
algo:
  shared_policy: true
  use_double_dqn: true
  mixer_embed_dim: 32
```

#### VDPPO (Actor + Q-network)

```yaml
algo_name: vdppo
actor_type: mlp
q_type: mlp
actor:
  hidden: [256, 256]
q_network:
  hidden: [64, 64]
  dueling: false
```

### Factory Functions (`configs/registry.py`)

```python
create_actor(actor_type, actor_config, obs_dim, action_dim, device) -> nn.Module
create_critic(critic_type, critic_config, critic_input_dim, device) -> nn.Module
create_q_network(q_type, q_config, input_dim, action_dim, device) -> nn.Module
```

Actor / Critic / Q-network 完全解耦，可以任意组合（如 RNN actor + MLP critic）。
算法层通过 `is_recurrent` / `is_critic_recurrent` / `is_any_recurrent` 属性自动适配不同组合的训练逻辑。

### RNN Critic Integration

当 critic 为 RNN 时，算法层 (`algorithm_base.py`) 会：
1. 在 `prepare_batch` 中逐步处理 critic 输入序列，done 边界重置 hidden state
2. 将每步 critic hidden 存入 `batch.critic_rnn_hidden`
3. 在 `update` 中使用 `critic.forward_sequence()` 和 `chunk_split` 处理 minibatch

`is_any_recurrent = is_recurrent or is_critic_recurrent`：只要 actor 或 critic 任一为 RNN，就使用 `chunk_split` 进行 minibatch 切分。

### Off-Policy RNN (DRQN) Integration

当 Q-network 为 RNN (QRNN) 时，off-policy 算法 (D3QN, IQL) 使用 DRQN 风格训练：

**数据存储**: `EpisodeReplayBuffer` 按整条 episode 存储，采样时切固定长度 `seq_len` 子序列。
短于 `seq_len` 的 episode 右侧 zero-pad，对应 `mask=0`。

**训练流程**:
1. Collector 维护 per-env RNN hidden state，episode 边界重置
2. Episode 结束时推入 `EpisodeReplayBuffer`
3. 采样 `SequenceBatch`（shape `(batch, seq_len, ...)`）
4. Online / Target Q-network 均用 `forward_sequence` 处理序列，`h0=zeros`
5. Loss 使用 `mask` 忽略 padding 步

**关键参数** (`OffPolicyParams`):
- `seq_len`: RNN 训练序列长度（默认 20）
- `max_episodes`: EpisodeReplayBuffer 容量（默认 5000）
- `burn_in_len`: 可选 burn-in 长度（预留，暂为 0）

**数据结构**:
- `SequenceBatch`: 所有字段 shape `(B, L, ...)`，含 `mask` 字段 `(B, L)`
- `EpisodeReplayBuffer`: deque 存储完整 episode，`sample(batch_size, seq_len)` 返回 `SequenceBatch`

MLP Q-network 路径通过 `is_recurrent` 条件守卫完全不受影响。

## 模型参数保存与读取

### 保存流程

训练结束（或 checkpoint 触发）时，`train.py` 调用 `net.get_config_dict(input_dim, output_dim)` 获得包含完整重建信息的 config dict，由 `utils/model_io.save_model` 写入 `model_dir/config.yaml`；权重写入 `policy.pt` / `critic.pt`。

```
train.py
  └── _make_net_config_dict(net, input_dim, output_dim)
        └── net.get_config_dict(input_dim, output_dim)  ← 网络类实现
              └── save_model → config.yaml + policy.pt / critic.pt
```

### 读取流程

评估时 `evaluators/test.py` 调用 `load_policy_for_eval`，读取 `config.yaml`，通过 `registry[type].from_config_dict(cfg)` 重建网络结构，再加载 `policy.pt` 权重。

```
evaluators/test.py
  └── load_policy_for_eval(model_dir, ...)
        ├── 读取 config.yaml
        ├── _get_class_registry()[type].from_config_dict(cfg)  ← 网络类实现
        └── 加载 policy.pt 权重
```

### 网络类必须实现的协议

所有参与 checkpoint 保存与评估加载的网络类必须实现：

```python
def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
    """返回可重建该网络的完整配置 dict，必须包含 type、input_dim、output_dim。
    
    推荐直接返回 self.input_dim / self.output_dim，而非使用传入的参数，
    这样无论调用方是否传入正确值，序列化结果始终与实例一致。
    """

@classmethod
def from_config_dict(cls, cfg: dict) -> nn.Module:
    """从 get_config_dict 产出的 cfg 重建网络实例。"""
```

同时必须在 `utils/model_io._get_class_registry` 中注册（类名 → 类对象映射）。

### 新增网络类规范（必须遵守）

新增任何 Actor / Critic / Q-network 类时，必须满足以下全部要求：

#### 1. 统一实例属性命名

所有网络类**必须**使用 `self.input_dim` 和 `self.output_dim` 存储输入输出维度，禁止使用其他命名（如 `self.obs_dim`、`self.action_dim`）：

```python
class MyActorNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, ...):
        super().__init__()
        self.input_dim = input_dim   # 必须，统一命名
        self.output_dim = output_dim # 必须，统一命名
        ...
```

这是因为 `imitation_trainer._save_actor` / `_save_checkpoint`、`train.py` 中的 `_make_net_config_dict` 均通过 `net.input_dim` / `net.output_dim` 读取维度。若命名不一致，checkpoint 保存时会抛出 `AttributeError`。

#### 2. 实现 `get_config_dict`

返回包含所有重建所需字段的 dict，`input_dim`/`output_dim` 从 `self` 读取：

```python
def get_config_dict(self, input_dim: int, output_dim: int) -> dict:
    return {
        "type": type(self).__name__,
        "input_dim": self.input_dim,    # 从 self 读取，忽略传入的参数
        "output_dim": self.output_dim,  # 从 self 读取，忽略传入的参数
        # ... 其他重建所需的超参数 ...
    }
```

#### 3. 实现 `from_config_dict`

从 `get_config_dict` 产出的 dict 完整重建实例：

```python
@classmethod
def from_config_dict(cls, cfg: dict) -> "MyActorNet":
    return cls(
        input_dim=cfg["input_dim"],
        output_dim=cfg["output_dim"],
        # ... 其他超参数从 cfg 读取 ...
    )
```

#### 4. 在 `ACTOR_REGISTRY` / `CRITIC_REGISTRY` / `Q_NETWORK_REGISTRY` 中注册

```python
# configs/registry.py
ACTOR_REGISTRY["my_net"] = {
    "module": "networks.my_module",
    "class_name": "MyActorNet",
    "config_class": MyNetConfig,   # 对应的 NetworkConfig 子类
}
```

#### 5. 在 `_get_class_registry` 中注册（供评估加载使用）

```python
# utils/model_io.py  _get_class_registry()
from networks.my_module import MyActorNet
registry["MyActorNet"] = MyActorNet
```

#### 6. 添加 Config Dataclass（如需 YAML 配置）

```python
# configs/network_configs.py
@dataclass(kw_only=True)
class MyNetConfig(NetworkConfig):
    my_param: int = 128
```

#### 7. 在 `create_actor` / `create_critic` 工厂函数中添加分支

```python
# configs/registry.py  create_actor()
elif isinstance(actor_config, MyNetConfig):
    return cls(input_dim=obs_dim, output_dim=action_dim,
               my_param=actor_config.my_param).to(device)
```

#### 完整 Checklist

新增网络类时逐项确认：

- [ ] `__init__` 中使用 `self.input_dim` / `self.output_dim`
- [ ] 实现 `get_config_dict`，内部引用 `self.input_dim` / `self.output_dim`
- [ ] 实现 `from_config_dict`（classmethod）
- [ ] 在 `ACTOR_REGISTRY` / `CRITIC_REGISTRY` / `Q_NETWORK_REGISTRY` 中注册
- [ ] 在 `utils/model_io._get_class_registry` 中注册
- [ ] （可选）在 `configs/network_configs.py` 中添加 Config Dataclass
- [ ] （可选）在 `create_actor` / `create_critic` 工厂函数中添加分支

### 各网络类实现位置

| 网络类 | 文件 |
|--------|------|
| ActorMLP, CriticMLP, QMLP | `networks/mlp.py` |
| ActorRNN, CriticRNN, QRNN | `networks/rnn.py` |
| SUNActor, SUNCritic | `networks/custom/suns.py` |
| MPNNActor | `networks/gnn.py` |

## Core Classes

### PatrolWorld (`envs/mdps/patrol_core.py`)

Physical world simulator managing the graph, agent positions, and node idleness.

**Key Methods:**
- `tick(dt)`: Advance simulation by specified time, returns `TickResult`
- `tick_to_next_event()`: Advance to next event (agent arrival/wait completion)
- `set_move_action(agent_id, target_node)`: Set agent movement
- `set_wait_action(agent_id)`: Set agent to wait
- `is_ready(agent_id)`: Check if agent can make decisions

### EventDrivenEnv (`envs/mdps/base_envs.py`)

Abstract base class extending PettingZoo `ParallelEnv`. Subclasses must implement:
- `observation_space(agent)`, `action_space(agent)`
- `_build_obs(result)`, `_build_info(result)`, `_compute_rewards(result)`
- `_action_to_target(agent_id, action_idx)`: Convert action index to target node

### MASUPEnv (`envs/mdps/masup.py`)

Main environment implementation with:
- **T_time truncation**: `worst_idleness` stays 0 until `obs_timer >= T_time`
- **Two truncation modes**: Time-based or step-based
- **Observation types**: `agent-index` (one-hot), `position`, or `decision`
- **Reward scaling**: `reward_scale`, `idi_scale`, `contribution_scale`

### Heuristic Policies (`policies/heuritic/`)

Base class `HeuriticBasePolicy` requires implementing:
- `compute_actions(obs_dict, global_state)`: Compute actions for all agents
- `_compute_action(agent_idx, obs, global_state)`: Single agent decision

**ERPolicy** (`policies/heuritic/er.py`):
- Computes expected idleness for each neighbor
- Uses intention table (`er_intention_eta`) for conflict avoidance
- Utility: `U = |I_exp| / travel_time`

**HPCCPolicy** (`policies/heuritic/hpcc.py`):
- Weighted utility: `U = w_idl * U_idl + w_dist * U_dist + w_conflict * U_conflict`
- Supports different distance metrics (shortest path, Euclidean)
- Configurable coordination parameters

### MultiAgentPolicy (`policies/marl/marl_base.py`)

Wrapper for single-agent policies to support multi-agent interface:
- `shared=True`: All agents share parameters
- Supports `action_mask` for invalid action filtering

## Data Format

### Heuristic Sampler Output

NPZ/HDF5 format with arrays:
- `obs`: [N, T, M, Obs_Dim] - Actor inputs
- `critic_states`: [N, T, M, State_Dim+M] - Centralized Critic inputs (global state + agent one-hot)
- `actions`: [N, T, M, 1] - Action indices
- `action_masks`: [N, T, M, Act_Dim] - Action masks
- `rewards`: [N, T, M, 1] - Per-agent rewards
- `padded_mask`: [N, T, 1] - Padding mask (1=real, 0=padding)
- `returns`: [N, T, M, 1] - Discounted cumulative returns

## Graph File Format

JSON files in `graphs/` directory:
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

- `nodes`: Node IDs
- `edges`: Directed edges with travel time weights
- `phi`: Node priorities for weighted idleness calculation

## Important Conventions

1. **Action index 0**: Always represents "wait" action when `enable_wait=True`
2. **info['active_mask']**: Required in all environments to mark which agents need decisions
3. **Sequential heuristic decisions**: Agent 0 decides first, intention recorded, then Agent 1, etc.
4. **HuggingFace-style model format**: `model_dir/` contains `config.yaml`, `policy.pt`, `critic.pt`
5. **训练与评估闭环**：新增任何环境类或网络类时，必须同时考虑训练和评估两个环节。
   - **新环境**：在 `configs/registry.py` 的 `ENV_REGISTRY` 注册，并在 `configs/eval/` 下提供对应的 env_config YAML，确保 `evaluators/test.py` 能正确加载。
   - **新网络**：实现 `get_config_dict(input_dim, output_dim)` 和 `from_config_dict(cls, cfg)` 两个方法，并在 `utils/model_io._get_class_registry` 中注册，否则 checkpoint 无法在评估时重建网络。如不遵守，`train.py` 和 `imitation_trainer` 会在 checkpoint 保存时抛出 `NotImplementedError`。

## Algorithm Factory System

### 工作原理

`create_algorithm()` 使用 `inspect` 自动过滤参数：根据算法类 `__init__` 签名中声明的参数名，
从 `context_kwargs` 池中只传递匹配的参数，多余的自动忽略。

```python
# train.py 中构建 context_kwargs 池（包含所有可能需要的参数）
context_kwargs = {
    "num_envs": ..., "action_dim": ..., "state_dim": ..., "n_agents": ...,
    "critic": critic_net,  "q_network": q_net,
    "total_iterations": ..., "optimizer_steps_per_iter": ...,
    "value_norm_config": ...,
}

# create_algorithm 自动过滤：只传算法 __init__ 中声明了的参数
algorithm = create_algorithm(algo_name, policy, algo_params, **context_kwargs)
```

**重要**：算法类的 `__init__` 不应使用 `**kwargs`，所有需要的参数必须显式声明。
这样如果传入了不存在的参数名（如拼写错误），Python 会在 `create_algorithm` 阶段安全忽略，
而非静默吞掉；如果声明了但 `context_kwargs` 中没有且无默认值，则会正常报 TypeError。

### 各算法 `__init__` 参数一览

以下列出每个算法类接收的参数（`policy` 和 `params` 是所有算法的公共必需参数，不再重复列出）：

#### On-Policy (Actor-Critic)

| 算法 | 模块 | 额外参数 |
|------|------|---------|
| `PPOAlgo` | `algorithms.rl.ppo` | `critic`, `num_envs`, `total_iterations`, `optimizer_steps_per_iter`, `value_norm_config` |
| `IPPOAlgo` | `algorithms.marl.ippo` | `critic`, `num_envs`, `total_iterations`, `optimizer_steps_per_iter`, `value_norm_config` |
| `MAPPOAlgo` | `algorithms.marl.mappo` | `critic`, `num_envs`, `total_iterations`, `optimizer_steps_per_iter`, `value_norm_config` |
| `VDPPOAlgo` | `algorithms.marl.vdppo` | `critic`, `num_envs`, `action_dim`, `state_dim`, `n_agents`, `total_iterations`, `optimizer_steps_per_iter`, `value_norm_config`, `q_network` |

#### Off-Policy (Value-Based)

| 算法 | 模块 | 额外参数 |
|------|------|---------|
| `D3QNAlgo` | `algorithms.rl.d3qn` | *(无额外参数)* |
| `IQLAlgo` | `algorithms.marl.iql` | *(无额外参数)* |
| `VDNAlgo` | `algorithms.marl.vdn` | `n_agents`, `state_dim` |
| `QMIXAlgo` | `algorithms.marl.qmix` | `n_agents`, `state_dim` |

### VDN/QMIX 继承关系

```
VDNAlgo (BaseAlgorithm)
└── QMIXAlgo
```

QMIX 继承 VDN，通过 override 三个扩展点实现差异：

| 扩展点方法 | VDN | QMIX |
|-----------|-----|------|
| `_init_mixer()` | SumMixer (无参数) | QMIXMixer (超网络, 需 state) |
| `_init_optimizer()` | 仅 Q-network 参数 | Q-network + Mixer 参数 |
| `_update_target_networks()` | 仅 target Q-network | target Q-network + target Mixer |

`_compute_loss()` 和 `update()` 完全复用基类。Mixer 的多态性（SumMixer 忽略 state，QMIXMixer 使用 state）
由各 Mixer 的 `forward()` 自动处理。

### 奖励聚合

VDN/QMIX 的联合奖励使用 `r_tot = Σ r_i`（各 agent 奖励之和），
与 `Q_tot = Σ Q_i`（VDN）或 `Q_tot = f(Q_1,...,Q_N)`（QMIX）的值分解语义一致。

### Global State 需求

- **QMIX**: `collect_state=True`，ReplayBuffer 开启 `has_state=True`，Mixer 超网络需要 state
- **VDN**: `collect_state=False`，SumMixer 不使用 state，节省内存和采集开销

## 图结构 Actor 观测协议（MPNNActor / GraphSageActor）

`networks/gnn.py` 提供两个图结构 Actor，均不绑定具体环境实现：

| Actor | YAML `actor_type` | 架构特点 |
|-------|-------------------|---------|
| `MPNNActor` | `mpnn` | 节点+边+全局三路更新；global_feat 进入网络 |
| `GraphSageActor` | `sage` | 严格复现 Algorithm 1（SAGE + edge attr）；边静态不更新；global_feat **不**进入网络，仅用于定位 identity |

两者共享完全相同的观测拼接协议；差异体现在 `actor:` 段的配置字段（见下方）。
任何环境若想使用上述任意 Actor，必须遵守以下观测构造协议。

### 观测向量拼接顺序（固定）

```
[node_feats]    num_nodes × node_feat_dim  个 float
[edge_src]      max_edges                  个 float  （边源节点索引）
[edge_dst]      max_edges                  个 float  （边目标节点索引）
[edge_attr]     max_edges × edge_feat_dim  个 float
[edge_mask]     max_edges                  个 float  （1=有效, 0=填充）
[global_feat]   global_feat_dim            个 float
[identity]      identity_dim               个 float  （见下方说明）
```

其中：
- `num_nodes = 静态节点数 + agent_num`（每个 agent 对应一个虚拟节点）
- `max_edges = 静态有向边数 + agent_num × 2`（每个 agent 的动态边上限）
- 静态节点下标：`0 ~ 静态节点数-1`；虚拟 agent 节点下标：`静态节点数 + agent_idx`（0-based）
- `edge_src` / `edge_dst` 填上述下标（float）；填充边索引任意，被 `edge_mask=0` 屏蔽

### identity 与 role_imformation

| role_imformation | identity 编码                                   | identity_dim    |
|------------------|--------------------------------------------------|-----------------|
| `agent-index`    | one-hot，第 `i` 位为 1                          | `agent_num`     |
| `position`       | `[float(当前节点下标)]`                          | `1`             |
| `decision`       | `[float(当前节点下标)] + one-hot(决策顺序索引)` | `1 + agent_num` |

### YAML actor 段必填字段

**MPNNActor（`actor_type: mpnn`）**

```yaml
actor_type: mpnn
actor:
  graph_path: graphs/xxx.json    # 用于推导 num_nodes / max_edges
  agent_num: N
  role_imformation: agent-index  # 须与环境 custom_configs.role_imformation 一致
  node_feat_dim: 2               # 须与环境 _build_obs 实际维度严格一致
  edge_feat_dim: 1
  global_feat_dim: 2
  hidden_dim: 64
  gnn_layers: 2
  actor_mlp_layers: 1
  gnn_mlp_layers: 2              # MPNNActor 专有：MPNN 内部 MLP 层数
  mlp_activation: silu
```

**GraphSageActor（`actor_type: sage`）**

```yaml
actor_type: sage
actor:
  graph_path: graphs/xxx.json
  agent_num: N
  role_imformation: agent-index
  node_feat_dim: 2
  edge_feat_dim: 1
  global_feat_dim: 2             # 仅用于 obs 解码，不进入网络
  hidden_dim: 64
  gnn_layers: 2                  # Algorithm 1 中的 K（层数）
  actor_mlp_layers: 1
  mlp_activation: relu           # 默认 relu，对应原论文非线性 σ
  # 无 gnn_mlp_layers：SAGE 每层仅单线性变换 W_k，无内部 MLP
```

> `node_feat_dim` / `edge_feat_dim` / `global_feat_dim` 与 `MASUPGraphEnv` 的 `custom_configs`
> 同名字段含义相同，二者必须严格一致，否则观测解包会越界或产生静默错误。

## Common Development Tasks

### Adding a new RL algorithm

新增算法只需 **2 个文件改动**（算法类 + 注册表），无需修改 `train.py`：

**步骤 1：实现算法类**

```python
# algorithms/marl/my_algo.py
from algorithms.algorithm_base import BaseAlgorithm, TrainingStats

class MyAlgo(BaseAlgorithm):
    def __init__(
        self,
        policy,
        params: MyAlgoParams,
        n_agents: int = 1,       # 从 context_kwargs 池中按需声明
        state_dim: int = 0,      # 只声明需要的参数
    ):
        super().__init__(policy)
        ...

    def update(self, batch, **kwargs) -> TrainingStats:
        ...
```

**规则**：
- `__init__` 中 **不要** 使用 `**kwargs`，所有参数必须显式声明
- `policy` 和 `params` 是必需的（`create_algorithm` 固定传入）
- 其余参数从 `context_kwargs` 池中按名称匹配（见上方参数一览表）
- 如需全新的上下文参数（池中没有的），则需在 `train.py` 中添加到 `context_kwargs` 池

**步骤 2：注册到 `ALGO_REGISTRY`**

```python
# configs/registry.py 的 ALGO_REGISTRY 中添加一项
"my_algo": {
    "module": "algorithms.marl.my_algo",
    "class_name": "MyAlgo",
    "params_class": MyAlgoParams,
    "trainer_type": "off_policy",   # "on_policy" or "off_policy"
    "policy_type": "value",         # "actor" or "value"
},
```

**步骤 3（如需）：添加参数 dataclass**

```python
# configs/algo_configs.py
@dataclass(kw_only=True)
class MyAlgoParams(OffPolicyParams):  # 或继承其他合适的基类
    my_custom_param: float = 0.1
```

**步骤 4：编写实验配置 YAML**

```yaml
# configs/experiments/my_algo_tsp12.yaml
algo_name: my_algo
env_type: masup
q_type: mlp       # 或 actor_type / critic_type

algo:
  my_custom_param: 0.2
  ...
```

完成以上步骤后，即可通过 `python train.py configs/experiments/my_algo_tsp12.yaml` 启动训练。

### 继承现有算法

如果新算法是对现有算法的改进（如 QMIX 之于 VDN），推荐继承：

```python
class MyImprovedAlgo(VDNAlgo):
    """只 override 差异化的部分。"""

    def _init_mixer(self, n_agents, state_dim, params):
        self.mixer = MyMixer(...)
        self.target_mixer = copy.deepcopy(self.mixer)
        ...

    def _init_optimizer(self, params):
        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.mixer.parameters()),
            lr=params.lr,
        )
```

### Adding a new heuristic policy

1. Inherit from `HeuriticBasePolicy` in `policies/heuritic/`
2. Implement `compute_actions()` and `_compute_action()`
3. Create config in `configs/policies/`
4. Update `HeuristicSampler` to support new policy if needed

### Adding a new graph environment

1. Add JSON file to `graphs/` directory
2. Update `configs/MASUPEnv.yaml` or create new environment config

### Modifying observation space

Edit `envs/mdps/masup.py`:
- `_build_obs()` for observation construction
- `observation_space()` for space definition
- Ensure bounds match between space and actual observations

## A+ 方案：active_mask 透明 GAE + Loss Masking

### 背景

在多智能体巡逻环境中，智能体在边上移动时（`ON_EDGE` 状态）无法做出有意义的决策，动作被强制限制为 `no-op`。A+ 方案通过 `active_mask` 机制，让这些无效转移不污染策略梯度和价值梯度，同时在 GAE 计算中正确累积跨越 inactive 步的奖励和折扣。

### 数据流

```
Environment._build_info()          Collector                    MAPPO
┌──────────────────────┐    ┌──────────────────────┐    ┌───────────────────────┐
│ active_mask:          │    │ 从 info 提取         │    │ prepare_batch:        │
│   1 = READY (决策点)  │───▶│ active_mask          │───▶│   传入透明 GAE        │
│   0 = ON_EDGE (跳过)  │    │ 存入 buffer          │    │   合并到 RolloutBatch │
└──────────────────────┘    │ 拼接到 RolloutBatch  │    │                       │
                            └──────────────────────┘    │ update:               │
                                                        │   policy loss × am    │
                                                        │   entropy    × am    │
                                                        │   value loss × am    │
                                                        └───────────────────────┘
```

### 涉及文件及改动

#### `data/batch.py`
- `RolloutBatch` 新增 `active_mask: (batch,)` 字段，`1=READY`，`0=ON_EDGE`

#### `data/collector.py` — MACollector
- `_reset_buffer`: 新增 `'active_mask'` 缓冲列表
- `collect`: 从 `info[agent][i]['active_mask']` 提取，默认为 `1`（向后兼容）
- 构建 batch 时 `active_mask` 拼接入 `RolloutBatch`

#### `algorithms/marl/mappo.py` — MAPPOAlgo

**`prepare_batch`**:
- 将 `active_mask` reshape 为 `(T, N)` 传入 GAE
- 合并后的 `RolloutBatch` 包含 `active_mask`
- Value Normalization `ret_rms` 仅用 active 步的 return 更新

**`_compute_gae_vectorized` — 透明 GAE 算法**:

无 `active_mask` 时退化为标准 GAE，完全向后兼容。有 `active_mask` 时：

```
维护累积器 (N,):
  acc_reward:     累积 inactive 步的折扣奖励
  acc_discount:   γ^k (纯折扣因子)
  acc_gae_decay:  (γλ)^k (GAE λ-加权衰减)
  next_active_val: 时间上最近 active 步的 V

反向循环:
  inactive 步: reward 和 discount 累积到累积器
  active 步:
    δ_eff = r_t + γ * (acc_reward + acc_discount * V_next_active) - V_t
    A_t = δ_eff + γλ * acc_gae_decay * last_gae
    重置累积器
  done 边界: 重置所有累积器 + 注入 terminal/trunc bootstrap
```

数学原理：对于 active 步 t，若 t+1 到 t+K-1 都是 inactive 步，t+K 是下一个 active 步：
- 多步 TD-error: `δ_eff = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^K * V_{t+K} - V_t`
- GAE 递推: `A_t = δ_eff + (γλ)^K * A_{t+K}`

**`update` — Loss Masking**:
- Policy loss: `(per_sample_loss × am).sum() / am_sum`
- Entropy loss: `(entropy × am).sum() / am_sum`
- Value loss: `(v_loss_per_sample × am).sum() / am_sum`
- Advantage normalization: 仅对 active 样本计算 mean/std

### 环境端要求

所有环境的 `_build_info` 必须返回 `active_mask` 字段：
```python
def _build_info(self, result):
    infos = {}
    for agent_str in self.agents:
        agent_id = int(agent_str.split('_')[1])
        active_mask = 1 if self.world.is_ready(agent_id) else 0
        infos[agent_str] = {"action_mask": ..., "active_mask": active_mask}
    return infos
```

已支持的环境: `MASUPEnv`（`masup.py`）、`BBLAEnv`（`bbla.py`）及其子类 `GBLAEnv`、`ExGBLAEnv`。

### 向后兼容性

- 环境不提供 `active_mask` 时，默认为 `1`（全部 active）
- GAE 无 `active_mask` 时走标准路径，无任何行为变化
- `update` 中 `am=None` 时退化为原始 `.mean()` 聚合

## 经验回放同步机制（Synchronized Experience Replay）

### 问题

Off-policy 逐步存储 `(obs, act, rew, next_obs)` 时，决策步绑了 reward=0（agent 刚出发），到达步的 arrival reward 绑了 no-op 动作。Q-network 学到的是 Q(ON_EDGE, noop)=high 而非 Q(decision_obs, go_to_B)=high。

### 适用范围

- **IQL（MLP + RNN）**：完整支持。`sync_replay: true` 开启。
- **VDN / QMIX**：**不支持**。

### IQL MLP 同步路径

Collector 层 per-agent per-env 跟踪 pending decision，ON_EDGE 步累积折扣 reward，agent 回到 READY 时 flush 一条同步 transition（含 `gamma_power = γ^k`）。算法层 TD target 使用 `gamma_power` 替代固定 `γ`。

### IQL RNN 多步 Bootstrap

保留所有步（hidden state 连续性），Collector 额外存 `active_mask` 到 `SequenceReplayBuffer`。算法层反向扫描计算多步 TD target 到下一个 active 步，loss 仅在 `active_mask * mask` 上计算。

### VDN / QMIX 不适用说明

VDN/QMIX 的 `Q_tot = mixer(Q_1, ..., Q_N, state)` 要求 per-agent buffer 保持 shared-index 对齐（同一 index = 同一时间步）。同步机制压缩掉 ON_EDGE 步后，各 agent transition 数量不对齐（不同边权 → 不同 decision 间隔），破坏 shared-index 前提。

理论上可用"所有 agent 同时 READY"作为联合同步点，但巡逻环境中 agent 到达后立即出发，READY 窗口仅 1 step，多 agent 同时 READY 要求边权恰好对齐（LCM），在不规则图上几乎不可能。

### 配置

```yaml
algo:
  sync_replay: true   # 仅 IQL 生效
```

### 向后兼容

- `sync_replay: false`（默认）时所有组件走原有路径，零行为变化
- `gamma_power=None` 时 MLP 退化为标准 `γ` bootstrap
- `active_mask=None` 时 RNN loss 退化为标准全步计算

## Key Dependencies

- PettingZoo: Multi-agent environment interface
- PyTorch: Neural network implementation
- YAML: Configuration files
- NetworkX: Graph algorithms and visualization
- WandB (optional): Experiment tracking

## Directory Structure

- `envs/`: Environment implementations (mdps/, venvs/)
- `policies/`: Policy implementations (heuritic/, rl/, marl/)
- `trainers/`: Training modules (rl_trainer.py, imitator/)
- `configs/`: YAML configuration system with registry
- `graphs/`: Environment graph files (JSON)
- `networks/`: Neural network architectures
- `data/`: Data collection utilities
- `utils/`: Visualization, logging, model I/O
- `evaluators/`: Testing and evaluation tools
- `algorithms/`: RL algorithms (ppo, mappo, ippo, vdppo, d3qn, iql, vdn, qmix)
