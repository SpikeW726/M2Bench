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
python train.py configs/experiments/mappo_tsp12_imi.yaml
python train.py configs/experiments/mappo_tsp12_scratch.yaml

# With wandb sweep
python sweep.py --base-config configs/experiments/mappo_tsp12_imi.yaml --sweep-config configs/sweep/mappo_hparam.yaml
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
- `ALGO_REGISTRY`: PPO, MAPPO, IPPO, VDPPO, D3QN, IQL
- `ENV_REGISTRY`: MASUP environment
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
                  actor    critic   q_network
PPO/IPPO/MAPPO    yes      yes      -
VDPPO             yes      -        yes (+ mixer)
DQN               -        -        yes
IQL               -        -        yes
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

## Common Development Tasks

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
- `algorithms/`: RL algorithms (ppo, mappo, ippo, vdppo, d3qn, iql)
