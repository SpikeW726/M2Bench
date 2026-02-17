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
- **Networks**: Network architectures are defined in experiment configs
- **Evaluation**: `configs/eval/*.yaml` - Environment configs for evaluation

### Registry Pattern

All components are registered in `configs/registry.py`:
- `ALGO_REGISTRY`: MAPPO, IPPO
- `ENV_REGISTRY`: MASUP environment
- `NETWORK_REGISTRY`: MLP networks
- `TRAINER_REGISTRY`: On-policy, Off-policy trainers

Factory functions automatically instantiate components from YAML config:
```python
from configs.registry import load_config, create_vec_env, create_networks

config = load_config("configs/experiments/mappo_tsp12_imi.yaml")
vec_env = create_vec_env(config.env_type, config.env, num_envs=16)
actor, critic = create_networks(config.network_type, config.network, ...)
```

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
- `algorithms/`: RL algorithms (mappo, ippo)
