# Q-table 算法：obs 编码与环境兼容性说明

## 1. Obs 向量如何作为 Q-table 的 key？

**是的**，当前实现默认将环境返回的 `obs` 向量处理为元组作为 table 的键。

### 实现位置

`algorithms/tabular/qtable.py`:

```python
def _obs_to_key(self, obs: np.ndarray) -> tuple:
    return tuple(obs.flat)
```

### 行为说明

- `obs.flat`：按 C 顺序遍历 `obs` 的所有元素（支持 1D/2D/多维）
- `tuple(...)`：转为不可变元组，可作为 `dict` 的 key
- 适用于**小整数离散观测**（如 BBLA/GBLA 的 4 维整数 obs）

### 注意事项

- 若 obs 为浮点或高维，状态空间会非常大，表格法不适用
- 需保证同一语义状态对应唯一 obs，否则会分散到不同 key（见 `bbla.py` 平局时取 `min(max_nodes)` 的注释）

---

## 2. 是否兼容 ParallelEnv 与 Gymnasium Env？

**当前：仅支持 PettingZoo ParallelEnv，不支持 gymnasium.Env。**

### 支持情况

| 环境类型 | 向量化接口 | Q-table 训练路径 |
|---------|------------|------------------|
| PettingZoo ParallelEnv | `vec_env.agents`, `obs_dict`, `rew[agent][i]` | ✅ 支持 |
| gymnasium.Env | `vec_env.agents=None`, `obs` 为 ndarray | ❌ 会报错 |

### 不兼容原因

`trainers/qtable_trainer.py` 与 `train.py` 的 `_train_qtable` 假定多智能体接口：

1. **`vec_env.agents`**  
   - ParallelEnv：`["agent_0", "agent_1", ...]`  
   - Gymnasium：`None`  
   - 使用处：`for agent in self.agents` 会因 `None` 报错

2. **`action_dim`**  
   - ParallelEnv：`list(vec_env.action_space.values())[0].n`  
   - Gymnasium：`action_space` 为单个 Space，无 `.values()`，会报错

3. **数据格式**  
   - ParallelEnv：`obs_dict[agent][i]`, `rew[agent][i]`, `term[agent][i]`  
   - Gymnasium：`obs[i]`, `rew[i]`, `term[i]`（无 agent 维度）

### 核心算法层

`QTablePolicy` 和 `QTableAlgo` 本身与 agent 数量无关，只处理 `obs`、`action`、`reward` 等标量/向量，理论上可支持单智能体。不兼容的是**训练器**和 `_train_qtable` 对 Multi-agent 接口的依赖。

---

## 3. 兼容 Gymnasium 的改造思路

若需支持 `suns_gym` 等单智能体环境，需在 `QTableTrainer` 和 `_train_qtable` 中：

1. 检测 `vec_env.is_parallel_env`（或 `vec_env.agents is not None`）
2. **Gymnasium 分支**：
   - 使用虚拟 agent：`agents = ["agent_0"]`
   - `obs_dict = {"agent_0": obs}`（obs 为 `(num_envs, *obs_shape)`）
   - `action_space` 直接取 `vec_env.action_space`，不调用 `.values()`
3. **ParallelEnv 分支**：保持现有逻辑不变

---

## 4. 相关文件索引

- `algorithms/tabular/qtable.py`：`_obs_to_key`, `QTablePolicy`, `QTableAlgo`
- `trainers/qtable_trainer.py`：`QTableTrainer`（obs_dict/agents 循环）
- `train.py`：`_train_qtable`（`action_dim`、`agent_ids`）
- `envs/venvs.py`：`BaseVectorEnv` 对两种环境的统一封装
