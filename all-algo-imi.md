# Training error and index out of range
_Exported on 2026/4/5 at GMT+8 15:56:08 from Cursor (2.6.12)_

---

**User**

当我从头开始用ippo训练时，actor和critic网络的实例数量以及网络的输入分别是啥，与mappo的区别是啥

---

**Cursor**

正在从代码库中查找 IPPO 与 MAPPO 的策略/价值网络实例化与输入构造方式。



根据当前仓库里的实现，结论如下（以 **PettingZoo 式并行环境、`train.py` 主路径** 为准）。

---

## 1. Actor / Critic 各有几个实例？

- **Critic：始终 1 个** `critic_net`，所有 agent 共用。
- **Actor：当前代码里 IPPO 与 MAPPO 一样，都是 1 个共享 Actor。**

依据在 `train.py` 的 `_build_policy`：只要是并行环境 + actor 类策略，就写死 `shared=True`，没有用 `config.algo.shared_policy` 去区分 IPPO：

```231:238:/home/wzh/MAP-imitation-framework/train.py
        else:
            return MultiAgentPolicy(
                agent_ids=dims["agent_ids"],
                obs_space=dims["obs_space"],
                action_space=dims["action_space"],
                policy_class=ActorPolicy,
                policy_kwargs={"actor": actor_net},
                shared=True,
            )
```

`MultiAgentPolicy` 在 `shared=True` 时只构造 **一个** `policy_class` 实例，所有 agent 的 `get_policy(agent)` 都指向同一套参数：

```45:49:/home/wzh/MAP-imitation-framework/policies/marl/marl_base.py
        if shared:
            # All agents share one policy instance
            self._shared_policy = policy_class(
                obs_space, action_space, **policy_kwargs
            ).to(self.device)
```

因此：**从头训练时，IPPO / MAPPO 在本项目里都是「1 个共享 Actor + 1 个 Critic」**。  
论文里常见的「IPPO = 每 agent 独立 Actor」在这里**没有**通过 `train.py` 打开；`IPPOParams` 里的 `shared_policy` 只作用在 **value 型** MultiAgentPolicy 上，不作用在这条 actor 分支。

---

## 2. 网络输入分别是什么？

- **Actor（每个 agent 一次 forward）**  
  输入是该 agent 的 **局部观测** `obs`，维数 = `observation_space(agent).shape[0]`（同质 agent 时各 agent 维数相同）。内容由具体环境定义（如 MASUP 的拼接观测）。

- **Critic**  
  - **IPPO**：用 `RolloutBatch` 里的 `global_state`；在 `IPPOAlgo.prepare_batch` 里为  
    `critic_input = batch.global_state if batch.global_state is not None else batch.obs`  
    即正常情况是 **`env.state()` 提供的全局向量**，没有 agent one-hot。  
  - **MAPPO**：在 `CentralizedCriticMixin._prepare_batch_ctde` 里为  
    `critic_input = concat(global_state, one_hot(agent_index))`，维数 = **`state_dim + num_agents`**。

`train.py` 里对 `critic_input_dim` 的推断与之一致：

```109:114:/home/wzh/MAP-imitation-framework/train.py
    if algo_name in ("mappo", "vdppo"):
        critic_input_dim = state_dim + num_agents
    else:
        critic_input_dim = state_dim
```

以 **MASUP** 为例，`state()` 大致是：所有 agent 的 `(last_pos, target, nominal_remaining)` + 全图加权 idleness + `worst_idleness_fromT` + `obs_timer`（见 `masup.py` 的 `state()`）。

---

## 3. 与 MAPPO 的主要区别（在「实例数相同」的前提下）

| 项目 | IPPO | MAPPO |
|------|------|--------|
| Actor 实例（当前实现） | 1 个共享 | 1 个共享 |
| Critic 输入 | `global_state`（`state_dim`） | `global_state ∥ one-hot(agent)`（`state_dim + N`） |
| GAE / batch | 按 agent 保留 `Dict`，各自算 value | CTDE：拼 one-hot 后合并成单一大 batch |
| 优化器 | **一个** Adam：policy + critic | **两个**：actor_lr / critic_lr 可分开 |
| 其它 | `IPPOAlgo` 里 per-agent 累加 loss 再 step | `MAPPOAlgo` + `CentralizedCriticMixin`，trunc bootstrap 等带 one-hot |

---

**一句话**：在你这份代码里，从头训 IPPO 时并不是「N 个独立 Actor」；和 MAPPO 一样都是 **共享 Actor + 共享 Critic**，差别主要在 **Critic 是否拼接 agent one-hot**、**batch/GAE 是否 CTDE** 以及 **单/双优化器**。若你要严格论文版「每 agent 独立 Actor」，需要改 `train.py` 里 actor 分支的 `shared` 逻辑（当前为 Ask/说明模式，未改代码）。

---

**User**

我感觉这个倒是还好，IPPO在实践的时候为了优化效率一般也会使用网络参数共享，但是问题在于IPPO的actor和critic网络都应该只能输入每个智能体自己的obs, critic不能输入全局state, 你觉得这个理解是否正确？

---

**Cursor**

你的理解对应的是**最严格、也最符合「独立学习 + 部署时完全去中心化」**那一版 IPPO 定义，在这套定义下是**正确**的。

---

### 严格意义上的 IPPO（常见教材/论文表述）

- **Actor**：\(\pi_i(a_i \mid o_i)\)，只用智能体 \(i\) 自己的观测。  
- **Critic**：\(V_i(o_i)\)（或 \(Q_i(o_i,a_i)\)），同样**只**用 \(o_i\)，**不**用联合状态 / 全局 `state()`。

这样训练和执行时信息集一致：没有「训练时看全局、执行时只看局部」的 CTDE 结构；其它智能体的影响全部当作环境非平稳性，由独立 PPO 去扛。

所以：**「Actor 和 Critic 都只能吃自己的 obs，Critic 不能吃全局 state」**——在**严格 IPPO** 下，这个说法是对的。

---

### 实践里常见的「折中」

很多人仍叫 IPPO，但会：

- **Actor**：局部 obs（或再加参数共享）；  
- **Critic**：训练阶段用 **centralized critic**（全局 state 或 joint obs），为了减小方差、更好做 credit assignment。

这在文献里有时被叫作 IPPO + centralized critic、或和 MAPPO 的界限会变糊（MAPPO 更强调参数共享 + 常配合 CTDE；IPPO 强调「各 agent 一条 PPO 轨迹」）。所以：**不是**「名字叫 IPPO 就绝对不能给 critic 全局信息」，而是**严格定义**下你那句话成立；**工程上**用全局 critic 很常见。

---

### 和你当前仓库的关系

你们现在的 IPPO：`prepare_batch` 里 critic 用的是 `global_state`（`env.state()`），**不是**严格「仅局部 obs」的 critic。若你要对齐**你描述的那种严格 IPPO**，需要把 critic 输入改成与 actor 同源的 `obs`（并相应改 `critic_input_dim`、采集与 GAE 里喂给 critic 的张量）。这是设计取舍问题，不是谁对谁错的绝对判断。

**简短结论**：在**理论上的纯去中心化 IPPO**里，你的理解正确；在**工程上的 IPPO 变体**里，critic 用全局 state 也常见，但那就已经偏离「critic 也只能看自己的 obs」这一条了。

---

**User**

我想修改ippo的critic输入，具体需要改哪些地方？给出详细计划，需要改动很多细节吗？

---

**Cursor**

现在已经掌握全貌了。计划如下：

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

全部通过。

全部改动完成，总结如下：

**改动了 4 个文件：**

**`train.py`** — `_infer_dims_parallel` 加入 `ippo` 分支，`critic_input_dim = obs_dim`（而不是 `state_dim`），使 critic 网络以正确的维度初始化。

**`data/batch.py`** — `RolloutBatch` 新增 `final_obs: list = None` 字段，`chunk_split` 里在跳过 `final_global_state` 的同一处也跳过 `final_obs`。

**`data/collector.py`** — `_reset_buffer` 加入 `final_obs` key；`collect` 循环中，每次 step 之后按 agent 单独检查 `trunc[agent][i]`，截断时将 `next_obs[agent][i].copy()` 填入 `final_obs_list`，否则填 `None`；batch 打包时赋值到 `RolloutBatch.final_obs`。

**`algorithms/marl/ippo.py`** — `prepare_batch` 改为 `critic_input = batch.obs`，`_compute_trunc_bootstrap` 改传 `batch.final_obs`（per-agent，而非全局 state）；`update` 里 minibatch 的 value loss 计算改为 `critic_input = mb.obs`。

MAPPO 的所有路径完全不受影响，`global_state` 的采集依然持续进行。

---

**User**

我想知道这样修改完之后，当我想要用预训练模型初始化ippo算法训练时，在构建监督学习网络结构时，怎么知道input-dim的大小呢？目前的pipeline是否还是仅针对mappo？

---

**Cursor**

以下是对你问题的完整回答：

---

## 结论：目前模仿学习 pipeline **完全针对 MAPPO**，IPPO 用不了

### 1. 问题在哪里

`_infer_dims_from_env`（`imitation_trainer.py` 第 50 行）硬编码了：

```python
critic_state_dim = state_dim + num_agents   # MAPPO 式 one-hot 拼接
```

不论调用时你用的是 IPPO 还是 MAPPO，这里都会给 critic 建一个 `input_dim = state_dim + N` 的 `CriticMLP`。

---

### 2. 数据集本身也是 MAPPO 式的

`heuristic_sampler.py` 采集数据时（第 256-261 行）：

```python
g_state = env.state()          # [state_dim]
one_hot = zeros(num_agents)
one_hot[agent_id] = 1.0
critic_states.append(concat([g_state, one_hot]))  # shape: [state_dim + N]
```

HDF5 里存的 `critic_states` 字段维度是 `[N, T, M, state_dim+M]`，**完全是 MAPPO 格式**。IPPO 的 obs-based critic 需要的是 `[N, T, M, obs_dim]`，与此不兼容。

---

### 3. 如果想让 IPPO 也支持模仿学习初始化，需要改哪里

| 位置 | 改动 |
|---|---|
| `heuristic_sampler.py` `_collect_single_episode` / `_build_critic_states` | 不再拼 one-hot，直接存 per-agent `obs` 作为 critic input（或同时存两份） |
| `imitation_trainer.py` `_infer_dims_from_env` | 根据 algo 类型选择 `critic_state_dim = obs_dim`（IPPO）还是 `state_dim + N`（MAPPO） |
| `imitation_trainer.py` `load_imitator_config` | YAML 里加 `algo_name` 字段，传给 `_infer_dims_from_env` |
| `imitation_trainer.py` `_train_from_hdf5` / `_train_from_npz` | `critic_states_batch` 改从 `obs` 字段读（若统一存到 `obs`）或新字段 `critic_obs` |

目前如果你直接把 MAPPO 的 imitator 产出的模型（`critic.pt`）喂给 IPPO 的 `train.py`，critic 的 `input_dim` 是 `state_dim + N`，而 IPPO 现在期望的是 `obs_dim`，**加载时会因为维度不匹配直接报错**。只有 actor 部分（`policy.pt`）是兼容的——因为两者 actor 输入都是 per-agent obs。

---

**User**

是的，应该加上一个algo_name参数然后在内部根据这个参数进行分类；不过我想再讨论一下，首先对于iql,如果想搞模仿学习初始化应该也是好做的，就是先用监督学习拟合值网络，但如果对于vdppo/qmix/vdn这类涉及mixer网络的算法，是不是这个mixer层难以预训练？总之如果我的目标是6个算法全都能用这套模仿学习预训练的框架，你觉得如何修改最为简洁清晰且明智

---

**Cursor**

先看几个相关文件再给出建议。

现在全貌清晰了。下面是综合分析：

---

## 六个算法的预训练需求对比

| 算法 | 网络组成 | Actor BC 来源 | Critic/Q 监督目标 | Critic/Q 输入 | Mixer 预训练 |
|------|---------|-------------|-----------------|-------------|-------------|
| MAPPO | Actor + Critic | obs → action | returns | `global_state + one-hot` | 无 |
| IPPO | Actor + Critic | obs → action | returns | `obs` | 无 |
| VDPPO | Actor + Q-net + QPLEXMixer | obs → action | TD Q-targets | `global_state + one-hot` | ❌ 不可行 |
| IQL | Q-net | obs → argmax Q = action | returns | `obs` | 无 |
| VDN | Q-net + VDN加法Mixer | obs → argmax Q = action | returns | `obs` | ✅ 无参数，随机初始化即可 |
| QMIX | Q-net + QMIX超网络Mixer | obs → argmax Q = action | returns | `obs` | ❌ 不可行 |

**三点关键结论：**

**1. Mixer 不能/不需要预训练：**
- VDN Mixer 没有可学习参数（直接求和），无需预训练
- QMIX/VDPPO Mixer 依赖运行时 Q-network 输出，在 BC 阶段还没有联合的 Q-distribution 可以对齐，强行拟合没有意义，直接随机初始化是标准做法

**2. 所有算法只需预训练两类网络，数据也统一：**
- **Base Network**（Actor 或 Q-net）：都用 obs → action 做 BC，接口完全一致
- **Value Network**（Critic 或 Q-net 同一张网络）：用 returns 做 MSE，输入维度按算法类型分路

**3. VDPPO 的 Q-net 输入是 `global_state + one-hot`**（见 `train.py` 第 536 行），与 MAPPO Critic 相同，可以复用同一套逻辑

---

## 推荐的修改方案——最简洁的分类

只需在数据和配置层面做一次分类，**不需要改 heuristic_sampler 的采集逻辑**，因为数据集中同时存了 `obs` 和 `critic_states`（`global_state+one-hot`），两者都有：

### 分类规则（算法名 → 预训练策略）

```python
# imitation_trainer.py 中新增的路由函数
def _get_pretraining_config(algo_name: str, dims: dict) -> dict:
    """
    返回 (network_type, critic_input_key, critic_input_dim)

    network_type:
      "actor"  → 预训练 actor（行为克隆），critic 用 value MSE
      "value"  → 预训练 q-network（行为克隆取 argmax），直接用 q-net 做 value MSE

    critic_input_key:
      "obs"           → 用 batch['obs'] 作为 value network 输入
      "critic_states" → 用 batch['critic_states']（global_state+one-hot）作为输入
    """
    if algo_name in ("mappo", "vdppo"):
        return {"network_type": "actor",  "value_input_key": "critic_states",
                "value_input_dim": dims["state_dim"] + dims["num_agents"]}
    elif algo_name == "ippo":
        return {"network_type": "actor",  "value_input_key": "obs",
                "value_input_dim": dims["obs_dim"]}
    elif algo_name in ("iql", "vdn", "qmix"):
        return {"network_type": "value",  "value_input_key": "obs",
                "value_input_dim": dims["obs_dim"]}
    else:
        raise ValueError(f"Unsupported algo_name for pretraining: {algo_name}")
```

### 需要改动的地方（共 3 处）

**1. `_infer_dims_from_env`** — 不再硬编码，改为返回所有维度，让调用方按 `algo_name` 决策：

```python
return {
    "obs_dim": obs_dim,
    "action_dim": action_dim,
    "state_dim": state_dim,
    "num_agents": num_agents,
    # 不再返回 critic_state_dim，由上层 _get_pretraining_config 决定
}
```

**2. `load_imitator_config`** — 加 `algo_name` 参数，构建 `base_network`（actor 或 Q-net）和 `value_network`（critic 或 Q-net 复用），调用 `_get_pretraining_config` 分路：

```yaml
# YAML 加一个字段
algo_name: ippo    # mappo / ippo / vdppo / iql / vdn / qmix
```

**3. `imi_trainer.train`** — 训练循环内部，根据 `network_type` 决定：
- `actor` 模式：行为克隆 loss = cross_entropy(actor(obs), action)，value loss = MSE(critic(value_input), returns)
- `value` 模式：行为克隆 loss = cross_entropy(q_net(obs)选最大动作, action)，value loss = MSE(q_net(obs)中目标动作的Q值, returns)

> 注意 `value` 模式下 actor 和 value 实际上是同一张网络，可以合并为一个 MSE+CE 联合 loss

### 数据集无需改动

`heuristic_sampler` 里 `critic_states` 字段（`global_state+one-hot`）和 `obs` 字段已经同时存在，IPPO/IQL/VDN/QMIX 直接用 `obs` 那一列即可，不需要重新采集数据。

---

**总结：** 核心改动只在 `imitation_trainer.py` 里加一个 `algo_name` 参数和一个路由函数，通过 `value_input_key` 在 `obs` 和 `critic_states` 间切换，对 Mixer 类网络直接跳过预训练。数据层和采集器不需要动，六个算法都能覆盖。需要我来实现吗？

---

**User**

actor 模式和 value 模式指的是算法为policy-based还是value-based对吗？前者是PPO系列，后者是qmix/vdn/iql

---

**Cursor**

对，就是这个意思，与代码里已有的 `ALGO_REGISTRY` 的 `policy_type` 字段完全一致：

```60:109:/home/wzh/MAP-imitation-framework/configs/registry.py
    "mappo": { ..., "policy_type": "actor" },
    "ippo":  { ..., "policy_type": "actor" },
    "vdppo": { ..., "policy_type": "actor" },
    "iql":   { ..., "policy_type": "value" },
    "vdn":   { ..., "policy_type": "value" },
    "qmix":  { ..., "policy_type": "value" },
```

所以 `_get_pretraining_config` 里甚至不需要手写 `if algo_name in (...)` 枚举，直接查 registry 就行：

```python
from configs.registry import ALGO_REGISTRY, get_policy_type

policy_type = get_policy_type(algo_name)   # "actor" 或 "value"

if policy_type == "actor":
    if algo_name in ("mappo", "vdppo"):
        value_input_key, value_input_dim = "critic_states", state_dim + num_agents
    else:  # ippo
        value_input_key, value_input_dim = "obs", obs_dim
else:  # value-based: iql / vdn / qmix
    value_input_key, value_input_dim = "obs", obs_dim
```

两类行为的本质区别：

- **actor 模式**：actor 和 value network 是**两张独立的网络**，BC 只训 actor，value 单独用 MSE 训 critic；两张网分别存 `policy.pt` / `critic.pt`
- **value 模式**：Q-network **既是策略（取 argmax 动作）也是价值函数**，BC loss（cross-entropy，让正确动作 Q 最大）+ value MSE loss 可以作用在**同一张网络**上，联合训练；只存一个 `policy.pt`（即 Q-net 权重），Mixer 不存也不载入

---

**User**

VDPPO 的 Q-net 其实预测的还是输入obs 预测state-value对吗？那么是不是也可以和mappo的critic网络一样使用采样时一个episode之后逆向运算的真是return来监督学习？我还有一个疑问是对于value-based算法，既要依据Q网络做决策，又要预测准确的action-value值，监督学习时用什么当label呢？

---

**Cursor**

两个问题都分析清楚了。

---

## 问题一：VDPPO 的 Q-net 输入是什么，能否用 return 监督？

**输入是 `global_state + one-hot`，不是 obs**。代码第 178 行：
```python
q_in = torch.cat([global_state, one_hot], dim=-1)  # (state_dim + N)
Q_i = self.q_network(q_in)  # → (action_dim)
```

它输出的是 **action-value Q(s, ·)**（每个动作一个值），不是 state-value V(s)。`V_i = Q_i.max()` 是派生出来的。

**不能直接用 return 做监督**，原因有两点：
1. VDPPO 的 TD target 是 `y_tot = r + γ * Q_tot_next`，需要 mixer 参与（第 231 行），预训练阶段 mixer 随机初始化，y_tot 完全是噪声
2. 即使只预训练单 agent 的 Q-net，target 应该是 per-agent Q 值，而 heuristic 采集的 `returns` 是折扣累积回报 R（对应 V），用它监督 Q 会把 Q 训成 V，训练启动后 PPO actor 的 advantage 估计倒还凑合，但 Q-update 阶段会出现偏差

**VDPPO 预训练的合理做法**：只预训练 actor（行为克隆），**跳过 Q-net 预训练**，随机初始化 Q-net 和 mixer，让 RL 阶段从头学值函数。这也是最简洁的选择。

---

## 问题二：value-based 算法预训练时用什么当 label？

这里有一个**根本性的歧义**需要区分：

**监督学习目标有两个，来自不同的"任务"：**

| 任务 | 输入 | 输出 | Label 来源 | Loss |
|------|------|------|----------|------|
| 策略克隆（BC） | obs | Q(obs, a) for all a | 启发式策略的**动作** | cross-entropy：让正确动作对应的 Q 最大（argmax = expert action）|
| 值函数预训练 | obs | Q(obs, a_taken) | 蒙特卡洛**return** R | MSE：Q(obs, a) ≈ R |

关键在于，**BC 和值函数预训练对同一个网络是相互矛盾的**：
- BC 的目标是让 Q(s, a_expert) 比其他动作大（只关心相对大小）
- 值预训练的目标是让 Q(s, a_taken) 绝对值接近 R（关心绝对大小）

**实践上最合理的做法是联合 loss：**

```python
# Q-net 输出所有动作的 Q 值: [batch, action_dim]
q_all = q_network(obs_batch)

# 1. BC loss: 让专家动作的 Q 最大 → cross-entropy
bc_loss = cross_entropy(q_all, expert_action)

# 2. Value loss: 让专家动作的 Q 接近真实 return
q_taken = q_all.gather(1, expert_action.unsqueeze(1)).squeeze(1)
value_loss = mse_loss(q_taken, returns_batch)

total_loss = bc_loss + lambda * value_loss
```

这样 BC loss 约束了**动作排序**，value loss 约束了**绝对量级**，两者互补。实验上 `lambda=0.5` 附近通常效果不错。

**对于 IQL 和 VDN**（Q-net 输入是 obs，没有 mixer 干扰），这套联合 loss 完全可行，是合理的预训练方式。

