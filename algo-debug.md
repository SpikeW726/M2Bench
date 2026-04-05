# Normalization mechanism in reinforcement learning
_Exported on 2026/4/5 at GMT+8 15:24:49 from Cursor (2.6.12)_

---

**User**

使用qmix训练的loss奇大无比，这是什么原因？我记得我之前在实现mappo+模仿学习pipeline的时候有对critic网络的输出做归一化，但我有点不记得了，能不能详细讲解一遍从启发式采样到后续rl训练全流程中这个归一化机制如何运作，以及有没有可能在mappo和qmix等其他算法的from-scratch直接训练时也用上
[MAX_LATENCY]: 40.0000
[WAIT_ACTIONS]: 59/306 (19.28%)
[Iter 16590/26042] steps=12741120, reward=-27.66, loss=44865462688.0000, SPS=1904

---

**Cursor**

正在快速查看 superpowers 技能指引，并在你的代码库中搜索与 critic/value 归一化相关的实现。



下面分三块说明：**QMIX 里 loss 巨大可能原因**、**你这套 MAPPO+模仿里 value 归一化怎么串起来**、**QMIX/从零 RL 里能不能用类似机制**。

---

## 1. QMIX 的 loss 为什么会到 \(10^{10}\) 量级？

在你仓库里，QMIX（继承 `VDNAlgo`）的 TD 目标是完全**原始奖励尺度**上的 Bellman 目标：

```162:166:/home/wzh/MAP-imitation-framework/algorithms/marl/vdn.py
            q_tot_target = self.target_mixer(q_next_chosen, next_state)
            r_tot = rew_all.sum(dim=-1)  # (B,) — 联合奖励 = 各 agent 奖励之和
            td_target = r_tot + self.gamma * (1.0 - done) * q_tot_target

        loss = F.smooth_l1_loss(q_tot, td_target)
```

因此 **`loss` 不是“有没有学到”的无量纲指标**，而是直接反映 \(|Q_{\mathrm{tot}} - y|\) 的量级。出现 \(4.48\times 10^{10}\) 通常意味着下面几类情况（可并存）：

1. **奖励 / 回报尺度过大**  
   环境若每步奖励绝对值大、或折扣累加后 \(Q\) 应在 \(10^3\sim10^5\) 量级，而网络仍接近初始化（或发散），Huber 在误差很大时近似线性，**标量 loss 会非常大**。你日志里 `reward=-27.66` 若是某种滑动平均，不矛盾——单步或序列上的 `r_tot` 仍可能和当前 \(Q\) 严重不匹配。

2. **多智能体 `rew_all.sum(dim=-1)`**  
   若每个 agent 的 `rew` 都是同一份全局奖励的拷贝，**求和会把奖励放大 \(N\) 倍**，TD 目标整体放大，loss 同量级放大。需要确认采集端每个 agent 的 `rew` 是「各 agent 局部奖励」还是「重复的全局奖励」。

3. **训练已发散或数值不稳**  
   学习率过大、梯度爆炸、`q_tot` 与 `td_target` 一方失控，都会让 Huber loss 飙高。QMIX 对 mixer、state 质量也比较敏感。

4. **和 MAPPO 的对比**  
   MAPPO 在 `use_value_norm=True` 时，**critic 的回归目标在归一化空间里**，value loss 的典型量级会小很多；QMIX **没有**走这条路径，所以**不能**和 MAPPO 的 loss 数值直接对比。

若要排查，建议在 `TrainingStats` 里已有 `q_tot_mean`、`td_error` 基础上，再看 `td_target`、\(r_{\mathrm{tot}}\) 的均值/最大绝对值，并核对 `rew` 是否在多智能体上被重复加总。

---

## 2. 从启发式采样到 RL：你项目里的「critic / return 归一化」全流程

这套逻辑和 **「在原始回报尺度上做 GAE，但让 critic 拟合归一化后的 return」** 一致，并与模仿学习阶段统计量衔接。

### 阶段 A：模仿学习（`ImitationTrainer`）

- 从 HDF5 等读出 **蒙特卡洛回报** `returns`。
- 若 `use_value_norm`：在全数据上算  
  \(\mu = \mathbb{E}[\text{return}],\ \sigma = \mathrm{std}(\text{return})\)。
- Critic 优化的是 **归一化目标**：  
  \(\text{target\_norm} = (\text{return} - \mu) / (\sigma + \epsilon)\)，与 `critic_pred` 做 MSE（见 `imitation_trainer.py` 中 `target_norm` 与 `critic_loss`）。
- 检查点里写入 `value_normalization`（`ret_mean` / `ret_std` 等），供 RL 阶段初始化 `ret_rms`。

### 阶段 B：RL（以 MAPPO/IPPO + `ctde_mixin` 为例）

1. **Critic 前向**得到的是 **归一化空间**的值，记为 `values_norm`。
2. **算 GAE 时要用「真实尺度」的 value**（与 `reward` 同一量纲）：  
   \(V_{\text{real}} = V_{\text{norm}} \cdot \sigma + \mu\)（`prepare_batch` / `_prepare_batch_ctde` 里对 `values_norm` 做反归一化）。
3. 用 \(r\) 与 \(V_{\text{real}}\) 做 GAE，得到 **`ret` 仍在真实回报尺度**。
4. **更新 RunningMeanStd**：对当前 batch 的 `ret`（或 active 步上的 `ret`）做 `ret_rms.update`，使 \(\mu,\sigma\) 随训练漂移（若从模仿加载了初值，则是在此基础上的在线修正）。
5. **Critic 的 value loss**（如 `ippo.py`）：  
   \(\text{target} = (\text{ret} - \mu_{\text{rms}}) / (\sigma_{\text{rms}} + \epsilon)\)，与 **网络输出的 `new_value`（归一化空间）** 对齐。

要点：**Bootstrap、GAE、advantage 在「真实世界」奖励尺度上算；只有 critic 的监督信号被归一化**，这样梯度尺度更稳，又不改变策略梯度里 advantage 的语义（advantage 仍是真实尺度上的 TD/GAE 残差）。

---

## 3. MAPPO / QMIX「从零训练」能不能用类似机制？

可以，但要**按算法改对接方式**，不能照搬一行配置开关。

| 思路 | MAPPO 类（你已有） | QMIX / DQN 类 |
|------|-------------------|----------------|
| **Return / value 归一化（你现在的做法）** | critic 输出 norm，GAE 用反归一化 \(V\) | 更常见的是 **只缩放奖励** \(r' = r / \sigma_r\)，令 \(Q\) 与 target 同尺度；或 **PopART**（输出层自适应仿射，保证内部预测归一化）。 |
| **Reward 归一化** | 可选，与 value norm 二选一或组合需谨慎 | 很实用：减均值/除标准差（或 RMS），目标与 \(Q\) 一起缩小，Huber loss 数值会下来。 |
| **Obs 归一化** | 通用 | 同样通用，和 loss 大小间接相关。 |

对 **QMIX from scratch** 较务实的做法是：

1. **先确认 `r_tot` 定义**（是否重复加全局奖励）。  
2. **奖励缩放**：例如 \(r \leftarrow r / \max(1, \mathrm{RMS}(r))\) 或固定除以一个与环境相关的常数，使 TD target 落在合理数量级。  
3. 若希望和 MAPPO **同一套哲学**：可对 **TD 目标**做与 running \(\mu,\sigma\) 一致的变换，并令 mixer 前的各 agent \(Q_i\) 预测归一化值，再在 loss 里对齐——实现量会比 MAPPO 大，因为要动 target 网络与 possibly PopArt。

**结论**：你记得的「对 critic 输出做归一化」在这份代码里，本质是 **critic 拟合归一化 return + GAE 用反归一化 value + `ret_rms` 在线更新**；QMIX 当前实现**没有**这条链路，所以 loss 数值会比开了 `use_value_norm` 的 MAPPO 大很多属正常现象；若已到 \(10^{10}\) 且 `td_error` 同步爆炸，应优先查 **奖励定义/缩放** 与 **是否发散**，再考虑给 off-policy 分支加 **reward scaling** 或 **PopART** 类目标归一化。

---

**User**

能不能把latex数学公式都好好渲染一遍重新输出，现在全是乱码根本无法阅读

---

**Cursor**

下面用常见 Markdown 支持的写法重新整理（行内用 `$...$`，独立公式用 `$$...$$`），避免 `\(` 等在部分客户端里显示成乱码。

---

## 1. QMIX 的 loss 为什么会到 $10^{10}$ 量级？

QMIX（继承 `VDNAlgo`）在**原始奖励尺度**上做 Bellman：

$$
y = r_{\mathrm{tot}} + \gamma (1 - d)\, Q_{\mathrm{tot}}'(s', a')
$$

$$
\mathcal{L} = \mathrm{SmoothL1}\bigl(Q_{\mathrm{tot}}(s,a),\, y\bigr)
$$

因此 **loss 的数值本身不是“好坏”的无量纲指标**，而是直接跟 $|Q_{\mathrm{tot}} - y|$ 的量级挂钩。出现约 $4.48 \times 10^{10}$ 常见原因包括：

1. **奖励 / $Q$ 尺度过大**  
   若真实应学到的 $|Q|$ 很大，而网络仍接近初始化或已发散，Huber 在大误差下近似**线性**，平均 loss 会极大。

2. **多智能体 `rew_all.sum(dim=-1)`**  
   若每个 agent 的 `rew` 都是**同一份全局奖**的拷贝，求和会把 $r_{\mathrm{tot}}$ **放大 $N$ 倍**，TD 目标与 loss 同量级变大。

3. **训练发散**  
   学习率过大、梯度爆炸、mixer/state 问题等，都会使 $Q_{\mathrm{tot}}$ 与 $y$ 严重错位。

4. **和 MAPPO 对比**  
   MAPPO 在 `use_value_norm` 时，**critic 回归在归一化目标上**，value 相关 loss 数值通常小得多；QMIX **没有**这条路径，**不能**和 MAPPO 的 loss 数字直接比大小。

---

## 2. 启发式采样 → 模仿 → RL：value / return 归一化怎么串起来

核心思想：**GAE / bootstrap 仍在「真实奖励尺度」上算；只有 critic 的监督信号用归一化后的 return。**

### 阶段 A：模仿学习

- 数据里是蒙特卡洛回报，记为 $G$。
- 在全数据上估计：
$$
\mu = \mathbb{E}[G],\quad \sigma = \mathrm{std}(G)
$$
- Critic 拟合归一化目标：
$$
G_{\mathrm{norm}} = \frac{G - \mu}{\sigma + \varepsilon}
$$
- 用 MSE：$\mathcal{L}_V = \bigl(V_\theta - G_{\mathrm{norm}}\bigr)^2$（实现里即 `target_norm` 与 `critic_pred`）。
- 把 $\mu,\sigma$（及开关）存进 checkpoint，供 RL 初始化 `ret_rms`。

### 阶段 B：MAPPO / IPPO（CTDE）

1. Critic 输出**归一化空间**的值 $V_{\mathrm{norm}}$。
2. 算 GAE 时要与 $r$ **同量纲**，先做**反归一化**：
$$
V_{\mathrm{real}} = V_{\mathrm{norm}} \cdot \sigma + \mu
$$
3. 用 $r$ 与 $V_{\mathrm{real}}$ 做 GAE，得到的 **return $R$（或 `ret`）仍在真实尺度**。
4. 用当前 batch 的 $R$ 更新 RunningMeanStd（`ret_rms.update`），得到随时间更新的 $\mu_t,\sigma_t$。
5. Value loss 里，**目标**是归一化后的 return：
$$
R_{\mathrm{norm}} = \frac{R - \mu_t}{\sigma_t + \varepsilon}
$$
与网络输出的 $V_{\mathrm{norm}}$ 对齐。

**小结：** $\delta$、GAE、$V(s')$ bootstrap 用 $V_{\mathrm{real}}$；**critic 只学** $R_{\mathrm{norm}}$。这样既稳定梯度，又不把 advantage 的“真实尺度”语义搞错。

---

## 3. MAPPO / QMIX 从零训练能否用类似机制？

可以，但要按算法改接口，不能指望只开一个开关。

| 思路 | On-policy（MAPPO 等，你仓库已有） | Off-policy（QMIX 等） |
|------|-----------------------------------|---------------------|
| Return / value 归一化 | Critic 输出 norm + GAE 用反归一化 $V$ | 更常见：**奖励缩放** $r' = r / s$，或 **PopART**（输出端自适应仿射） |
| 仅缩放奖励 | 可与 value norm 组合时需小心一致性 | 很实用，能直接压低 $|y|$ 与 Huber loss 数值 |
| 观测归一化 | 通用 | 通用，间接稳定训练 |

对 **QMIX from scratch** 较务实的顺序：

1. 确认 $r_{\mathrm{tot}}$ 是否被**重复加全局奖励**。  
2. 做 **reward scaling**（例如除以 running RMS 或环境相关常数）。  
3. 若要和 MAPPO **同一哲学**：需对 TD 目标与 $Q$ 预测做一致的仿射归一化（接近 PopART 或显式 normalized target），实现量比 on-policy 那条路径大。

---

若你当前客户端仍不渲染 `$...$`，可在设置里打开 Markdown 数学公式（KaTeX/MathJax），或把本文复制到支持 LaTeX 的编辑器里查看。

---

**User**

1. 现在控制是否使用use_value_norm的开关在哪？我怎么感觉没有在任何yaml文件中看到配置？
2. @mappo_masup_longedge_scratch.yaml (24-27) masup环境有reward_scale机制，在使用qmix训练时难道不起作用吗？
3. 我们现在先聚焦于qmix, 你说的重复求和全局奖励很有道理，这里是否需要增加一个参数进行分类控制？比如像masup这种所有智能体获得的奖励相等的环境设计，其实就可以把这个当成全局奖励，而其他各个智能体获得奖励不同的环境，就应该正常计算q_tot，你认为需要这样处理吗？

---

**Cursor**

正在在代码库中查找 `use_value_norm` 的配置来源及 `reward_scale` 在 QMIX 训练路径中是否生效。



下面是基于当前仓库实现的直接结论与设计建议。

---

## 1. `use_value_norm` 开关在哪？为什么实验 YAML 里少见？

- **定义位置**：在 `OnPolicyParams` 里，默认是关掉的：

```44:45:/home/wzh/MAP-imitation-framework/configs/algo_configs.py
    use_value_norm: bool = False        # return 归一化 (RunningMeanStd)
```

`MAPPOParams` 继承这条，所以 **MAPPO / IPPO 等**都走同一字段。

- **实验 YAML**：你贴的 `mappo_masup_longedge_scratch.yaml` 里 **`algo:` 段没有写 `use_value_norm`**，因此会用默认值 **`False`**，等价于不用 return 归一化。

- **显式出现在哪些 YAML**：目前仓库里**只有模仿学习配置**里写了，例如：

  - `configs/imitator/masup_gnn_tsp12.yaml`：`use_value_norm: true`
  - `configs/imitator/masup_mlp_grid.yaml`：`use_value_norm: true`

  主训练实验 YAML（`configs/experiments/...`）里普遍**没写**这一项。

- **和预训练的关系**：`train.py` 在加载 actor/critic 时会读 checkpoint 旁的 `config.yaml` 里的 `value_normalization`，并作为 `value_norm_config` 传给算法。但 **`AlgorithmBase` 里只有 `params.use_value_norm == True` 时才会创建 `ret_rms` 并应用归一化**；若实验 YAML 里仍是默认 `False`，**光有 `value_norm_config` 也不会启用**（`ret_rms` 根本不会被创建）。  
  因此：**要在 MAPPO 上用 value norm，应在实验配置的 `algo:` 下显式写 `use_value_norm: true`**（并与模仿学习统计量/预训练衔接）。

**QMIX**：`QMIXParams` 继承的是 off-policy 分支，**没有** `use_value_norm` 这套字段；VDN/QMIX 损失里也不走 `ret_rms`。

---

## 2. `reward_scale` 在 QMIX 里会不会生效？

**会。** `reward_scale` 是 **MASUP 环境**在 `__init__` 里从 `kwargs` 读的，并在算奖励时乘上去：

```34:34:/home/wzh/MAP-imitation-framework/envs/mdps/masup.py
        self.reward_scale = float(kwargs.get('reward_scale', 1.0))
```

```507:517:/home/wzh/MAP-imitation-framework/envs/mdps/masup.py
    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        ...
            rewards[agent_str] = reward * self.reward_scale
```

算法（MAPPO / QMIX）只消费环境返回的标量 `rew`；只要实验里 `env.custom_configs.reward_scale` 会传进 `MASUPEnv`，**QMIX 和 MAPPO 用的是同一套缩放后的奖励**。  
`configs/experiments/masup/qmix_masup_longedge.yaml` 里同样配置了 `reward_scale: 0.01`，与 MAPPO 实验一致。

若 QMIX 仍出现极大 loss，更可能来自 **`r_tot = rew_all.sum(dim=-1)` 在「各 agent 奖励相同」时的 N 倍放大**，而不是 `reward_scale` 没生效。

---

## 3. QMIX：要不要加参数区分「全局奖励」vs「各 agent 不同奖励」？

**从建模上**：若环境本质是 **team reward**（每个 agent 的 `rew` 都是同一个全局量），VDN/QMIX 标准写法里 **团队回报应对齐一次该标量**，而不是对 N 个拷贝再求和。当前实现是 **对每 agent 的 `rew` 求和**，在「人人相同」时等价于 **`N × r_global`**，Bellman 目标整体放大 N 倍，和「单次团队转移只发生一次团队奖励」的常见定义不一致。

**是否值得加参数**：**值得**，但语义要定清楚，例如：

- **`team_reward` / `use_global_reward`**：为 `true` 时 `r_tot = rew_all[:, 0]`（或 `mean(dim=-1)`，在完全相等时与取第一个等价）；为 `false` 时保持 `sum`（真正的 per-agent 奖励可分解场景）。
- 或者 **在环境层**只让「一个虚拟 team slot」拿团队奖、其余为 0，再在算法里统一 `sum`（改动环境协议，侵入性更大）。

**注意**：若未来真有 **per-agent 不同奖励** 且要符合「可分解 + mixer」的设定，应继续用 **求和**（或论文里对应的分解形式），这时 **不能**再强行用「只取一个 agent」。

**实践建议**：  
- MASUP 这类 **人人相同的全局 shaping**：用参数固定为「不重复加总」更干净，也避免靠 `reward_scale: 0.01` 去抵消一个额外的因子 N。  
- 通用 MARL 基准里 **各 agent `rew` 可能不同**：默认 **`sum` 保持现状**，由配置在「全局奖环境」上切换模式即可。

---

**小结**：  
1）`use_value_norm` 在 **`configs/algo_configs.py` 默认 False**；要在 MAPPO 实验里打开，需在 **`configs/experiments/...` 的 `algo:` 下写 `use_value_norm: true`**；模仿 YAML 里已有示例。  
2）**`reward_scale` 在环境里生效，QMIX 同样适用。**  
3）**为「全 agent 相同的全局奖励」增加开关（或用 mean/单路）是合理且与 QMIX 目标定义一致的**；与「per-agent 不同奖励」应用 `sum` 并不矛盾，用配置分支即可。

---

**User**

但实际上在masup中，如果某个step之后某个智能体在edge上，它的rew为0, 所以也不是始终各智能体奖励相等

---

**Cursor**

正在确认 MASUP 中奖励是否按 agent 状态（例如在边上为 0）分支计算。



你的说法里有一半和「当前仓库里的实现」一致，有一半需要拆开说。

## 1. MASUP 里奖励什么时候会「不一样」？

在 `masup.py` 里，**基础项** `_compute_rewards` 对每个 agent 用的是**同一个**全局量（`worst_idleness_fromT` 等），再乘 `reward_scale`，所以**这一段在代码上对所有 agent 是相同的**：

```507:517:/home/wzh/MAP-imitation-framework/envs/mdps/masup.py
    def _compute_rewards(self, result: TickResult) -> Dict[str, float]:
        ...
        for agent_str in self.agents:
            ...
            rewards[agent_str] = reward * self.reward_scale
```

**会拉开各 agent 差异的**，主要是后面的 **IDI** 和 **个体贡献奖励**（依赖 `idi_scale` / `contribution_scale` 以及每个 agent 自己的量）：

```189:212:/home/wzh/MAP-imitation-framework/envs/mdps/masup.py
        for i, agent_str in enumerate(self.agents):
            if self.idi_scale > 0 and i in idi_baselines:
                ...
                rewards[agent_str] += diff * self.idi_scale
            if self.contribution_scale > 0:
                contrib = self._agent_idleness_reduction.get(i, 0.0)
                if contrib > 1e-9:
                    rewards[agent_str] += contrib * self.contribution_scale
```

因此：**只要 IDI/贡献项非零，各智能体奖励就不必相等**；这和「是否在边上」没有直接写在同一段公式里，但**决策节奏、贡献不同**会导致向量不同。

若你观察到「在边上某步为 0」，需要区分是：**环境 `step` 返回的字典里该 agent 就是 0**，还是**日志/可视化里别的含义**；就当前文件而言，**没有**「`ON_EDGE` 则把该 agent 的 `rew` 置 0」的分支。

## 2. 和 QMIX 采集（MLP 同步路径）的关系

QMIX 用的 off-policy collector 在 **`_sync_mode`** 下会把「一次决策 → 下次可决策/结束」之间的**多步环境奖励**按 \(\gamma\) **折现累加**进**一条** transition 的 `rew`，而不是「边上每一步都存一条 rew=0」：

```880:883:/home/wzh/MAP-imitation-framework/data/collector.py
                pend = self._pending[agent][i]
                if pend is not None:
                    pend['acc_reward'] += pend['gamma_power'] * rew[agent][i]
                    pend['gamma_power'] *= gamma
```

所以：**回放进 QMIX 的标量 `rew`，往往是跨多物理步的累积**，和「某一物理步在边上 instant reward」不是同一粒度。

## 3. 对之前「重复加全局奖励」结论的修正

- 若**每一步、每个 agent 的 `rew` 都完全相同**，则 `r_tot = \sum_i r_i` 仍是 **`N` 倍**同标量，这是我之前强调的情形。  
- 若**各 agent 的 `rew` 本就不等**（例如 IDI/贡献不同），则 `r_tot` 表示的是**这些局部奖励之和**，语义上是「团队标量目标 = 各 agent 奖励之和」，**不再是简单的「N 份拷贝」**；是否和你要的「团队只计一次全局 shaping」一致，要看**任务定义**而不是只看「有时在边上」。

**结论**：在 MASUP 里，**不等价于始终全局复制**——尤其开了 IDI/贡献时；但**大块相同的基础项**若仍人人一份，`sum` 仍会把那部分放大约 `N` 倍。是否需要「全局奖只计一次」的开关，仍取决于你是否把 `worst_idleness` 那部分视为**团队单次信号**；与「边上 rew 是否为 0」可以分开设计、分别验证（看 `step` 返回的 `rew` 字典和 `r_tot` 定义是否一致）。

---

**User**

帮我配置一下vdppo/ippo/qmix/vdn/iql算法使用masup在SF地图上experiment/sweep/eval的参数文件

---

**Cursor**

正在查看现有的 MASUP + SF 地图配置及各类算法的实验模板，以便为 vdppo、ippo、qmix、vdn、iql 添加 experiment、sweep 与 eval 配置。



正在创建 5 个 SF 实验配置、5 个对应 sweep 配置，并修正 `masup_SF.yaml` 评估环境与 `mappo_masup_SF_scratch.yaml` 中重复的 `T` 键。



当前环境处于 **Ask 模式**，无法在仓库里直接新建/修改文件。请 **切换到 Agent 模式** 后让我再执行一次写入；或你自行把下面内容保存为对应路径。

下面按 **experiment → sweep → eval** 给出与 `mappo_masup_SF_scratch.yaml` 一致的 SF 环境块（`SFcrimemap.json`、`deltaT: 0.333`、`T: 200`、`reward_scale` 等），并修正原 `mappo_masup_SF_scratch.yaml` 里重复的 `T` 键（你本地可顺手改一行）。

---

### 1. Experiment：`configs/experiments/masup/`

**`ippo_masup_SF.yaml`**

```yaml
# IPPO + MASUPEnv (SF / SFcrimemap) — 从零训练
# 用法: python train.py configs/experiments/masup/ippo_masup_SF.yaml

algo_name: ippo
env_type: masup
actor_type: mlp
critic_type: mlp

exp_name: ippo-SF
graph_name: SF
track_wandb: true
wandb_project: MAP-RL

env:
  graph_path: graphs/SFcrimemap.json
  enable_wait: true
  deltaT: 0.333
  num_agents: 3
  init_positions: [0, 5, 11]
  episode_len: 3000
  custom_configs:
    T: 200.0
    role_imformation: decision
    reward_scale: 0.01
    truncate_by_time: true
    idi_scale: 0.05
    contribution_scale: 0.05

algo:
  use_active_mask: true
  lr: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coef: 0.5
  ent_coef: 0.01

training:
  num_envs: 16
  use_subproc: true
  num_steps: 2048
  total_steps: 20000000
  save_interval: 200
  minibatch_size: 4096
  update_epochs: 5

actor:
  hidden: [256, 256]
critic:
  hidden: [256, 256]
```

**`vdppo_masup_SF.yaml`**

```yaml
# VDPPO (MLP) + MASUPEnv (SF / SFcrimemap)
# 用法: python train.py configs/experiments/masup/vdppo_masup_SF.yaml

algo_name: vdppo
env_type: masup
actor_type: mlp
q_type: mlp

exp_name: vdppo-SF
graph_name: SF
track_wandb: true
wandb_project: MAP-RL

env:
  graph_path: graphs/SFcrimemap.json
  enable_wait: true
  deltaT: 0.333
  num_agents: 3
  init_positions: [0, 5, 11]
  episode_len: 3000
  custom_configs:
    T: 200.0
    role_imformation: decision
    reward_scale: 0.01
    truncate_by_time: true
    idi_scale: 0.05
    contribution_scale: 0.05

algo:
  use_active_mask: true
  actor_lr: 5.0e-4
  q_lr: 5.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coef: 0.5
  ent_coef: 0.01
  target_kl: 0.3
  tau: 0.1
  target_update_freq: 1
  q_clip_range: null
  mixer_embed_dim: 64
  data_chunk_length: 10

training:
  num_envs: 16
  use_subproc: true
  num_steps: 2048
  total_steps: 20000000
  save_interval: 200
  minibatch_size: 4096
  update_epochs: 5

actor:
  hidden: [256, 256]
q_network:
  hidden: [256, 256]
  dueling: true
```

**`qmix_masup_SF.yaml`**

```yaml
# QMIX + MASUPEnv (SF / SFcrimemap)
# 用法: python train.py configs/experiments/masup/qmix_masup_SF.yaml

algo_name: qmix
env_type: masup
q_type: mlp

exp_name: qmix-SF
graph_name: SF
track_wandb: true
wandb_project: MAP-RL

env:
  graph_path: graphs/SFcrimemap.json
  enable_wait: true
  deltaT: 0.333
  num_agents: 3
  init_positions: [0, 5, 11]
  episode_len: 3000
  custom_configs:
    T: 200.0
    role_imformation: decision
    reward_scale: 0.01
    truncate_by_time: true
    idi_scale: 0.05
    contribution_scale: 0.05

algo:
  lr: 1.0e-4
  gamma: 0.99
  tau: 1.0
  target_update_freq: 200
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.9995
  use_double_dqn: true
  shared_policy: true
  mixer_embed_dim: 32

training:
  num_envs: 16
  use_subproc: true
  num_steps: 64
  total_steps: 10000000
  save_interval: 500
  batch_size: 256
  warmup_steps: 5000
  buffer_size: 50000
  collect_per_step: 16
  update_per_step: 1

q_network:
  hidden: [256, 256]
  dueling: false
```

**`vdn_masup_SF.yaml`**：与 `qmix_masup_SF.yaml` 相同，仅将 `algo_name: vdn`、`exp_name: vdn-SF`，并去掉 `mixer_embed_dim`；`epsilon_decay: 0.995` 与 longedge VDN 一致。

**`iql_masup_SF.yaml`**：在 `vdn_masup_SF` 基础上改为 `algo_name: iql`、`exp_name: iql-SF`，并设：

- `sync_replay: true`
- `shared_policy: false`
- `tau: 0.005`、`target_update_freq: 1`（与 `IQLParams` 默认 soft-update 风格一致）

完整文件：

```yaml
# IQL + MASUPEnv (SF) — 每 agent 独立 Q-network
# 用法: python train.py configs/experiments/masup/iql_masup_SF.yaml

algo_name: iql
env_type: masup
q_type: mlp

exp_name: iql-SF
graph_name: SF
track_wandb: true
wandb_project: MAP-RL

env:
  graph_path: graphs/SFcrimemap.json
  enable_wait: true
  deltaT: 0.333
  num_agents: 3
  init_positions: [0, 5, 11]
  episode_len: 3000
  custom_configs:
    T: 200.0
    role_imformation: decision
    reward_scale: 0.01
    truncate_by_time: true
    idi_scale: 0.05
    contribution_scale: 0.05

algo:
  sync_replay: true
  lr: 1.0e-4
  gamma: 0.99
  tau: 0.005
  target_update_freq: 1
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.9995
  use_double_dqn: true
  shared_policy: false

training:
  num_envs: 16
  use_subproc: true
  num_steps: 64
  total_steps: 10000000
  save_interval: 500
  batch_size: 256
  warmup_steps: 5000
  buffer_size: 50000
  collect_per_step: 16
  update_per_step: 1

q_network:
  hidden: [256, 256]
  dueling: false
```

**`vdn_masup_SF.yaml` 全文**（避免你漏改）：

```yaml
# VDN + MASUPEnv (SF / SFcrimemap)
# 用法: python train.py configs/experiments/masup/vdn_masup_SF.yaml

algo_name: vdn
env_type: masup
q_type: mlp

exp_name: vdn-SF
graph_name: SF
track_wandb: true
wandb_project: MAP-RL

env:
  graph_path: graphs/SFcrimemap.json
  enable_wait: true
  deltaT: 0.333
  num_agents: 3
  init_positions: [0, 5, 11]
  episode_len: 3000
  custom_configs:
    T: 200.0
    role_imformation: decision
    reward_scale: 0.01
    truncate_by_time: true
    idi_scale: 0.05
    contribution_scale: 0.05

algo:
  lr: 1.0e-4
  gamma: 0.99
  tau: 1.0
  target_update_freq: 200
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995
  use_double_dqn: true
  shared_policy: true

training:
  num_envs: 16
  use_subproc: true
  num_steps: 64
  total_steps: 10000000
  save_interval: 500
  batch_size: 256
  warmup_steps: 5000
  buffer_size: 50000
  collect_per_step: 16
  update_per_step: 1

q_network:
  hidden: [256, 256]
  dueling: false
```

---

### 2. Sweep：`configs/sweep/masup/`

在现有 longedge sweep 上只改 **base-config 路径** 与 **`collect_per_step`**（与 `num_envs: 16` 一致）。可复制现有文件为 `*_SF.yaml` 并替换文首注释与下列字段：

| 新文件 | 建议 `--base-config` |
|--------|----------------------|
| `ippo_masup_SF.yaml` | `configs/experiments/masup/ippo_masup_SF.yaml` |
| `vdppo_masup_SF.yaml` | `configs/experiments/masup/vdppo_masup_SF.yaml` |
| `qmix_masup_SF.yaml` | `configs/experiments/masup/qmix_masup_SF.yaml` |
| `vdn_masup_SF.yaml` | `configs/experiments/masup/vdn_masup_SF.yaml` |
| `iql_masup_SF.yaml` | `configs/experiments/masup/iql_masup_SF.yaml` |

**`qmix_masup.yaml` 的 SF 版**：复制 `qmix_masup.yaml`，把 `collect_per_step: 24` 改为 `16`（与 16 并行环境一致）。

**`vdn_masup.yaml` 的 SF 版**：同样 `collect_per_step: 16`。

**`ippo_masup.yaml` 的 SF 版**：全文不变，仅注释里的 base 改为 `ippo_masup_SF.yaml`。

**`vdppo_masup_SF.yaml` sweep**：可复制 `configs/sweep/oucs/vdppo_gru_oucs.yaml`，删去或固定 `data_chunk_length: 10`（MLP Actor），metric 建议与 QMIX 一致用 `env/wi_fromT`（若你日志里该标量存在）。

**`iql_masup_SF.yaml` sweep**：复制 `vdn_masup_SF.yaml` sweep，并固定：

```yaml
  sync_replay:
    value: true
  shared_policy:
    value: false
```

（若你当前 `sweep.py` 不支持嵌套键，再改为与仓库里 `custom_configs.*` 相同的扁平约定；没有的话需在 sweep 脚本里加映射。）

---

### 3. Eval：`configs/eval/masup/masup_SF.yaml`

当前文件中 **`deltaT: 200.0` 与训练用的 `0.333` 不一致**，建议改成与 SF 实验一致，并补上 `custom_configs`，例如：

```yaml
# MASUP SF 评估（与 experiments/masup/*_masup_SF.yaml 对齐）
# 用法: python evaluators/test.py --env_config configs/eval/masup/masup_SF.yaml --model models/xxx/final

env_type: masup

env:
  graph_path: graphs/SFcrimemap.json
  enable_wait: true
  deltaT: 0.333
  num_agents: 3
  init_positions: [0, 5, 11]
  episode_len: 3000
  edge_time_jitter: false
  custom_configs:
    T: 200.0
    role_imformation: decision
    reward_scale: 0.01
    truncate_by_time: true
    idi_scale: 0.05
    contribution_scale: 0.05

eval:
  num_episodes: 10
  save_plot: evaluators/results/SF/masup/best_eval.png
  show_plot: false
  animation: true
  event_driven: true
  max_frames: 2000
```

---

### 4. 顺带修正 `mappo_masup_SF_scratch.yaml`

删除重复的 `T: 200.0` 行，只保留一个。

---

请 **切换到 Agent 模式** 后说一声「写入 SF 配置」，我可以在项目里自动创建上述 5 个 experiment、5 个 sweep（及更新 eval / 修 mappo 重复键），避免手工复制出错。

---

**User**

直接创建新文件，注意我希望并行环境数num_envs统一为24, 此外sweep时的actor_lr的下界都设置到1e-6量级

---

**Cursor**

正在创建 SF 的 experiment 与 sweep 配置，`num_envs` 与 `collect_per_step` 设为 24，并将含 `actor_lr`/`lr` 的 sweep 下界设为 `1e-6`。



正在创建实验与 sweep 文件，并将 `mappo_masup_SF_scratch.yaml`、`masup_SF.yaml` 同步为 `num_envs: 24` 及一致的环境字段。



已按你的要求做完，摘要如下。

### 新建实验配置（`num_envs: 24`，off-policy 的 `collect_per_step: 24`）

| 文件 |
|------|
| `configs/experiments/masup/ippo_masup_SF.yaml` |
| `configs/experiments/masup/vdppo_masup_SF.yaml` |
| `configs/experiments/masup/qmix_masup_SF.yaml` |
| `configs/experiments/masup/vdn_masup_SF.yaml` |
| `configs/experiments/masup/iql_masup_SF.yaml` |

### 新建 Sweep（学习率下界 `1e-6`）

- **IPPO**：搜的是 `lr`（单优化器），`min: 1.0e-6`  
- **VDPPO**：`actor_lr` 与 `q_lr` 均为 `min: 1.0e-6`  
- **QMIX / VDN / IQL**：搜的是 `lr`，`min: 1.0e-6`  
- **collect_per_step**：QMIX / VDN / IQL 的 SF sweep 里固定为 **24**（与 `num_envs` 一致）

| 文件 |
|------|
| `configs/sweep/masup/ippo_masup_SF.yaml` |
| `configs/sweep/masup/vdppo_masup_SF.yaml` |
| `configs/sweep/masup/qmix_masup_SF.yaml` |
| `configs/sweep/masup/vdn_masup_SF.yaml` |
| `configs/sweep/masup/iql_masup_SF.yaml` |

IQL sweep 里固定了 `sync_replay: true`、`shared_policy: false`（与 `sweep.py` 的扁平 `algo` 覆盖一致）。

### 同步修改

- **`mappo_masup_SF_scratch.yaml`**：去掉重复的 `T: 200.0`，`num_envs` 改为 **24**。  
- **`configs/eval/masup/masup_SF.yaml`**：与 SF 训练环境对齐（`deltaT: 0.333`、`T` / `reward_scale` / IDI 等）。  
- **`configs/sweep/masup/vdn_masup.yaml`**（LongEdge 用）：`lr` 下界从 `5e-5` 改为 **`1e-6`**，与「sweep 学习率下界」一致。

### Sweep 示例

```bash
python sweep.py --base-config configs/experiments/masup/qmix_masup_SF.yaml \
                --sweep-config configs/sweep/masup/qmix_masup_SF.yaml --count 40
```

如需再为 **MAPPO + SF** 单独加一个 `mappo_masup_SF.yaml` 的 sweep 文件（base 指向 `mappo_masup_SF_scratch.yaml`），可以说明一下我帮你补上。

---

**User**

拓扑图的graph diameter是怎么定义的？对于我现在graphs中的各个json文件，graph diameter分别是多少

---

**Cursor**

说明图直径的定义，并计算 `graphs` 目录下各 JSON 图的直径。



## 图直径（graph diameter）怎么定义？

在**无向简单图**里（把 JSON 里 `from`–`to` 看成无向边，双向各一条时合并成一条）：

- **无权重（按“跳数”）直径**：  
  先算任意两点间**最短路的边数** \(d(u,v)\)（BFS）。  
  **直径** \(D = \max_{u,v} d(u,v)\)，即所有顶点对里**最短路径边数的最大值**。  
  直观理解：图上离得最远的两个点，中间至少要经过几条边。

- **带权重直径**（你们 JSON 里有 `weight` 时常用）：  
  先按边权做**最短路长度**（Dijkstra，边权非负）。  
  **加权直径** \(D_w = \max_{u,v} d_w(u,v)\)，即所有点对**最短加权距离**的最大值。

若图**不连通**，一般两点在不同连通块时距离为 \(+\infty\)，整体直径常记为 \(+\infty\)，或按**各连通块直径的最大值**分别说。你仓库里这些 JSON 在当前无向解释下都是 **1 个连通块**。

---

## `graphs/*.json` 上的结果

处理约定：边视为**无向**；JSON 里 `A→B` 与 `B→A` 合并为一条无向边；`|E|` 为无向边条数（有向条目数 / 2）。

| 文件 | \|V\| | 无向边数 | **跳数直径** | **加权直径** |
|------|--------|-----------|--------------|----------------|
| `Chain10nodes9edges.json` | 10 | 9 | 9 | 10.0 |
| `SFcrimemap.json` | 12 | 66 | **1** | 51.0 |
| `cumberland.json` | 40 | 44 | 13 | 72.9 |
| `grid.json` | 50 | 91 | 10 | 24.0 |
| `island.json` | 50 | 84 | 12 | 72.71 |
| `long_edge.json` | 4 | 4 | 2 | 6.0 |
| `mapA.json` | 50 | 105 | 8 | 66.65 |
| `mapB.json` | 50 | 69 | 12 | 79.04 |
| `marostica.json` | 40 | 52 | 13 | 32.0 |
| `milwaukee.json` | 40 | 56 | 15 | 166.6 |
| `simple_TSP_12.json` | 12 | 20 | 4 | 14.0 |

说明：**`SFcrimemap.json` 在 12 个顶点上几乎是完全图**（66 条无向边 = \(C_{12}^2\)），所以任意两点都有直连，**跳数直径为 1**；加权直径仍是“按权重的最远最短路”，所以是另一个数（约 51）。

若你后续在论文或实验里要统一口径，需要明确写的是 **hop diameter** 还是 **weighted diameter**，以及边是否按无向处理；需要的话我可以对照 `env` 里建图代码再核对是否与上述一致。

---

**User**

现在回到之前讨论on-policy算法的critic网络输出归一化，目前在不采用模仿学习初始化的情况下，从头训练直接在配置参数中启用这个开关，能正常训练吗？是否mappo/vdppo/ippo全都适配？

---

**Cursor**

正在核对 `use_value_norm` 在无预训练时的初始化及 MAPPO、IPPO、VDPPO 中的使用路径。



结论先说：**无模仿学习、只在 YAML 里打开 `use_value_norm`，MAPPO / IPPO 可以正常从零训**；**VDPPO 当前实现基本没有接上这套 critic 归一化，开了也几乎不起作用**。

---

## 从零训练时开关会怎样？

`AlgorithmBase` 里只要 `use_value_norm=True`，就会创建 `RunningMeanStd`，**不依赖** `value_norm_config`（模仿学习 checkpoint 只负责把 `mean/var` 设成初值）：

```116:122:/home/wzh/MAP-imitation-framework/algorithms/algorithm_base.py
        if self.use_value_norm:
            from utils.train_utils import RunningMeanStd
            self.ret_rms = RunningMeanStd(shape=(1,)).to(self.device)
            if value_norm_config is not None:
                self.ret_rms.mean.fill_(value_norm_config.get('ret_mean', 0.0))
                self.ret_rms.var.fill_(value_norm_config.get('ret_std', 1.0) ** 2)
                self.ret_rms.count.fill_(1.0)
```

`RunningMeanStd` 默认 `mean=0`、`var≈1`，初期等价于「先按近似恒等尺度做 GAE」，随后用 rollout 的 `ret` 在线更新统计量，**逻辑上是自洽的**。

---

## MAPPO / IPPO：已适配

- **MAPPO**：`prepare_batch` 走 `CentralizedCriticMixin._prepare_batch_ctde`，对 critic 输出做反归一化再算 GAE，并在该阶段 `ret_rms.update`；**update** 走 `A2CBase.update`，value 目标为 \((ret - \mu)/\sigma\)：

```249:252:/home/wzh/MAP-imitation-framework/algorithms/rl/a2c.py
                if self.use_value_norm and self.ret_rms is not None:
                    target = (mb_ret - self.ret_rms.mean) / (self.ret_rms.std + 1e-8)
                else:
                    target = mb_ret
```

- **IPPO**：自定义 `prepare_batch` 里同样对 `values` 做反归一化、算 GAE、汇总各 agent 的 `ret` 去 `update` `ret_rms`；`update` 里与 IPPO 一致使用归一化后的 value target（`ippo.py` 255–256 行附近）。

因此在 **`algo:` 下设置 `use_value_norm: true`**（并确保用的是带该字段的 on-policy 参数类，如 `MAPPOParams` / `IPPOParams`）即可从零训练。

---

## VDPPO：**未适配** value norm

VDPPO **没有独立 critic**，baseline 来自 **\(V_i=\max_a Q_i\)**，且 **`prepare_batch` 完全自写**，里面**没有**对 Q 值做 `ret_rms` 的反归一化，也**没有** `ret_rms.update`；**`update` 里只有 Q_tot 的 TD 型 loss 和 actor PPO**，**没有**走 `A2CBase` 里那段 critic MSE + `use_value_norm` 的逻辑。

因此即 `VDPPOParams` 继承 `OnPolicyParams` 且能设 `use_value_norm: true`，也**不会改变 Q 训练或 GAE 所用尺度**，等价于**基本无效**（最多多占一个 buffer）。若要对 VDPPO 做「值归一化」，需要**单独设计**（例如对 \(Q/y\) 或 return 做 PopArt / reward scaling），不能指望与 MAPPO 共用现成开关。

---

## 小结

| 算法   | 仅配置 `use_value_norm: true` 从零训练 | 说明 |
|--------|--------------------------------------|------|
| MAPPO  | 可以，路径完整                         | CTDE `prepare_batch` + 共享 `A2CBase.update` |
| IPPO   | 可以，路径完整                         | 自写 `prepare_batch` + 自写 `update` 里 value target |
| VDPPO  | **不算适配**                          | 无 critic value loss；prepare 未用 `ret_rms` |

**配置方式**：在对应实验 YAML 的 `algo:` 下增加一行，例如：

```yaml
algo:
  use_value_norm: true
  # ... 其余不变
```

早期若干步 `ret_rms` 方差会随数据快速变化，属正常现象；若出现不稳定，可再调 `vf_coef`、学习率或观察 `ret` 尺度，与「是否从零」无冲突。

---

**User**

现在的模仿学习采样以及监督学习与训练的pipeline是否适用于所有算法+masup？应该如何配置？

---

**Cursor**

梳理模仿学习数据流、训练入口及各算法与 MASUP 的兼容性。



结论可以概括成三句话：**采样 + 模仿监督这条链路本身是「环境驱动」的，MASUP 下图、智能体数一致就能跑**；**接 RL 微调时，只有 on-policy 且带 Actor/Critic 的算法会加载 `policy.pt` / `critic.pt`**；**QMIX/VDN/IQL 等 off-policy 在 `train.py` 里会直接跳过预训练加载**。

---

## 1. 当前 Pipeline 分三段

| 阶段 | 入口 | 作用 |
|------|------|------|
| 启发式采样 | `trainers/imitator/heuristic_sampler.py` | 在环境里用启发式（如 ER）采轨迹，写 **HDF5**（`obs` / `critic_states` / `actions` / `returns` 等） |
| 模仿监督 | `trainers/imitator/imitation_trainer.py` + `configs/imitator/*.yaml` | 读 HDF5，训 **ActorMLP + CriticMLP**，保存 `policy.pt`、`critic.pt`，并在同目录 `config.yaml` 里可带 **`value_normalization`** |
| RL 训练 | `python train.py <experiment.yaml>` | 若配置了 **`actor_path` / `critic_path`** 且算法为 **on-policy**，则 `_load_pretrained` 加载权重，并可读旁路 `config.yaml` 的 `value_norm_config` |

`train.py` 里写得很明确：**仅 on-policy 会加载预训练**；off-policy 直接打印跳过：

```542:551:/home/wzh/MAP-imitation-framework/train.py
    # ---- 5. 加载预训练权重（仅 on-policy，off-policy 一般不预训练） ----
    value_norm_config = None
    if trainer_type == "on_policy":
        value_norm_config = _load_pretrained(
            actor_net, critic_net,
            config.actor_path, config.critic_path,
            device,
        )
    else:
        print("[Train] Off-policy algorithm, skipping pretrained weight loading")
```

因此：**并不是「所有算法 + MASUP」都会在 RL 阶段吃模仿权重**。

---

## 2. 是否适用于「所有算法 + MASUP」？

### MASUP 本身

- **采样**：`heuristic_sampler` 通过 `ENV_REGISTRY` 建环境，默认 **`env_type` 多为 `masup`**；图、智能体数、奖励等由 **env 配置**决定。换 `graphs/xxx.json`、换 eval YAML，只要仍是 **MASUP 接口**（`state()`、`obs` 等），就可以采。
- **模仿训练**：`imitation_trainer` 用 **`env_config` 指向的 eval YAML** 起临时环境，**自动推断** `obs_dim` / `action_dim` / `critic_state_dim`（全局 state + agent one-hot）。因此 **任意 MASUP 图**都可以配，只要：
  - HDF5 与 **该图、该 `num_agents`、该观测/动作空间** 一致；
  - 与后续 RL 实验里的 **观测维、动作数、critic 输入维（CTDE 全局 state + one-hot）** 一致。

### 按算法类型

| 类型 | 代表 | 模仿 → RL 是否「接上」 |
|------|------|------------------------|
| **On-policy Actor–Critic** | MAPPO、IPPO、MAA2C、单智能体 PPO/A2C 等 | **可以**：`actor_path` / `critic_path` 会加载；需 **`actor_type`/`critic_type` 与模仿时结构一致**，维度与 MASUP 一致。 |
| **VDPPO** | 有 **Actor + Q 网络**，**无独立 Critic** | **部分**：实验里通常 **`critic_type` 为空** → `critic_net` 不会被创建，**只会加载 actor**（若配了 `critic_path` 也不会进网络）。**模仿得到的 critic 权重不能自动变成 Q 网络**。 |
| **Off-policy** | QMIX、VDN、IQL、D3QN 等 | **RL 阶段不加载** 模仿的 `policy.pt`/`critic.pt`；若要做「预训练 Q」，需要**另写加载逻辑或改 `train.py`**，当前 pipeline **未适配**。 |

---

## 3. 应如何配置（推荐 checklist）

### A. 模仿侧（任意 MASUP 图）

1. 准备 **eval 风格 YAML**（与 RL 里 `env` 一致或至少 **图、智能体数、deltaT、观测定义** 一致），例如 `configs/eval/masup/masup_SF.yaml`。  
2. **采样**：用 `heuristic_sampler` 指定同一套 env（具体 CLI 以脚本参数为准），得到 **`.h5`**。  
3. **`configs/imitator/xxx.yaml`**：  
   - `env_type` + `env_config: <上述 eval yaml>`  
   - `data_path` 指向 HDF5  
   - `actor` / `critic_hidden_sizes` 与后续 RL 的 **MLP 宽度**一致（否则 `load_state_dict` 对不上）  
   - `use_value_norm: true` 若希望与 RL 的 **`use_value_norm` + `value_norm_config`** 衔接  

### B. RL 微调侧（仅 on-policy 全功能）

在 **experiment YAML 顶层**（与 `mappo_masup_tsp12_imi.yaml` 相同层级）设置：

```yaml
actor_path: models/.../policy.pt
critic_path: models/.../critic.pt
```

并保证：

- **`env`** 与采数/模仿时 **同一 MASUP 设定**（图、智能体数、关键 `custom_configs`）。  
- **`actor` / `critic` 结构** 与模仿训练一致。  
- 若模仿保存了 `value_normalization`：RL 的 `algo` 里设 **`use_value_norm: true`**，否则 `ret_rms` 不会用 checkpoint 里的均值方差（`train.py` 只会把 `value_norm_config` 传进算法，是否启用仍由 **`params.use_value_norm`** 决定）。

### C. 若你坚持用 QMIX/VDN/IQL + MASUP

- 现有 **`train.py` 不会**加载模仿的 actor/critic。  
- 可行方向包括：把 **行为克隆后的策略** 单独导出为初始化、或扩展 off-policy 的 **checkpoint 加载**；这已超出当前「模仿 pipeline」的默认行为。

---

**一句话**：**采样 + 模仿学习对「任意 MASUP 图」在数据与维度一致的前提下都适用**；**和 RL 自动衔接的，目前主要是 MAPPO/IPPO 等 on-policy Actor–Critic**；**VDPPO 只能稳妥用 actor 预训练**；**QMIX/VDN/IQL 要接模仿需改代码或换流程**。

---

**User**

1. 当我在预训练后面接上ippo的时候，是每个智能体的独立网络都用相同的预训练权重初始化吗？
2. vdppo以及其他off-policy能用类似mappo预训练采集数据时的思路对Q网络等值网络也做监督学习吗？

---

**Cursor**

基于以上代码，以下是对两个问题的完整回答：

---

## 1. IPPO：所有 agent 的独立网络都用相同的预训练权重初始化吗？

**是的，完全相同。** 这由以下两段逻辑共同决定：

**第一步，`_load_pretrained` 是把权重加载到 `actor_net` 这一个 PyTorch 对象上：**

```172:178:/home/wzh/MAP-imitation-framework/train.py
    if has_actor_path:
        actor_p = Path(actor_path)
        if actor_p.exists():
            actor_ckpt = torch.load(actor_p, map_location=device, weights_only=True)
            actor_net.load_state_dict(actor_ckpt.get("actor_state_dict", actor_ckpt))
            ...
```

**第二步，`_build_policy` 传给 IPPO 时 `shared=True`（MARL actor 路径固定为 True）：**

```230:238:/home/wzh/MAP-imitation-framework/train.py
        else:
            return MultiAgentPolicy(
                agent_ids=dims["agent_ids"],
                ...
                policy_class=ActorPolicy,
                policy_kwargs={"actor": actor_net},
                shared=True,       # ← 强制共享
            )
```

因此 **MAPPO 和 IPPO 在参数加载层面是完全一样的**——它们都用一个共享的 `actor_net` 实例。区别只在 **RL 阶段之后**：

- **MAPPO**：参数共享贯穿始终，所有 agent **永远**使用同一套权重。
- **IPPO**：`MultiAgentPolicy` 里 `shared=True` 意味着在这个框架实现里也是参数共享！换句话说，**当前仓库里 IPPO 也是参数共享的 actor**，只是 critic 输入不拼全局 state + one-hot（而是用各 agent 自己的 obs），和真正「每个 agent 独立网络」不同。

这意味着：**模仿预训练 → 接 IPPO 的路径和接 MAPPO 的路径在「网络如何加载」上没有区别**，都是一个网络对象，所有 agent 共享它。

---

## 2. VDPPO / off-policy 能对 Q 网络做类似 Critic 的监督学习吗？

**可以做，但逻辑上有差异，且当前 pipeline 不支持，需要设计。**

### 先明确「对 Q 做监督」的含义

模仿学习里 Critic 的监督目标是 **蒙特卡洛回报 $G$**（从轨迹直接累加），这是「状态价值函数」$V(s)$ 的无偏估计。在 CTDE 里，Critic 接的是 **全局 state + agent one-hot**，监督目标是 $G_i$（各 agent 的 MC return）。

Q 网络接的是 **观测 + 动作** 输出 $Q(o, a)$，要监督它需要：

| 监督目标 | 数据要求 | 和现有 HDF5 的匹配度 |
|----------|----------|----------------------|
| 蒙特卡洛 Q：$G_t$（轨迹中 t 步的回报，代入实际动作） | 已有 `returns` 与 `actions` | **可以直接用**：$Q(o_t, a_t) \approx G_t$ |
| TD 目标：$r + \gamma \max Q'$ | 需要完整 transition + next\_obs | HDF5 里有 `obs` 序列，可以从相邻步推 |
| 对 $Q_{\text{tot}}$（QMIX）做监督 | 需要 joint state + 全 agent 动作 | 现有 `critic_states`（包含 state）+ `actions`（含每个 agent）可以凑出来 |

### VDN/QMIX/IQL 用 MC Return 监督的可行路径

数学上，对 Q 的监督类似：

$$Q_\theta(o_t^i, a_t^i) \approx G_t^i$$

对 VDN/QMIX，还可以在混合之后监督 $Q_{\text{tot}}$：

$$Q_{\text{tot}}(s_t, \mathbf{a}_t) \approx \sum_i G_t^i$$

HDF5 里已有：`obs`、`actions`、`returns`（MC 回报），理论上**完全够用**，但：

1. **MC return 在长 episode 里方差很大**，尤其 SF 图上 $T=200$，监督效果会不如 Critic 里用 `ret_rms` 归一化后的学习；
2. **$Q(o, a)$ 不如 $V(s)$ 稳定**：$V$ 对观测估值，$Q$ 还多了动作维度，用 MC 监督时方差更高；
3. **QMIX 的 Mixer 单调约束**：如果用 MC 来监督 $Q_{\text{tot}}$，没有办法像 RL TD loss 那样天然保持单调性约束，可能需要额外处理。

### VDPPO 的特殊情况

VDPPO 的 Q 网络接的是 **`global_state + agent_one_hot`**，这和 Critic 用的维度**完全一致**，所以 `critic_states`（已经是 state + one-hot）可以直接喂进去。监督目标是 $Q_i(s, a_i)$ 近似用 $G_i$：

$$Q_\theta(s_t, \text{one\_hot}_i, a_t^i) \approx G_t^i$$

这个**在数据层面完全可行**，和 Critic 的监督（$V(s) \approx G$）的区别只在于：**是否 gather 实际动作后才计算 loss**（V 直接输出标量，Q 需要先选动作）。

### 当前 pipeline 缺什么

`imitation_trainer.py` 只实现了 **ActorMLP（行为克隆）+ CriticMLP（V 函数回归）** 的监督；以下组件**没有**：

1. **Q 网络监督模块**：需要在 batch forward 时从 Q 矩阵里 `gather` 实际动作对应的 Q 值，再与 `returns` 对比；
2. **Mixer 初始化**（QMIX）：QMIX 的 Mixer 是超网络，Q 个体值经过它混合才输出 $Q_{\text{tot}}$；监督时是否绕过 Mixer 直接对每个 $Q_i$ 做回归，还是联合监督 $Q_{\text{tot}}$，需要设计；
3. **权重加载路径**：`train.py` 里 off-policy 完全跳过 `_load_pretrained`；VDPPO 的 `actor_path` 可以加载 actor，但 **`q_net` 加载没有对应的配置字段和加载逻辑**。

**总结**：技术上可行，需要：

- 在 `imitation_trainer.py` 里增加 Q 网络监督分支（gather 实际动作 → MSE(Q, G)）；
- 在 `train.py` 的 off-policy 分支加 `q_path` 字段 + 加载逻辑；
- 在 `ExperimentConfig` 里增加 `q_path: Optional[str]`；
- 或者在 VDPPO 里加 `actor_path`（现有字段）+ 单独的 `q_path`（新增）。

这不是一个复杂改动，只是当前**没有实现**。如果你认为「用 MC return 给 Q 做初始化有必要」（比如在大图 SF 上冷启动确实慢），可以进入 Agent 模式让我实现这段逻辑。

---

**User**

我新建了一个@MAP-imitation-framework/configs/sweep/masup/mappo_masup_SF.yaml 用来区分不同map上不同的各参数sweep范围，我希望这个文件里面放SF地图的，而MAP-imitation-framework/configs/sweep/masup/mappo_masup.yaml中放long-edge的

---

**Cursor**

两张图的关键参数差异：SF 的 episode 极长（`T=200`，`episode_len=3000`）、环境更复杂，需要更大的 rollout、更高的 gamma、更频繁保存；long_edge 极短（`T=5`，`episode_len=100`），可以用更小批次快速迭代。

完成。两个文件的核心差异对应图的物理特性：

| 参数 | `mappo_masup.yaml`（LongEdge） | `mappo_masup_SF.yaml`（SF） |
|------|-------------------------------|---------------------------|
| `gamma` | `[0.97, 0.99, 0.999]` | `[0.99, 0.995, 0.999]` |
| `num_steps` | `[512, 1024, 2048]` | `[2048, 4096, 8192]` |
| `minibatch_size` | `[256, 512, 1024, 2048]` | `[1024, 2048, 4096, 8192]` |
| `total_steps` | `10000000` | `20000000` |
| `save_interval` | `1000` | `200` |
| `use_value_norm` | 不搜（由 base config 决定） | 固定 `true` |

**设计原因**：
- SF 的 episode 长达 3000 步（`T=200`），一个 rollout 如果太小则几乎采不到完整 episode，因此 `num_steps` 下界拉高到 2048；gamma 也应偏高（折扣到 episode 尾端仍有效）。
- LongEdge 的 episode 只有 100 步（`T=5`），小 rollout 就能装多个完整 episode，不需要那么大的批次，gamma 搜索范围可以更宽。
- `save_interval` SF 设 200 是因为每个 epoch 消耗时间长，需要更密集地保存 checkpoint 以防错过最优点；LongEdge 迭代快所以 1000 即可。

---

**User**

突然发现了一个问题，sweep之后的自动评估有个问题，当我对masup环境的custom_configs：idi_scale和contribution_scale做sweep时，评估的这两个参数确实早已确定死的，没有读取评估的某个trial在训练时使用的是啥；有办法解决这个问题吗？

---

**Cursor**

先摸清整个评估触发链路，再看 `config.yaml` 里存了什么信息。

现在完全清楚问题所在，也找到了两处修复点：

现在清楚了问题根源，实现两处最小改动：

1. **`train.py`**：把训练时的 `custom_configs` 存进 checkpoint 的 `extra` 段；
2. **`evaluators/test.py`**：`test_trained_policy` 读 checkpoint 的 `config.yaml`，把保存的 `custom_configs` 合并进评估环境配置，同时 `run_eval_from_config` 把这一步串起来。

现在修改 `test.py`，在 `test_trained_policy` 里加上合并逻辑，并在 `run_eval_from_config` 里读取 checkpoint 的 `custom_configs` 传进去：

现在修改 `run_eval_from_config`，从模型的 `config.yaml` 里读出 `train_env_custom_configs`，传给 `test_trained_policy`：

改动完成。来整体描述一下方案：

---

## 问题根因

`eval_best_runs` → `run_eval_from_config` → `test_trained_policy` 的整条链路，环境只从**固定的 eval YAML**（如 `masup_SF.yaml`）读取 `idi_scale` / `contribution_scale`，完全不知道当前 trial 训练时 sweep 用了哪个值。

---

## 修复方案：checkpoint 自描述

不依赖 wandb API，让 **模型 `config.yaml` 本身记录训练时的 `custom_configs`**，评估时自动读回。

### 改动 1：`train.py` — 写入 checkpoint

在 `_eval_meta` 中增加一个字段，保存训练时（包含 sweep 覆盖后）的 `custom_configs`：

```python
"train_env_custom_configs": dict(config.env.custom_configs or {}),
```

这样每次保存 checkpoint（中间 `iter_N` 和最终 `final`）时，`config.yaml` 的 `extra` 段都会带上例如：

```yaml
extra:
  train_env_custom_configs:
    idi_scale: 0.1
    contribution_scale: 0.0
    T: 200.0
    reward_scale: 0.01
    ...
```

### 改动 2：`evaluators/test.py` — 评估时读回并合并

**`run_eval_from_config`**：加载模型 `config.yaml`，提取 `extra.train_env_custom_configs`，传给 `test_trained_policy`。

**`test_trained_policy`**：新增 `env_custom_config_overrides` 参数，在 `load_eval_config` 之后把训练时值合并进 `env_cfg.custom_configs`，训练值优先于 eval YAML 里的同名字段。

---

## 效果

| 场景 | 行为 |
|------|------|
| `sweep.py eval_best_runs` | 每个 trial 的评估环境自动使用该 trial 训练时的 `idi_scale`/`contribution_scale` |
| 手动 `test.py --model xxx` | 同样读 `config.yaml`，有 `train_env_custom_configs` 就用，没有就以 eval YAML 为准（向后兼容） |
| 老 checkpoint（没有这个字段） | `train_custom_configs = None`，跳过覆盖，行为与改动前完全一样 |

---

**User**

(Patrolling) root@autodl-container-57fd45942c-3c9d11df:~/MAP-imitation-framework# python sweep.py --base-config configs/experiments/masup/mappo_masup_SF_scratch.yaml --sweep-config configs/sweep/masup/mappo_masup_SF.yaml --create-only
Traceback (most recent call last):
  File "/root/MAP-imitation-framework/sweep.py", line 363, in <module>
    main()
  File "/root/MAP-imitation-framework/sweep.py", line 341, in main
    sweep_id = create_sweep(args, project)
  File "/root/MAP-imitation-framework/sweep.py", line 250, in create_sweep
    sweep_config = yaml.safe_load(f)
  File "/root/miniconda3/envs/Patrolling/lib/python3.10/site-packages/yaml/__init__.py", line 125, in safe_load
    return load(stream, SafeLoader)
  File "/root/miniconda3/envs/Patrolling/lib/python3.10/site-packages/yaml/__init__.py", line 81, in load
    return loader.get_single_data()
  File "/root/miniconda3/envs/Patrolling/lib/python3.10/site-packages/yaml/constructor.py", line 49, in get_single_data
    node = self.get_single_node()
  File "/root/miniconda3/envs/Patrolling/lib/python3.10/site-packages/yaml/composer.py", line 39, in get_single_node
    if not self.check_event(StreamEndEvent):
  File "/root/miniconda3/envs/Patrolling/lib/python3.10/site-packages/yaml/parser.py", line 98, in check_event
    self.current_event = self.state()
  File "/root/miniconda3/envs/Patrolling/lib/python3.10/site-packages/yaml/parser.py", line 171, in parse_document_start
    raise ParserError(None, None,
yaml.parser.ParserError: expected '<document start>', but found '<block mapping start>'
  in "configs/sweep/masup/mappo_masup_SF.yaml", line 12, column 1

为何有这个报错？

---

**Cursor**

文件内容本身语法没问题，但错误指向第 12 行（`method: bayes`），提示 YAML 解析器在应该见到「文档开始」时却见到了「块映射开始」。这通常是文件头部存在 **UTF-8 BOM**（`\xef\xbb\xbf`）或不可见零宽字符，让 YAML 解析器误判了文档结构。

本地文件解析完全正常（PyYAML 6.0.3）。让我看看 `sweep.py` 在第 250 行如何打开文件，确认有没有特殊处理：

服务器上的文件解析出错，但本地文件完全正常。这说明**服务器端的文件内容与本地不同**，很可能是同步时编码损坏或者文件不完整。先在服务器上确认实际内容：

```bash
# 在 autodl 服务器上执行
xxd configs/sweep/masup/mappo_masup_SF.yaml | head -5
```

不过更直接的方式是检查服务器上到底哪行有问题：

```bash
python3 -c "
with open('configs/sweep/masup/mappo_masup_SF.yaml', 'rb') as f:
    for i, line in enumerate(f, 1):
        if i <= 15:
            print(i, repr(line[:60]))
"
```

---

**YAML `expected '<document start>'` 报错的根本原因**

PyYAML 会将文件开头以 `%` 开头的行视为 YAML 指令（如 `%YAML 1.1`、`%TAG`），解析完指令后它要求必须有一个显式的 `---` 文档分隔符，才能开始读取块映射（`key: value`）。若服务器上的文件：

- 开头有 **UTF-8 BOM** (`\xef\xbb\xbf`)，导致第一个 `#` 注释被误判
- 某行注释因编码损坏变成了 `%` 开头
- 传输时产生了乱码字符

就会触发这个错误，且位置精准定位在第一个真正 YAML 内容出现的行（即第 12 行的 `method: bayes`）。

---

**最简单的修复方式**：在服务器上直接用 Python 重写该文件（跳过编码问题）：

```bash
cd ~/MAP-imitation-framework
python3 << 'PYEOF'
content = """\
method: bayes

metric:
  name: env/wi_fromT
  goal: minimize

parameters:
  actor_lr:
    min: 1.0e-6
    max: 1.0e-3
    distribution: log_uniform_values
  critic_lr:
    min: 1.0e-6
    max: 5.0e-3
    distribution: log_uniform_values

  clip_range:
    min: 0.1
    max: 0.4
  vf_coef:
    min: 0.1
    max: 2.0
  ent_coef:
    min: 0.0
    max: 0.3
  gae_lambda:
    min: 0.9
    max: 1.0

  gamma:
    values: [0.99, 0.995, 0.999]

  use_value_norm:
    value: true

  num_steps:
    values: [2048, 4096, 8192]
  minibatch_size:
    values: [1024, 2048, 4096, 8192]
  update_epochs:
    values: [3, 5, 10]

  total_steps:
    value: 20000000

  save_interval:
    value: 200

  custom_configs.idi_scale:
    values: [0.0, 0.05, 0.1]
  custom_configs.contribution_scale:
    values: [0.0, 0.05, 0.1]
"""
with open('configs/sweep/masup/mappo_masup_SF.yaml', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done")
PYEOF
```

这会用纯 ASCII 注释的干净文件覆盖，彻底规避编码问题。之后再重新运行 `sweep.py` 即可。

---

**User**

再帮我把六个算法+masup+grid图的experiment/sweep/eval参数文件全部配置好吧，同样注意根据这张图的特点配置sweep范围；在这张图上统一使用6个智能体，初始位置均匀分布在节点上，episode_len: 500, T：50，确定开启critic归一化

---

**Cursor**

先看看现有文件结构和 grid 图，再参照已有配置来创建。

现在批量创建所有文件。Grid 图关键参数：50 节点、直径=10、边权≈2，用 `deltaT=0.5`，6 智能体均匀分布节点 `[1, 9, 17, 25, 33, 41]`。

现在创建 sweep 文件。Grid 图特点对应调整：`gamma` 范围下移（直径小、时间跨度短）、`num_steps` 减小、off-policy `epsilon_decay` 加快。

Sweep 目录里 ippo 文件少了（Write 工具那次报错）。确认一下：

全部 13 个文件已创建完毕。汇总如下：

---

**Graph 特性分析（用于调参依据）**

| 属性 | Grid | SF |
|------|------|-----|
| 节点数 | 50 | 大型真实图 |
| 边数 | 182（密集） | 稀疏 |
| 无权直径 | **10** | 更大 |
| T | 50 | 200 |
| episode_len | 500 | 3000 |

---

**Experiment 文件（6 个）**

| 文件 | 关键设置 |
|------|---------|
| `mappo_masup_grid.yaml` | `use_value_norm: true`, `num_steps: 1024` |
| `ippo_masup_grid.yaml` | `use_value_norm: true`, `lr` 单优化器 |
| `vdppo_masup_grid.yaml` | `actor_lr/q_lr` 双优化器，无 value_norm |
| `qmix_masup_grid.yaml` | `epsilon_decay: 0.995`，`warmup_steps: 2000` |
| `vdn_masup_grid.yaml` | 同上 |
| `iql_masup_grid.yaml` | `shared_policy: false`，软更新 `tau: 0.005` |

所有 experiment 文件通用设置：
- `deltaT: 0.5`（边权≈2，每条边需4步穿越）
- `num_agents: 6`，`init_positions: [1, 9, 17, 25, 33, 41]`（均匀分布）
- `episode_len: 500`，`T: 50.0`
- `num_envs: 24`，`total_steps: 10000000`

---

**Sweep 与 SF 的主要差异**

- `gamma` 范围：`[0.97, 0.98, 0.99]`（SF 为 `[0.99, 0.995, 0.999]`），因时间跨度短
- on-policy `num_steps`：`[512, 1024, 2048]`（SF 为 `[2048, 4096, 8192]`）
- off-policy `epsilon_decay`：`[0.990, 0.998]`（衰减更快，因 episode 短）
- off-policy `warmup_steps`：`[1000, 2000, 5000]`（SF 为 `[2000, 5000, 10000]`）
- off-policy 加了 `early_terminate: hyperband`，`min_iter: 500`

---

**User**

从头到尾仔细检查一遍当前vdppo/qmix/vdn/iql算法的实现，尤其是网络实例数量，我需要明确知道是否与该算法的论文完全严谨一致，避免出现之前我未知的ippo的critic输入问题

---

**Cursor**

我来系统地审查这四个算法的实现。先找到相关文件。

现在看网络、Policy层和 Mixer 实现。

现在看 `train.py` 中实际如何实例化这四个算法的网络和 policy。

再看 PPOBase 基类和 `create_algorithm` 工厂，理解 VDPPO 继承关系。

已读完全部相关代码。下面是系统性审查报告。

---

## 四个算法代码审查报告

---

### 一、网络实例数量总览

| 算法 | 网络实例 | 是否参数共享 |
|------|---------|------------|
| **VDN** | 1 Q-net + 1 target Q-net + SumMixer（无参） | 所有 agent 共享 1 个 Q-net |
| **QMIX** | 1 Q-net + 1 target Q-net + 1 QMIXMixer + 1 target QMIXMixer | 所有 agent 共享 1 个 Q-net |
| **IQL** | N Q-nets + N target nets + N 独立 optimizer | 每个 agent 独立一套 |
| **VDPPO** | 1 actor（shared）+ 1 Q-net + 1 target Q-net + 1 QPLEXMixer + 1 target QPLEXMixer | actor 共享；Q-net 一个但靠 one-hot 区分 agent |

---

### 二、VDN

**论文设计**：Q_tot = Σ_i Q_i(o_i, a_i)，参数共享，SumMixer 无可学习参数。

**实现路径**：
```
MultiAgentPolicy(shared=True)
  └── _shared_policy: ValuePolicy(q_network=q_net)

VDNAlgo.__init__:
  self.q_network = policy.get_policy(agent_ids[0]).q_network
  # 因 shared=True，get_policy() 无论输入哪个 agent_id 都返回 _shared_policy
  # → self.q_network 就是唯一的共享 Q-net
  self.target_q_network = deepcopy(self.q_network)
  self.mixer = SumMixer(n_agents, 0)
  self.target_mixer = SumMixer(n_agents, 0)  # 无参，冗余但无害
  self.optimizer → 只优化 self.q_network.parameters()
```

**发现的问题**：

❶ **`target_mixer` 冗余**：`SumMixer` 无任何可学习参数，`target_mixer` 的 `forward` 等价于 `mixer.forward`。创建两个实例纯属冗余。不影响正确性，但占内存。

❷ **`r_tot = rew_all.sum(dim=-1)` 的联合奖励问题**（之前讨论过）：VDN 论文假设 team reward 是全局统一的 r_t。在 MASUP 中逐 agent 求和会导致 N 倍放大（若各 agent 奖励相同时），且 edge agent 奖励为 0 时求和又不严格等于全局奖励。这是设计权衡，不是 bug，但需知晓。

❸ **潜在风险**：若意外配置 `shared_policy: false`，`VDNAlgo` 只取 `agent_ids[0]` 的 Q-net 并更新它，其他 agent 的网络永不被优化，严重 bug。当前 config 强制 `shared_policy: true`，所以无问题，但代码层没有防御性检查。

**论文符合性：✓ 正确**

---

### 三、QMIX

**论文设计**：Q_tot = f(Q_1,...,Q_N, s)，超网络保证 ∂Q_tot/∂Q_i ≥ 0（单调性）。

**实现路径**：
```
VDNAlgo(shared Q-net base)
  ↓ override
QMIXAlgo._init_mixer():
  self.mixer = QMIXMixer(n_agents, state_dim, embed_dim)
  self.target_mixer = deepcopy(self.mixer)  # ← 这里有可学习参数，deepcopy 必要

optimizer → self.q_network.parameters() + self.mixer.parameters()

_update_target_networks():
  super()  # 更新 target Q-net
  同时更新 target_mixer
```

**单调性约束验证**：

```python
# QMIXMixer.forward
w1 = self.hyper_w1(states).abs().view(B, N, self.embed_dim)  # ← abs() 保证非负
w2 = self.hyper_w2(states).abs().view(B, self.embed_dim, 1)  # ← abs() 保证非负
```

✓ 两层权重均通过 `abs()` 保证非负，满足 IGM 单调性约束。

**发现的问题**：

❶ **state 收集控制**：`needs_state = config.algo_name in ("qmix",)` 仅 QMIX 开启 state 收集，VDN/IQL/VDPPO 均为 False。`ReplayBuffer` 通过 `has_state=needs_state` 控制是否存储 state。QMIX `_compute_loss` 内有显式 RuntimeError 检查（state 为 None 时报错），防御完备。✓

❷ **Double DQN 实现方式**：
```python
# online Q 选 action
q_next_online = self.q_network(next_obs_all.view(B*N,-1)).view(B,N,-1)
next_actions = q_next_online.argmax(dim=-1, keepdim=True)
# target Q 评估
q_next_target = self.target_q_network(next_obs_all.view(B*N,-1)).view(B,N,-1)
q_next_chosen = q_next_target.gather(-1, next_actions).squeeze(-1)
# target mixer
q_tot_target = self.target_mixer(q_next_chosen, next_state)
```
这是 DDQN 对 Q_i 级别的扩展，原论文 QMIX 不含 DDQN，但这是标准改进。注意：DDQN 的 action selection 用 online Q_i，但 mixing 时用的是 `target_mixer`，这保证了 target 侧的一致性。✓

**论文符合性：✓ 正确**

---

### 四、IQL

**论文设计**：每个 agent 完全独立地运行 DQN，无任何中心化协调。

**实现路径**：
```
MultiAgentPolicy(shared=False)
  └── _policy_dict:
      agent_0: ValuePolicy(q_net_0)  # deepcopy
      agent_1: ValuePolicy(q_net_1)  # deepcopy
      ...

IQLAlgo.__init__:
  # 防御性检查：
  if getattr(policy, "shared", False):
      raise ValueError("IQLAlgo 不支持 shared_policy=True")

  target_policies: {aid: deepcopy(policy.get_policy(aid))}
  _optimizers:    {aid: Adam(policy.get_policy(aid).parameters())}
```

每 agent 完全独立的 Q-net、target、optimizer。

**发现的问题**：

❶ **`_build_policy` 中的 deepcopy 时机**：
```python
# MultiAgentPolicy.__init__ (shared=False 路径)
for aid in agent_ids:
    agent_kwargs = {
        k: copy.deepcopy(v) if isinstance(v, nn.Module) else v
        for k, v in policy_kwargs.items()
    }
    self._policy_dict[aid] = policy_class(..., **agent_kwargs)
```
`q_net`（nn.Module）被 deepcopy，每个 agent 得到参数相同但完全独立的网络。✓

❷ **epsilon 衰减**：IQL 的 epsilon 衰减逻辑有两种模式（`epsilon_decay_by_step` 和乘法衰减），但所有 agent 统一到同一 epsilon，不维护 per-agent epsilon。对于同构 agent 这没问题；若 agent 异构（探索需求不同）则需改进。

❸ **sync_replay 模式**：`sync_replay=True` 时 buffer 仅存 active（READY）步，通过 `gamma_power` 补偿折扣。这是 IQL 专属的跳步采样优化，逻辑独立且正确。

**论文符合性：✓ 正确**

---

### 五、VDPPO

**论文设计（Ma & Luo 2022）**：PPO actor + QPLEX 值分解 + 双优化器。

**实现路径**：
```
MultiAgentPolicy(shared=True)  ← actor 共享
  └── _shared_policy: ActorPolicy(actor=actor_net)

VDPPOAlgo:
  self.q_network      # 一个 Q-net，输入 (state_dim + n_agents)
  self.target_q_network  # deepcopy
  self.mixing_net     # QPLEXMixer(n_agents, state_dim, embed_dim)
  self.target_mixing_net  # deepcopy
  actor_optimizer → policy.parameters()
  q_optimizer → q_network.parameters() + mixing_net.parameters()
```

**Q-net 输入构建**（`prepare_batch` 和 `update`）：
```python
# 为第 i 个 agent 构建输入
one_hot = torch.zeros(total_size, num_agents, ...)
one_hot[:, i] = 1.0
q_in = torch.cat([global_state, one_hot], dim=-1)  # (T*N, state_dim + n_agents)
Q_i = self.q_network(q_in)  # 单一网络，靠 one-hot 区分 agent
```

✓ 单 Q-net 被所有 agent 共享，通过 one-hot 区分身份，是 VDPPO/QPLEX 的设计。

**QPLEXMixer 正确性**：
```python
# QPLEXMixer.forward
v_tot = v_vals.sum(dim=-1)                             # Σ V_i
lam = F.softplus(self.lambda_net(states)) + 1e-6       # λ_i > 0（严格正）
a_tot = (lam * a_vals).sum(dim=-1)                     # Σ λ_i * A_i
return v_tot + a_tot                                   # Q_tot = V_tot + A_tot
```

✓ softplus 保证 λ_i 严格正，满足 Advantage-IGM 约束。

**发现的问题**：

❶ **`batch.global_state` 字段被计算但未使用**：

```python
# prepare_batch 中
all_critic_input = []
for i in range(num_agents):
    one_hot = torch.zeros(total_size, num_agents, ...)
    one_hot[:, i] = 1.0
    all_critic_input.append(torch.cat([global_state, one_hot], dim=-1))

return RolloutBatch(
    ...
    global_state=torch.cat(all_critic_input, dim=0),  # ← 存入但…
    ...
)
```

```python
# update 中的 actor 更新
new_lp, entropy = self.policy.evaluate_actions_flat(
    mb.obs, mb.act, action_mask=mb.action_mask,  # ← 只用 obs，global_state 被忽略
)
```

`batch.global_state` 在 `RolloutBatch` 中存了每个 agent 的 `[state, one_hot]`，但 VDPPO 的 `update()` 完全不读它（actor 只用 obs，Q-network 的输入在 `update()` 里重新从 `q_data["states"]` 计算）。这些张量多占了 `N_agents × T × N × (state_dim + N_agents)` 的显存，**是资源浪费但不影响正确性**。

❷ **`use_value_norm` 设置了但 VDPPO 不使用**：

`ActorCriticOnPolicyAlgo.__init__` 在 `use_value_norm=True` 时会创建 `ret_rms`。若 VDPPOParams 含 `use_value_norm: true`（不建议），`ret_rms` 会被创建，但 `VDPPOAlgo.prepare_batch` 完全不调用任何归一化逻辑。该开关在 VDPPO 中是静默无效的。当前 VDPPO 实验 yaml 未配置此字段，所以没有实际影响。

❸ **TD target 中 `A_tgt_all = zeros`**：

```python
A_tgt_all = torch.zeros_like(V_tgt_all)
Q_tot_next = self.target_mixing_net(V_tgt_all, A_tgt_all, next_flat)
```

Target 侧将 A 设为 0，即 Q_tot_next = Σ V_i_target = Σ max_a Q_i(s', a)。这等价于在 target 计算中取 greedy 策略的价值，是正确的（greedy action 的 advantage = 0 by definition）。✓

**论文符合性：✓ 基本正确，但有显存浪费**

---

### 六、需要警惕的横切问题

| 问题 | 影响算法 | 严重性 |
|------|---------|-------|
| `shared_policy=false` 时 VDNAlgo 只优化 agent_0 的网络，其余 agent 停止学习 | VDN/QMIX | 高（配置错误时致命），当前 config 已正确设 `true` |
| `r_tot = rew_all.sum(dim=-1)` 在同质奖励环境下 N 倍放大 TD target | VDN/QMIX | 中（影响收敛，建议加 `reward_global` 开关区分） |
| VDPPO `batch.global_state` 存储了 `N_agents × T×N × (state_dim+N)` 张量但从未使用 | VDPPO | 低（显存浪费，不影响正确性） |
| VDPPO `use_value_norm=true` 静默无效 | VDPPO | 低（只要 yaml 不配置此项） |
| IQL `epsilon` 所有 agent 统一同一值，不支持 per-agent 差异化探索 | IQL | 低（同构场景无问题） |

---

**User**

与图片中描述的是否一致，有何区别？

---

**Cursor**

下面逐一对照图片描述与实际代码实现，梳理一致之处和关键差异。

---

## 对照分析

### VDN / QMIX

**图片描述**：
- VDN：物理上只有 1 个 Agent Network（接收 o_i + agent ID），Mixing 是纯 `torch.sum()`，无独立 Mixer 网络实体；+ 1 个 Target Agent Network
- QMIX：1 个 Agent Network + 1 个 Mixer Network（含超网络）+ 各自的 Target 网络，共 4 个

**实际实现**：

```python
# VDNAlgo.__init__
self.q_network = policy.get_policy(agent_ids[0]).q_network  # 共享
self.target_q_network = copy.deepcopy(self.q_network)
self.mixer = SumMixer(n_agents, state_dim)         # 额外实例化
self.target_mixer = SumMixer(n_agents, state_dim)  # 额外实例化

# Q-network 输入
q_all = self.q_network(obs_all.view(B * N, -1))   # 只有 obs，无 agent ID
```

**差异**：

| 项目 | 图片 | 实现 |
|------|------|------|
| Q-net 输入 | `obs + agent_ID（one-hot）` | **只有 `obs`，无 agent ID** |
| VDN Mixer | 无独立实体，纯 sum() 操作 | 实例化了 `SumMixer` + `target_mixer` 两个对象（无参数，但多占内存） |
| 网络实体数 | VDN: 2，QMIX: 4 | VDN: 4（含两个无参 SumMixer），QMIX: 4 ✓ |

**Q-net 缺少 agent ID 的影响**：若两个 agent 在同一节点观测完全相同，Q-net 输出也完全相同，无法区分 agent 身份。在 MASUP 中由于观测包含位置信息、两个 agent 同时在同节点的概率较低，实际影响有限，但严格来说与图片所描述的 "每 agent 输入专属 Agent ID" 不一致。

---

### VDPPO

图片描述的与实现差异最大，有三处根本性区别：

**差异 1：Critic/Q-net 输入——局部 obs vs 全局 state**

| 项目 | 图片 | 实现 |
|------|------|------|
| 值函数输入 | 局部观测 `o_i` + agent ID | **全局状态 `s`（global state）+ agent one-hot** |
| 语义 | Local Critic V_i(o_i) | Centralized Q_i(s, a_i)，CTDE 范式 |

```python
# train.py line 536: VDPPO 的 q_input_dim
q_input_dim = dims["state_dim"] + dims["num_agents"]   # 全局 state + agent one-hot
# 而非 obs_dim + n_agents（局部）

# prepare_batch 中
q_in = torch.cat([global_state, one_hot], dim=-1)      # 全局 state 作为输入
Q_i = self.q_network(q_in)
```

图片描述的是 IPPO-Critic 风格（去中心化，局部 obs），实现走的是 MAPPO-Critic 风格（中心化训练，全局 state）。论文原文（Ma & Luo 2022）本身也使用全局状态，因此**实现与原论文一致，是图片描述有所简化**。

---

**差异 2：是否需要 Target 网络**

| 项目 | 图片 | 实现 |
|------|------|------|
| Target 网络 | 图片明确说 "不需要额外的 Target Networks" | **有 target_q_network + target_mixing_net** |
| 训练范式 | 纯 On-policy（PPO 风格，用当前网络直接计算优势） | Q-network 用 TD learning（off-policy 风格），需要 target 稳定 |

```python
# VDPPOAlgo.__init__
self.target_q_network = copy.deepcopy(self.q_network)      # 存在
self.target_mixing_net = copy.deepcopy(self.mixing_net)    # 存在

# prepare_batch 中双 DQN 目标计算
greedy = self.q_network(q_in).argmax(dim=-1)               # online 选 action
Q_tgt = self.target_q_network(q_in)                        # target 评估 value
Q_tot_next = self.target_mixing_net(V_tgt_all, A_tgt_all, next_flat)  # target mixer
```

图片的逻辑是：VDPPO 继承 PPO 的同轨性质，Critic 直接用当前网络预测值计算 GAE，不需要固定 target。**但实现中 Q-network 是用 TD loss 单独训练的，这在本质上引入了 off-policy 的值函数学习**，所以 target 网络是必要的。

---

**差异 3：网络实例总数**

| 图片描述 | 实现 |
|---------|------|
| 3 个（Actor + Local Critic + Mixer） | **5 个（Actor + Q-net + target Q-net + QPLEXMixer + target mixer）** |

---

### 总结对照表

| 算法 | 项目 | 图片描述 | 实际实现 | 是否与原论文一致 |
|------|------|---------|---------|--------------|
| VDN | Q-net 输入 | obs + agent_ID | **只有 obs** | 实现不足（缺 agent ID） |
| VDN | Mixer 实体 | 无（纯 sum） | 有 SumMixer 对象（无参数） | 无害冗余 |
| QMIX | 网络数 | 4 | 4 ✓ | 一致 |
| QMIX | Q-net 输入 | obs + agent_ID | **只有 obs** | 实现不足 |
| VDPPO | 值函数输入 | 局部 obs + agent_ID | **全局 state + one-hot** | 实现=原论文，图片简化 |
| VDPPO | Target 网络 | 不需要 | **有 target Q-net + target mixer** | 实现=原论文（TD learning 必要），图片有误 |
| VDPPO | 网络数 | 3 | **5** | 实现=原论文 |

---

### 需要关注的潜在问题

**VDN/QMIX 缺少 agent ID**：这是与图片描述以及 PyMARL 等成熟实现的主要差异。在 MASUP 中由于各 agent 的局部观测本身已包含位置相关特征，不同 agent 通常观测不同，暂不影响训练。但若环境中存在多个 agent 观测完全相同的情况（如巡逻中多个 agent 聚集在同一节点），Q 值无法区分 agent 身份，可能导致训练不稳定。如果想与主流实现对齐，应在 Q-net 输入中拼接 agent one-hot。

