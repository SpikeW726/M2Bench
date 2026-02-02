# Value Normalization for MAPPO RL Training (Post-training Stage)

> **For Claude:** 精简版实现计划，仅包含核心功能修改。

**Goal:** 为 MAPPO RL 训练阶段添加 Value Normalization，支持从预训练 checkpoint 继承统计量。

**Architecture:** 使用 RunningMeanStd 增量更新统计量，在 prepare_batch 中反归一化 value 用于 GAE，训练时归一化 target。

---

## Overview

本计划在 **MAPPO RL 训练阶段** 添加 Value Normalization 功能，与第一部分（预训练阶段）衔接。

**关键设计**：
- 使用 `RunningMeanStd` 增量更新统计量
- 在 `prepare_batch` 中计算 value 后反归一化，用于 GAE 计算
- 在 `update` 中归一化 target 用于 Critic loss
- 支持从预训练 checkpoint 继承统计量

---

## Task 1: MAPPOAlgo.__init__ 添加参数和初始化

**File:** `algorithms/marl/mappo.py:24-51`

添加参数和 RunningMeanStd 初始化：

```python
def __init__(
    self,
    policy: MultiAgentPolicy,
    critic: nn.Module,
    num_envs: int,
    # ... 现有参数 ...
    # Value Normalization 参数
    use_value_norm: bool = False,
    value_norm_config: Optional[Dict] = None,  # 从预训练 checkpoint 传入
):
    # ... 现有初始化代码 ...

    # Value Normalization
    self.use_value_norm = use_value_norm
    self.ret_rms = None

    if self.use_value_norm:
        from utils.train_utils import RunningMeanStd
        self.ret_rms = RunningMeanStd(shape=(1,))

        # 从预训练 checkpoint 继承统计量
        if value_norm_config is not None:
            self.ret_rms.mean.fill_(value_norm_config.get('ret_mean', 0.0))
            self.ret_rms.var.fill_(value_norm_config.get('ret_std', 1.0) ** 2)
            self.ret_rms.count.fill_(1.0)  # 标记为已初始化
            print(f"[MAPPO] Loaded value_norm stats: mean={self.ret_rms.mean.item():.4f}, std={self.ret_rms.std.item():.4f}")
        else:
            print(f"[MAPPO] Initialized value_norm with default: mean=0, std=1")
```

---

## Task 2: MAPPOAlgo.prepare_batch 计算反归一化 value

**File:** `algorithms/marl/mappo.py:112` (在 values 计算后添加反归一化)

当前代码在第 112 行左右计算 values：
```python
with torch.no_grad():
    values = self.critic(critic_input).squeeze(-1).view(T, N)
```

修改为：
```python
with torch.no_grad():
    values_norm = self.critic(critic_input).squeeze(-1).view(T, N)

    # 反归一化 value 用于 GAE 计算
    if self.use_value_norm and self.ret_rms is not None:
        values = values_norm * self.ret_rms.std + self.ret_rms.mean
    else:
        values = values_norm
```

这样后续的 GAE 计算使用的是**真实的 value**。

---

## Task 3: MAPPOAlgo.prepare_batch 更新统计量

**File:** `algorithms/marl/mappo.py:145` (在 all_ret 计算完成后)

在 returns 聚合后，更新统计量：

```python
# Stack all returns: [num_agents, T, N] -> [T*N, num_agents] -> [T*N*num_agents]
all_ret_tensor = torch.stack([all_ret[i].view(-1) for i in range(len(all_ret))], dim=1).view(-1)

if self.use_value_norm and self.ret_rms is not None:
    self.ret_rms.update(all_ret_tensor)
    if self.verbose:  # 可选的打印
        print(f"[MAPPO] Updated value_norm: mean={self.ret_rms.mean.item():.4f}, std={self.ret_rms.std.item():.4f}")
```

---

## Task 4: MAPPOAlgo.update 归一化 Critic Target

**File:** `algorithms/marl/mappo.py:366-375`

修改 Critic loss 计算：

```python
# 原代码:
new_value = self.critic(mb_critic_input).squeeze(-1)
if self.clip_vloss:
    v_loss_unclipped = (new_value - mb_ret) ** 2
    # ...
else:
    v_loss = 0.5 * ((new_value - mb_ret) ** 2).mean()

# 修改为:
new_value = self.critic(mb_critic_input).squeeze(-1)

if self.use_value_norm and self.ret_rms is not None:
    # 归一化 target
    target_norm = (mb_ret - self.ret_rms.mean) / (self.ret_rms.std + 1e-8)
    target = target_norm
else:
    target = mb_ret

if self.clip_vloss:
    v_loss_unclipped = (new_value - target) ** 2
    v_clipped = mb_value + torch.clamp(new_value - mb_value, -self.clip_range, self.clip_range)
    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
else:
    v_loss = 0.5 * ((new_value - target) ** 2).mean()
```

---

## Task 5: train_mappo.py 传递 value_norm_config

**File:** `train_mappo.py:153-201`

在加载预训练模型后，读取 value_normalization 配置：

```python
# 在加载 actor/critic 权重后（约第 160 行之后）
value_norm_config = None
if actor_path is not None and os.path.exists(actor_path):
    # 尝试读取 config.yaml 中的 value_normalization
    config_dir = os.path.dirname(actor_path)
    config_file = os.path.join(config_dir, 'config.yaml')

    if os.path.exists(config_file):
        with open(config_file) as f:
            saved_config = yaml.safe_load(f)

        if saved_config.get('value_normalization') is not None:
            value_norm_config = saved_config['value_normalization']
            print(f"[Main] Loaded value_norm config from checkpoint: {value_norm_config}")

# 传入 algorithm
algorithm = MAPPOAlgo(
    # ... 现有参数 ...
    use_value_norm=value_norm_config is not None,  # 自动启用如果有配置
    value_norm_config=value_norm_config,
)
```

---

## Task 6: 更新配置示例

**File:** `train_mappo.py:46-69`

添加 `use_value_norm` 参数：

```python
# Training hyperparams
total_timesteps = 50_000_000
# ... 现有参数 ...

# Value Normalization config
use_value_norm = True  # 是否启用 Value Normalization
```

---

## Verification

运行训练验证：

```bash
python train_mappo.py
```

检查点：
- [ ] 如果有预训练 checkpoint，打印 "Loaded value_norm stats"
- [ ] 训练正常完成，没有 NaN
- [ ] TensorBoard/wandb 中 critic_loss 曲线正常

---

## Summary

**修改文件**: 2 个
- `algorithms/marl/mappo.py`: 添加参数、反归一化 value、更新统计量、归一化 target
- `train_mappo.py`: 读取并传递 value_norm_config

**修改位置**: 5 处
- MAPPOAlgo.__init__: 参数和初始化
- MAPPOAlgo.prepare_batch: 反归一化 value、更新统计量
- MAPPOAlgo.update: 归一化 target
- train_mappo.py: 读取配置、传递参数

**预计时间**: 30 分钟
