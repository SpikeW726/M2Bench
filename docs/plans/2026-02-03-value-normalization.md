# Value Normalization Implementation Plan (Pre-training Stage)

> **For Claude:** 精简版实现计划，仅包含核心功能修改。

**Goal:** 为 ImiTrainer 添加 Value Normalization，支持 Critic 网络的归一化目标训练。

**Architecture:** 引入 `use_value_norm` 配置开关，在数据加载阶段计算统计量，在训练时归一化 target，保存时持久化统计量。

---

## Overview

本计划仅在 **监督学习预训练阶段** 添加 Value Normalization 功能。

**关键设计**：
- 使用配置开关 `use_value_norm`，False 时保持原有行为
- 仅针对 Critic 的 Target (Return) 进行归一化
- 统计量保存到 checkpoint，供后续 RL 阶段使用

---

## Task 1: 添加配置参数到 ImiTrainer.__init__

**File:** `trainers/imitator/imitation_trainer.py:87-103`

在第 103 行（actor_stopped 之后）添加：

```python
# Value Normalization 配置
self.use_value_norm = kwargs.get("use_value_norm", False)
self.ret_mean = 0.0
self.ret_std = 1.0
```

---

## Task 2: 数据加载后计算统计量

**File:** `trainers/imitator/imitation_trainer.py:169-172`

在 returns 统计信息打印后添加：

```python
# 打印 returns 统计信息
returns_data = flattened_data["returns"]
print(f"[ImiTrainer] Returns range: [{returns_data.min():.4f}, {returns_data.max():.4f}], "
      f"mean={returns_data.mean():.2f}, std={returns_data.std():.2f}")

# 计算 Value Normalization 统计量
if self.use_value_norm:
    self.ret_mean = float(returns_data.mean())
    self.ret_std = float(returns_data.std())
    print(f"[ImiTrainer] Value Normalization: mean={self.ret_mean:.4f}, std={self.ret_std:.4f}")
```

---

## Task 3: 修改 Critic Loss 计算

**File:** `trainers/imitator/imitation_trainer.py:213-218`

将原有代码：
```python
critic_pred = self.critic(critic_states_batch)
critic_loss = nn.functional.mse_loss(critic_pred, returns_batch)
```

修改为：
```python
critic_pred = self.critic(critic_states_batch)

if self.use_value_norm:
    target_norm = (returns_batch - self.ret_mean) / (self.ret_std + 1e-8)
    critic_loss = nn.functional.mse_loss(critic_pred, target_norm)
else:
    critic_loss = nn.functional.mse_loss(critic_pred, returns_batch)
```

---

## Task 4: 修改 _save_checkpoint 保存统计量

**File:** `trainers/imitator/imitation_trainer.py:328-338`

在 config 字典中添加 `value_normalization` 字段：

```python
config = {
    'actor': {...},
    'critic': {...},
    'extra': {...},
    # 新增
    'value_normalization': {
        'use_value_norm': self.use_value_norm,
        'ret_mean': self.ret_mean,
        'ret_std': self.ret_std,
    } if self.use_value_norm else None,
}
```

---

## Task 5: 修改 _save_actor 保存统计量

**File:** `trainers/imitator/imitation_trainer.py:288`

在 _save_actor 的 config 字典中同样添加 `value_normalization` 字段（同 Task 4）。

---

## Task 6: 更新 __main__ 配置示例

**File:** `trainers/imitator/imitation_trainer.py:384`

在 config 字典中添加：

```python
config = {
    # ... 现有配置 ...
    "use_value_norm": True,  # Value Normalization 开关
}
```

---

## Verification

运行一次训练验证：

```bash
python trainers/imitator/imitation_trainer.py
```

检查点：
- [ ] 打印显示 `Value Normalization: mean=xxx, std=xxx`
- [ ] 训练正常完成
- [ ] `models/pure/*_final/config.yaml` 包含 `value_normalization` 字段

查看保存的配置：
```bash
cat models/pure/*_final/config.yaml
```

---

## Summary

**修改文件**: 1 个 (`trainers/imitator/imitation_trainer.py`)

**修改位置**: 6 处
- `__init__`: 添加参数
- `train()`: 计算统计量、修改 loss
- `_save_checkpoint()`: 保存统计量
- `_save_actor()`: 保存统计量
- `__main__`: 更新配置示例

**预计时间**: 25-30 分钟
