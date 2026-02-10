# WandB Sweep 使用指南

## 快速开始

### 1. 启动 Sweep（默认配置）

```bash
python sweep_mappo.py --project MAP-Sweep --count 50
```

### 2. 使用自定义配置文件

```bash
python sweep_mappo.py --project MAP-Sweep --count 50 --sweep-config configs/sweep_config.yaml
```

### 3. 指定预训练权重

```bash
python sweep_mappo.py --project MAP-Sweep --count 10 \
    --actor-path models/imi-pure-norm-random-init/imi_train__1770181266_final/policy.pt \
    --critic-path models/imi-pure-norm-random-init/imi_train__1770181266_final/critic.pt
```

### 4. 快速测试（短训练）

```bash
python sweep_mappo.py --project MAP-Sweep --count 5 --total-timesteps 1000000
```

## Sweep 方法

- `bayes`: 贝叶斯优化（推荐，高效）
- `random`: 随机搜索
- `grid`: 网格搜索（穷举所有组合）

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--project` | MAP-Sweep | WandB 项目名称 |
| `--count` | 50 | 运行次数 |
| `--method` | bayes | Sweep 方法 |
| `--sweep-config` | None | Sweep 配置文件路径 |
| `--actor-path` | "" | 预训练 Actor 权重 |
| `--critic-path` | "" | 预训练 Critic 权重 |
| `--total-timesteps` | 10000000 | 覆盖总训练步数 |

## 可调参数

以下参数在 Sweep 中自动调优：

- `actor_lr`: Actor 学习率 (1e-5 ~ 1e-3)
- `critic_lr`: Critic 学习率 (1e-4 ~ 1e-3)
- `clip_range`: PPO 裁剪范围 (0.1 ~ 0.4)
- `vf_coef`: 价值函数系数 (0.1 ~ 2.0)
- `ent_coef`: 熵系数 (0.0 ~ 0.3)
- `gae_lambda`: GAE lambda (0.9 ~ 1.0)
- `num_minibatches`: 小批次数量 (4, 8, 16, 32)
- `update_epochs`: 更新轮数 (3, 5, 10)
- `num_steps`: 每次采集步数 (1024, 2048, 4096)
- `gamma`: 折扣因子 (0.99 ~ 0.9999)

## 结果查看

训练结果自动上传到 WandB，可在网页端查看：

```bash
# 打开 WandB 项目页面
wandb dashboard
```

## 模型保存

每次 sweep 运行的模型保存在：
```
models/mappo-sweep-TSP12/{run_name}/final/
```

包括：
- `config.yaml`: 网络配置
- `policy.pt`: Actor 权重
- `critic.pt`: Critic 权重
