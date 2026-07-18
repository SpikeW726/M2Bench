# M2Bench

<p align="center">
  <img src="docs/assets/platform2.png" alt="M2Bench 平台架构" width="100%">
</p>

**面向加权图多机器人持久监测的统一基准框架**

[English](README.md) | [简体中文](README_zh-CN.md) | [使用文档](USER_GUIDE_zh-CN.md) | [论文](https://arxiv.org/abs/2605.09633v2)

M2Bench 是论文 *Minimizing Worst-Case Weighted Latency for Multi-Robot Persistent Monitoring: Theory and RL-Based Solutions* 的配套研究代码。项目为经典巡逻启发式、表格强化学习、深度强化学习和协作式多智能体强化学习提供统一的图仿真环境、训练流程与评估协议。

论文将我们提出的建模方法称为 **Tail Worst-case Latency-Optimizing MDP（TWLO-MDP）**。由于代码实现早于最终命名，相关 Python 模块、注册键和 YAML 路径仍使用历史标识 `masup`。本文档正文统一称其为 **TWLO**；命令和配置中的 `masup` 则按代码要求保留。

## 主要特性

- 在加权图上模拟连续行驶时间，同时支持 fixed-step 与 event-driven 推进方式。
- 统一计算带节点优先级权重的 IGI、AGI、IWI、WI，以及 TWLO 的暂态后最坏时延。
- 同时支持 PettingZoo 风格的多智能体环境与 Gymnasium joint 环境。
- 将单智能体 actor/value policy 复用于多机器人系统，并支持独立参数或参数共享的 policy mapping。
- 支持 on-policy、off-policy、循环网络、表格学习、图网络、模仿初始化与 W&B 超参数搜索。
- 对学习策略和启发式策略使用统一的评估、绘图和动画流程。

## 环境配置

参考环境面向 Linux x86-64、Python 3.10、PyTorch 2.6 和 CUDA 12.4。神经网络训练建议使用支持 CUDA 的 GPU；启发式与表格实验可以仅使用 CPU。

```bash
git clone https://github.com/SpikeW726/M2Bench.git
cd M2Bench
conda env create -f environment.yml
conda activate M2Bench
```

验证核心依赖：

```bash
python -c "import torch, gymnasium, pettingzoo, wandb; print(torch.__version__, torch.cuda.is_available())"
```

`environment.yml` 是推荐使用的可复现实验环境定义。`requirements-pip.txt` 是导出的包快照，其中可能包含与构建平台相关的路径，因此不建议将其作为主要安装入口。

普通训练可在 YAML 中设置 `track_wandb: false` 而不使用 Weights & Biases；运行 sweep 前则需要登录：

```bash
wandb login
```

## 快速开始

以下命令均应在项目根目录执行。`configs/` 中的 YAML 文件是完整的实验定义。

### 训练策略

```bash
python train.py configs/experiments/masup/mappo_masup_grid_a3.yaml
```

模型默认保存到 `models/<算法>-<环境>-<地图>/<时间戳>/`。无需修改 YAML 即可覆盖模型和自动评估结果的保存位置：

```bash
python train.py configs/experiments/masup/mappo_masup_grid_a3.yaml \
  --save-dir /path/to/checkpoint \
  --results-dir /path/to/results
```

仓库中的实验配置面向完整训练，部分配置包含数千万环境步。快速检查流程时，可复制一份配置并减小 `training.total_steps`、`training.num_envs` 和 `training.num_steps`。

### 评估学习策略

`--model` 应指向同时包含 `config.yaml` 与 `policy.pt` 的 checkpoint 目录：

```bash
python evaluators/test.py \
  --model models/mappo-masup-grid/<timestamp>/best \
  --env_config configs/eval/masup/masup_grid_a3.yaml \
  --num_episodes 10 \
  --no_show
```

默认将评估图与 CSV 数据写入 `evaluators/results/`；可使用 `--save_plot` 或 `--results-dir` 修改目标位置。

### 评估启发式策略

三个位置参数依次为地图、机器人数量和 episode 截断时刻：

```bash
python evaluators/heuristic_evaluator.py island 6 500 \
  --policy HCR \
  --init-positions 0 10 20 30 40 49 \
  --num_episodes 10 \
  --no_show
```

初始位置必须是地图中真实存在的节点 ID，而不是节点列表中的序号；省略 `--init-positions` 时将随机初始化。

### 超参数搜索

```bash
python sweep.py \
  --base-config configs/experiments/masup/mappo_masup_grid_a3.yaml \
  --sweep-config configs/sweep/masup/mappo_masup_grid_a3.yaml \
  --count 20
```

配置字段、评估选项、已实现方法及其参考文献和扩展流程详见[使用文档](USER_GUIDE_zh-CN.md)。

## 仓库结构

```text
algorithms/             RL、MARL 与表格优化算法
configs/                配置 dataclass、注册表和各类 YAML
data/                   rollout batch 与 replay buffer
envs/                   向量环境与巡逻 MDP
evaluators/             学习策略和启发式策略评估工具
graphs/                 加权监测图与可视化坐标
networks/               MLP、循环网络、图网络、MAT 与 TWLO 专用网络
policies/               可复用 RL policy 封装与启发式控制器
trainers/               on/off-policy、表格与模仿学习流程
utils/                  图、日志、checkpoint、上传与 sweep 工具
train.py                训练入口
sweep.py                Weights & Biases sweep 入口
run_all_heuristics.py   批量启发式评估入口
```

`configs/registry.py` 中的中央注册表将 YAML 标识映射到环境、网络、算法和 trainer。Checkpoint 同时保存网络重建信息与权重，因此 `evaluators/test.py` 无需原训练 YAML 即可恢复策略结构。

## 引用

使用 M2Bench 或 TWLO 建模方法时，请引用：

```bibtex
@article{wang2026minimizing,
  title={Minimizing Worst-Case Weighted Latency for Multi-Robot Persistent Monitoring: Theory and RL-Based Solutions},
  author={Wang, Weizhen and Wang, Ziheng and He, Jianping and Guan, Xinping and Duan, Xiaoming},
  journal={arXiv preprint arXiv:2605.09633},
  year={2026}
}
```

使用本仓库复现某个基线时，还应引用[已实现方法列表](USER_GUIDE_zh-CN.md#已实现的监测-mdp)中对应的原始工作。
