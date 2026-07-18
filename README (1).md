# 多智能体巡逻强化学习平台

本项目是一个基于强化学习的多智能体巡逻仿真平台，支持传统分布式算法和MARLlib深度强化学习算法的训练与评估。该平台专门针对图环境中的多智能体巡逻问题设计，提供了完整的实验框架和可视化工具。

## ✨ 核心特性

- 🏗️ **双框架支持**：同时支持传统分布式RL算法（Q-Learning、DQN）和MARLlib深度强化学习算法（MAPPO、VDPPO等）
- 🎯 **丰富算法库**：集成了GBLA、BBLA、S4R1、OUCS等经典巡逻mdp环境，以及多种启发式、RL算法
- 📊 **标准化环境**：基于PettingZoo标准的多智能体环境接口，支持动作掩码和全局状态
- 🔧 **模块化设计**：环境、智能体、训练器完全解耦，易于扩展新算法和新环境
- 📈 **完整评估**：集成训练、评估、可视化完整流程，支持性能曲线和轨迹动画生成
- ⚙️ **灵活配置**：YAML配置系统支持复杂实验设置，参数化程度高
- 🎮 **强制合作**：支持VDPPO等价值分解算法的全局奖励聚合机制

## 🚀 环境配置

### 系统要求
- Python 3.8+
- CUDA支持的GPU
- Linux/macOS

### 安装步骤

1. **创建conda环境**
```bash
conda create -n MAP python=3.8
conda activate MAP
```

2. **安装MARLlib**
```bash
# 如果需要使用深度强化学习算法
cd /path/to/your/workspace
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
pip install -e .
```

2. **安装基础依赖**
```bash
待更新
```

## 📁 项目结构

```
MultiAgentPatrolling/
├── main.py                      # 统一启动脚本
├── config/                      # 配置文件目录
│   ├── MARLlib_MAPPO_*.yaml    # MARLlib深度强化学习配置
│   ├── MARLlib_VDPPO_*.yaml    # 价值分解算法配置
│   ├── *_config.yaml           # 传统分布式算法配置
│   └── ...
├── envs/                        # 环境模块
│   ├── base_env.py             # 环境基类（PettingZoo标准）
│   ├── graph_utils.py          # 图结构工具
│   ├── mdp/                    # MDP环境实现
│   │   ├── GBLA_env.py         # GBLA算法环境
│   │   ├── BBLA_env.py         # BBLA算法环境
│   │   ├── S4R1_env.py         # S4R1算法环境
│   │   ├── OUCS_env.py         # OUCS算法环境
│   │   └── ...                 # 其他启发式算法环境
│   ├── fcoop/                  # 强制合作环境
│   │   └── s4r1_fcoop.py       # S4R1强制合作版本
│   └── custom/                 # 自定义环境示例
│       └── simple_grid_env.py  # 简单网格环境示例
├── agent/                       # 智能体模块
│   ├── base_agent.py           # 智能体基类
│   ├── q_learning_agent.py     # Q学习智能体
│   └── DQN_agent.py            # DQN智能体
├── trainer/                     # 训练器模块
│   ├── decentralized_trainer.py # 传统分布式训练器
│   └── marllib_trainer.py       # MARLlib训练器
├── evaluation/                  # 评估器模块
│   ├── decentralized_evaluator.py # 传统分布式评估器
│   ├── marllib_evaluator.py     # MARLlib评估器
│   └── visualize.py            # 可视化工具
├── utils/                       # 工具模块
│   ├── fcoop_registry.py       # 强制合作环境注册
│   └── ...
├── graphs/                      # 图拓扑文件
│   ├── essay_MapB.json         # 示例图结构
│   └── ...
├── log/                         # 实验输出目录
├── networks/                    # 神经网络模块（用于DQN等）
└── requirements.txt             # Python依赖列表
```

## 🎮 快速开始

### 1. 传统分布式算法训练

使用Q-Learning等传统算法：

```bash
# GBLA环境Q-Learning训练
python main.py --config config/GBLA_config.yaml --framework decentralized

# S4R1环境Q-Learning训练  
python main.py --config config/S4R1_config.yaml --framework decentralized

# 仅运行评估
python main.py --config config/GBLA_config.yaml --framework decentralized --eval-only
```

### 2. MARLlib深度强化学习训练

使用MAPPO、VDPPO等深度强化学习算法：

```bash
# MAPPO算法训练GBLA环境
python main.py --config config/MARLlib_MAPPO_GBLA_config.yaml --framework marllib

# VDPPO算法训练S4R1环境（价值分解）
python main.py --config config/MARLlib_VDPPO_S4R1_config.yaml --framework marllib

# 仅运行评估
python main.py --config config/MARLlib_MAPPO_GBLA_config.yaml --framework marllib --eval-only
```

### 3. 查看结果

所有实验结果保存在`log/`目录下，包括：
- 训练曲线图表
- 评估性能报告
- 智能体轨迹动画
- 检查点文件

## 🔧 支持的算法

### 巡逻算法
- **GBLA/BBLA**：基于图的强化学习算法
- **S4R1**：深度强化学习巡逻算法  
- **OUCS**：城市犯罪监控优化算法
- **启发式算法**：CBLS、Conscientious、DTA、GBS、MSP、SEBS等

### 深度强化学习算法
通过MARLlib支持MAPPO、VDPPO、QMIX、VDN等主流多智能体算法。

详细算法介绍和配置方法请参考[USER_GUIDE.md](USER_GUIDE.md)。

## 📈 实验输出

每次实验会在`log/`目录下生成：

1. **训练结果**
   - `training_curve.png` - 训练收敛曲线
   - `checkpoint_*` - 模型检查点文件
   - `training_log.txt` - 详细训练日志

2. **评估结果**
   - `evaluation_report.txt` - 评估性能报告
   - `evaluation_idleness_combined.png` - 性能对比图表
   - `evaluation_results.pkl` - 详细评估数据
   - `*_animation.mp4` - 智能体轨迹动画

3. **可视化内容**
   - 节点空闲度变化曲线
   - 智能体移动轨迹
   - 性能指标统计

## 📚 文档

- **[USER_GUIDE.md](USER_GUIDE.md)** - 详细使用指南，包含环境扩展、算法添加、参数配置等

## 🙏 致谢

- [MARLlib](https://github.com/Replicable-MARL/MARLlib) - 多智能体强化学习库
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) - 多智能体环境标准
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) - 分布式强化学习框架