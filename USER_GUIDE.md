# 多智能体巡逻平台用户指南

本指南详细说明了如何使用多智能体巡逻平台进行实验，包括项目使用方法、环境扩展、算法添加和参数配置等内容。

## 目录

- [项目使用方法](#项目使用方法)
- [如何添加新的自定义MDP](#如何添加新的自定义mdp)
- [如何添加强制合作环境](#如何添加强制合作环境)
- [如何添加新的算法](#如何添加新的算法)
- [参数文件配置详解](#参数文件配置详解)
- [MDP算法详解](#mdp算法详解)
- [故障排除](#故障排除)

## 项目结构

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
│   ├── fcoop/                  # 强制合作环境
│   └── custom/                 # 自定义环境示例
├── trainer/                     # 训练器模块
│   ├── decentralized_trainer.py # 传统分布式训练器
│   └── marllib_trainer.py       # MARLlib训练器
├── evaluation/                  # 评估器模块
│   ├── decentralized_evaluator.py # 传统分布式评估器
│   └── marllib_evaluator.py     # MARLlib评估器
├── log/                         # 统一实验输出目录
└── ...
```

---

## 项目使用方法

### 基本工作流程

本项目支持两种训练框架：

1. **传统分布式框架**：使用Q-Learning、DQN等传统强化学习算法
2. **MARLlib框架**：使用MAPPO、VDPPO等深度强化学习算法

### 命令行参数

```bash
python main.py [选项]
```

**可用参数：**
- `--config`: 配置文件路径（必需）
- `--framework`: 强制指定训练框架（`decentralized` 或 `marllib`）
- `--eval-only`: 仅运行评估，跳过训练
- `--checkpoint`: MARLlib检查点路径（用于评估）

### 使用示例

#### 1. 传统分布式算法

```bash
# Q-Learning训练GBLA环境
python main.py --config config/GBLA_config.yaml --framework decentralized

# 仅评估已训练的模型
python main.py --config config/GBLA_config.yaml --framework decentralized --eval-only

# S4R1环境训练
python main.py --config config/S4R1_config.yaml --framework decentralized
```

#### 2. MARLlib深度强化学习

```bash
# MAPPO训练GBLA环境
python main.py --config config/MARLlib_MAPPO_GBLA_config.yaml --framework marllib

# VDPPO训练S4R1环境（价值分解）
python main.py --config config/MARLlib_VDPPO_S4R1_config.yaml --framework marllib

# 使用特定检查点评估
python main.py --config config/MARLlib_MAPPO_GBLA_config.yaml --framework marllib --eval-only --checkpoint /path/to/checkpoint
```

### 实验输出结构

```
log/
├── GBLA_exp1/                   # 传统分布式实验输出
│   ├── agents_final.pkl         # 训练好的智能体模型
│   ├── training_curve.png       # 训练曲线
│   ├── evaluation_report.txt    # 评估报告
│   └── evaluation_animation.mp4 # 轨迹动画
└── MAPPO_GBLA_test/            # MARLlib实验输出
    ├── mappo_mlp_GBLA/         # 算法特定目录
    │   └── checkpoint_*        # 检查点文件
    ├── evaluation_report.txt   # 评估报告
    └── evaluation_idleness_combined.png # 性能图表
```

---

## 如何添加新的自定义MDP

### 步骤1：创建环境类

在`envs/mdp/`目录下创建新的环境文件：

```python
# envs/mdp/my_custom_env.py
#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple
import numpy as np
from gym.spaces import Box, Discrete

from agent.base_agent import BaseAgent
from envs.base_env import PatrolEnvBase
from envs.graph_utils import Graph


class MyCustomEnv(PatrolEnvBase):
    """自定义MDP环境实现"""
    
    def __init__(self, config: Dict, agents: List[BaseAgent]):
        # 计算环境特定参数
        graph = Graph(config['env_config']['graph_path'])
        self.max_neighbors = graph.get_max_degree()
        
        # 调用父类初始化
        super().__init__(config, agents)
        
        # 初始化环境特定状态
        self.custom_state = {}
        
        # 初始化空间
        self._init_spaces()
    
    def observation_space(self, agent):
        """定义观测空间"""
        max_node_id = max(self.graph.nodes)
        self.obs_size = 4  # 根据你的观测向量大小调整
        
        low = np.array([-1] * self.obs_size, dtype=np.int32)
        high = np.array([max_node_id] * self.obs_size, dtype=np.int32)
        
        return Box(low=low, high=high, dtype=np.int32)

    def action_space(self, agent):
        """定义动作空间"""
        return Discrete(self.max_neighbors + 1)

    def _get_max_action_space_size(self) -> int:
        return self.max_neighbors + 1
    
    def _build_observation(self, agent_id: int) -> np.ndarray:
        """构建观测向量"""
        if self.agents_on_edge[agent_id]:
            return np.full(self.obs_size, -1, dtype=np.int32)
        
        current_pos = self.agent_positions[agent_id]
        obs = np.full(self.obs_size, -1, dtype=np.int32)
        obs[0] = current_pos
        # 添加其他观测特征...
        
        return obs
    
    def _extract_state_key(self, observation: np.ndarray) -> Optional[tuple]:
        """从观测中提取状态键（用于Q-Learning）"""
        if np.all(observation == -1):
            return None
        
        state_components = []
        for i in range(len(observation)):
            state_components.append(int(observation[i]) if observation[i] != -1 else -1)
        
        return tuple(state_components)
    
    # 算法特定方法
    def _reset_algorithm_state(self):
        self.custom_state = {}
    
    def _handle_action_set(self, agent_id: int, target_node: int):
        pass
    
    def _handle_arrival_events(self, arrived_this_step: Dict[int, int]):
        pass
    
    def _get_algorithm_global_state(self) -> List[float]:
        return []
```

### 步骤2：在main.py中注册环境

```python
def create_env(config: Dict, agents: List[BaseAgent]) -> PatrolEnvBase:
    mdp_type = config['env_config']['mdp_type']
    
    if mdp_type == 'MyCustom':  # 添加你的环境
        from envs.mdp.my_custom_env import MyCustomEnv
        return MyCustomEnv(config, agents)
    # ... 其他环境
```

### 步骤3：创建配置文件

#### 传统分布式配置
```yaml
# config/MyCustom_config.yaml
env_config:
  graph_path: "graphs/essay_MapB.json"
  num_agents: 4
  mdp_type: "MyCustom"

train_config:
  algorithm: "Q-Learning"
  paradigm: "decentralized"
  reward_method: "Node-Idleness"
  num_episodes: 1000

log_config:
  log_dir: "log/MyCustom_exp1"
```

#### MARLlib配置
```yaml
# config/MARLlib_MAPPO_MyCustom_config.yaml
experiment:
  name: "MyCustom_MAPPO"
  suffix: "test"

environment:
  type: "MyCustom"
  graph_path: "graphs/essay_MapB.json"
  num_agents: 4

algorithm:
  name: "MAPPO"
  framework: "marllib"

training:
  episodes: 10
```

### 步骤4：测试环境

```bash
# 传统分布式测试
python main.py --config config/MyCustom_config.yaml --framework decentralized

# MARLlib测试
python main.py --config config/MARLlib_MAPPO_MyCustom_config.yaml --framework marllib
```

### 环境开发要点

- **继承PatrolEnvBase**：确保兼容项目框架
- **实现必需方法**：`observation_space()`, `action_space()`, `_build_observation()`, `_extract_state_key()`
- **算法特定方法**：根据需要实现`_reset_algorithm_state()`, `_handle_action_set()`, `_handle_arrival_events()`
- **观测空间设计**：考虑动作掩码支持，使用gym.spaces定义

## 命令行参数

- `--config`: 配置文件路径
- `--framework`: 强制指定训练框架 (`decentralized` 或 `marllib`)
- `--eval-only`: 仅运行评估，跳过训练
- `--checkpoint`: MARLlib检查点路径（用于评估）

## MDP详解
*除特殊说明的mdp以外，其余mdp动作空间均为邻接节点索引*
### BBLA/GBLA/Extended-GBLA
**原文链接:**
1. [Multi-agent patrolling with reinforcement learning](https://jmvidal.cse.sc.edu/library/santana04a.pdf) 
2. [Robust Multi-agent Patrolling Strategies Using Reinforcement Learning](https://link.springer.com/chapter/10.1007/978-3-319-12970-9_17)

**观测空间:**
 - BBLA: 智能体所处的节点、智能体从哪条边到达该节点、邻接节点中Idleness最大+最小的节点
 - GBLA: 在BBLA的基础上增加邻居节点的“意图”，即有没有其他智能体的目标为该节点，有的话该节点“意图”记录为1，没有的话记录为0
 - Extended-GBLA: 同GBLA

**奖励函数:**
 - BBLA/GBLA: 在时刻t，如果智能体恰好到达一节点，则奖励函数为该节点该时刻的 Instantaneous node idleness；如果在边上运动，则奖励为0
 - Extended-GBLA: 在GBLA的基础上增加一个惩罚机制，如果智能体A到达一节点时，智能体B正在朝该节点运动，则智能体A奖励为0

**原文采用算法:** Q-table
**算法对应文件:** `agent/q_learning_agent.py`, `trainer/decentralized_trainer.py`

**特殊说明:**
1. 以上三种MDP均未考虑节点权重
2. 以上三种MDP当智能体在边上运动时都会构建特殊观测，即所有维度全部用 -1 填满
3. 可以通过修改参数在以上三种MDP中使用 Penalised Idleness 作为奖励函数
4. 可以在以上三种MDP中使用 marllib 框架并调用其提供的各种算法进行实验

### S4R1
**原文链接**：
[A deep reinforcement learning approach for multi‐agent mobile robot patrolling](https://link.springer.com/article/10.1007/s41315-022-00235-1)

**观测空间**：所有智能体的出发节点和目标节点、自己目标节点的邻接节点的空闲度

**奖励函数**：Penalised Idleness $r=\frac{INI(t,\nu)^p}{IGI(t)}$

**原文采用算法:** DQN
**算法对应文件:** `agent/DQN_agent.py`, `trainer/decentralized_trainer.py`

**原文采用神经网络:** MLP
**神经网络对应文件:** `networks/DQN_S4R1.py`

**特殊说明:**
1. 此MDP未考虑节点权重
2. 在此MDP中可以使用 marllib 框架并调用其提供的各种算法进行实验

### OUCS
**原文链接**：
[Cooperative Patrol Routing:  Optimizing Urban Crime Surveillance through Multi-Agent  Reinforcement Learning](https://arxiv.org/abs/2501.08020)

**观测空间**：Position of each agent, Number of visits agents have made to neighbor nodes, Weight of neighbor nodes
         
**奖励函数:** 同GBLA 或 $r=\frac{INI(t,\nu)^p}{IGI(t)}$ 二者任选

**原文采用算法:** MAPPO/VDPPO/IPPO/MATRPO/VDA2C
**算法对应文件:** `MARLlib/marllib/marl/algos`

**原文采用神经网络:** RNN(GRU)
**神经网络对应文件:** `MARLlib/marllib/marl/models/zoo`

**特殊说明:**
1. 原文的源码中 line of sight 被定义为一个以智能体当前位置为中心、边长为 2 * line_of_sight + 1 的正方形观测区域，但是由于本项目中并不采取原文的网格化环境，因此简化为智能体只能观察到当前位置邻接节点的信息
2. 原文的 metric 不是 idleness，因此奖励函数直接选用前面几个MDP中所使用的

### GAT-PPO
**原文链接**:
[Balancing Efficiency and Unpredictability in Multi-robot Patrolling: A MARL-Based Approach](https://ieeexplore.ieee.org/document/10160923)

**观测空间**：观测为多维尺度变换后的节点坐标以及节点空闲度

**奖励函数**： $r=\frac{INI(t,\nu)^p}{IGI(t)}$ 

**原文采用算法**：PPO
**算法对应文件:**：`agent/gat_ppo_agent.py`, `trainer/marl_ppo_trainer.py`

**原文采用神经网络:** GAT
**神经网络对应文件:** `networks/gat_ppo_network.py`

**特殊说明:**
1. 原文使用欧式图，因此对仿真器中的拓扑图做多维尺度变换赋予节点坐标
2. 此MDP未考虑节点权重


### GBS
**原文链接**：[Distributed multi-robot patrol: A scalable and fault-tolerant framework - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0921889013001206)

**观测空间**：智能体当前所在节点，邻接节点空闲度(按当前顶点的邻居集合排序依次填入，不足 max_neighbors 的位置保留为 0),全局平均空闲度。

**奖励函数**：奖励定义为对基类奖励的线性增益修正： $r = base_{reward} + G_1 I_v + G_2 (I_v - \overline{I})$ 


**原文采用算法**：GBS
**算法对应文件:**：`agent/gbs_agent.py`


### SEBS
**原文链接**：[Scalable, fault-tolerant and distributed multi-robot patrol in real world environments | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/6697042)

**观测空间**：与SEBS一样

**动作空间**：设定动作时环境会将各智能体的目标节点广播给所有智能体，用于智能体内部避让达到协调的目的

**奖励函数**：与SEBS一样

**原文采用算法**：SEBS
**算法对应文件:**：`agent/sebs_agent.py`

### CBLS
**原文链接**：[Cooperative multi-robot patrol with Bayesian learning | Autonomous Robots](https://link.springer.com/article/10.1007/s10514-015-9503-7)

**观测空间**：与GBS一样

**奖励函数**：在基类奖励 base_reward 的基础上加入“本地空闲度增益”与“行进距离惩罚”

**原文采用算法**：CBLS
**算法对应文件:**：`agent/cbls_agent.py`

### DTAP/DTAG
**原文链接**：[Distributed on-line dynamic task assignment for multi-robot patrolling | Autonomous Robots](https://link.springer.com/article/10.1007/s10514-016-9579-8)

**观测空间**：观测为长度为max_neighbors+3+agent_number的向量，包括所有邻居的空闲度,当前位置编号，邻居tabu标记（0/1），全局平均空闲度，其他智能体的位置

**奖励函数**：$ r = base_{reward} + G_1 I_v - G_2 [\,v \in tabu\,] + G_3 (I_v - \overline{I})$


**原文采用算法**：DTAP/DTAG
**算法对应文件:**：`agent/dtap_agent.py`, `agent/dtag_agent.py`


## 参数文件配置详解

### MARLlib配置文件结构

```yaml
# 实验基本信息
experiment:
  name: "GBLA_MAPPO"        # 实验名称，用于日志目录命名
  suffix: "test_exp"        # 实验后缀，区分同一算法的不同实验

# 环境配置
environment:
  type: "GBLA"              # 环境类型：GBLA/BBLA/S4R1/OUCS
  graph_path: "graphs/essay_MapB.json"  # 相对于项目根目录的图文件路径
  num_agents: 4             # 智能体数量
  reward_method: "Node-Idleness"  # 奖励方法：Node-Idleness/Extended-Node-Idleness
  time_discrete: true       # 时间步类型：true=离散，false=连续
  agent_speeds: null        # 智能体速度：null=默认1.0，或{agent_id: speed}字典
  episode_length: 100       # 每个episode的最大步数

# MARLlib环境接口标志
marllib_env_flags:
  mask_flag: true           # 启用动作掩码（推荐true）
  global_state_flag: false  # 全局状态标志：true=调用env.state()，false=拼接obs
  opp_action_in_cc: false   # 是否在centralized critic中包含对手动作
  agent_level_batch_update: true  # 智能体级别批量更新
  force_coop: false         # 强制合作：VDPPO等价值分解算法需要设为true

# 算法配置
algorithm:
  name: "MAPPO"             # 算法名称：MAPPO/VDPPO/QMIX/VDN/MAA2C/MADDPG等
  framework: "marllib"      # 固定为marllib
  paradigm: "centralized_training_decentralized_execution"  # 训练范式
  architecture:
    core_arch: "mlp"        # 网络架构：mlp/gru/lstm
    encode_layer: "128-256" # 编码层结构
    hyperparam_source: "common"  # 超参数来源

# 训练配置
training:
  episodes: 10              # 训练episode数（用于早停）
  max_timesteps: 40000000   # 最大时间步数
  batch_size: 1000          # 批次大小
  rollout_fragment_length: 100  # rollout片段长度
  checkpoint_freq: 100      # 检查点保存频率
  share_policy: "all"       # 策略共享：all/group/individual

# 系统配置
system:
  local_mode: false         # 本地模式：true=单进程调试，false=分布式
  num_workers: 1            # 工作进程数
  num_gpus: 1              # GPU数量
  num_cpus_per_worker: 1   # 每个worker的CPU数
  object_store_memory_gb: 1 # 对象存储内存限制
  log_level: "INFO"         # 日志级别：DEBUG/INFO/WARNING/ERROR
  metrics_smoothing_episodes: 10  # 指标平滑窗口

# 评估配置
evaluation:
  episodes: 1               # 评估episode数
  steps: 100               # 每个episode的步数
  save_batch_freq: 1       # 批次保存频率
  animation_max_frames: 1000  # 动画最大帧数
  quick_preview: false     # 快速预览模式
  curve_title: "MARLlib MAPPO Training Progress on GBLA"  # 训练曲线图标题
```

### 传统分布式配置文件结构

```yaml
# 环境配置
env_config:
  graph_path: "graphs/essay_MapB.json"  # 图结构文件路径
  num_agents: 10           # 智能体数量
  mdp_type: "GBLA"         # 环境类型：GBLA/BBLA/S4R1等
  time_discrete: true      # 时间步类型：true=离散，false=连续
  agent_speeds: null       # 智能体速度配置：null=使用默认速度1.0，或指定{agent_id: speed}

# 智能体配置
agent_config:
  gamma: 0.9               # 折扣因子
  epsilon: 0.1             # 探索率（epsilon-greedy）
  
  # --- Learning Rate (Alpha) Configuration ---
  # 学习率策略选择
  # 可用策略:
  # - 'fixed': 使用固定学习率，由'fixed_alpha'指定
  # - 'visit_decay_v1': 基于访问次数衰减，公式: 1 / (c1 + visits / c2)
  # - 'visit_decay_v2': 简单访问衰减，公式: 1 / (1 + visits)
  alpha_strategy: 'visit_decay_v1'
  
  # --- 策略特定参数 ---
  # 仅在alpha_strategy为'fixed'时使用
  fixed_alpha: 0.1
  
  # 仅在alpha_strategy为'visit_decay_v1'时使用
  alpha_decay_c1: 2.0
  alpha_decay_c2: 15.0

  time_discrete: true      # 与env_config保持一致

# 训练配置
train_config:
  algorithm: "Q-Learning"   # 算法类型：Q-Learning/DQN
  paradigm: "decentralized" # 训练范式：decentralized
  reward_method: "Node-Idleness"  # 奖励方法：Node-Idleness, Extended-Node-Idleness, Penalised-Idleness
  num_episodes: 1000       # 训练episode数
  episode_len: 10000       # 每个episode最大步数

# 日志配置
log_config:
  log_dir: "log/GBLA_exp1" # 实验输出目录
  save_model_interval: 200 # 每200轮保存一次模型

# 评估配置
eval_config:
  evaluation_steps: 50000   # 评估总步数
  quick_preview: true       # 快速预览模式
```

### 参数详解

#### 奖励方法说明

- **Node-Idleness**: 标准节点空闲度奖励，智能体到达节点时获得该节点的即时空闲度作为奖励
- **Extended-Node-Idleness**: 扩展GBLA奖励，如果智能体到达节点时有其他智能体正在前往该节点，则奖励为0
- **Penalised-Idleness**: 惩罚空闲度，使用公式 r = INI(t,v)^p / IGI(t)

#### 时间步类型

- **time_discrete=true**: 离散时间步，智能体每步移动一个单位距离
- **time_discrete=false**: 连续时间步，考虑边权重和智能体速度

#### 策略共享模式

- **share_policy="all"**: 所有智能体共享同一个策略
- **share_policy="group"**: 按组共享策略
- **share_policy="individual"**: 每个智能体独立策略

#### 热启动功能

对于使用深度强化学习算法（如APO）的训练，可以通过指定预训练权重文件来实现热启动：

```yaml
train_config:
  pretrained_path: "path/to/pretrained/model.pt"  # 预训练权重文件路径
```

**功能特点：**
- **智能加载**：使用`strict=False`模式，支持部分权重加载
- **兼容性**：网络结构可以不完全一致，自动匹配可用的参数
- **详细日志**：显示加载成功的参数数量、缺失参数和多余参数
- **容错性**：加载失败不会中断训练，会打印错误信息
- **自动检测**：如果路径为空或文件不存在，从头开始训练

**支持的文件格式：**
- PyTorch .pt文件（通过`torch.save(agent.state_dict(), path)`保存）
- 权重字典格式（state_dict）

**使用场景：**
- **完全匹配**：网络结构相同，加载所有权重
- **部分迁移**：只加载部分层权重（如backbone层）
- **网络扩展**：在新网络中加载旧网络的权重，多余层随机初始化
- **网络缩减**：加载可用权重，缺失层随机初始化

---

## 故障排除

### 常见问题

#### 1. 导入错误
**问题**: `ImportError: No module named 'marllib'`

**解决方案**:
```bash
# 安装MARLlib
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
pip install -e .
```

#### 2. 环境创建失败
**问题**: `KeyError: 'mdp_type'`

**解决方案**:
- 检查配置文件中的环境类型字段
- 确保环境类已正确注册
- 检查图文件路径

#### 3. 动作掩码错误
**问题**: `KeyError: 'action_mask'`

**解决方案**:
- 确保设置了`mask_flag: true`
- 检查环境观测格式是否正确

### 调试技巧

#### 启用详细日志
```yaml
system:
  log_level: "DEBUG"
```

#### 使用本地模式
```yaml
system:
  local_mode: true
  num_workers: 0
```
