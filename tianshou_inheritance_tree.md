# Tianshou库 Algorithm 和 Policy 继承关系图

## Algorithm 继承关系

```
Algorithm (基类)
│
├── OnPolicyAlgorithm
│   │
│   ├── ActorCriticOnPolicyAlgorithm
│   │   │
│   │   ├── A2C
│   │   │   │
│   │   │   └── PPO
│   │   │       │
│   │   │       └── GAIL
│   │   │
│   │   └── NPG
│   │       │
│   │       └── TRPO
│   │
│   ├── Reinforce
│   │
│   ├── PSRL
│   │
│   └── OnPolicyWrapperAlgorithm
│       │
│       └── ICMOnPolicyWrapper
│
├── OffPolicyAlgorithm
│   │
│   ├── QLearningOffPolicyAlgorithm
│   │   │
│   │   ├── DQN
│   │   │
│   │   ├── C51
│   │   │   │
│   │   │   └── RainbowDQN
│   │   │
│   │   ├── QRDQN
│   │   │   │
│   │   │   ├── IQN
│   │   │   │
│   │   │   └── FQF
│   │   │
│   │   └── BDQN
│   │
│   ├── ActorCriticOffPolicyAlgorithm
│   │   │
│   │   ├── DDPG
│   │   │
│   │   ├── SAC
│   │   │
│   │   ├── REDQ
│   │   │
│   │   └── ActorDualCriticsOffPolicyAlgorithm
│   │       │
│   │       ├── TD3
│   │       │   │
│   │       │   └── TD3BC (多重继承: OfflineAlgorithm + TD3)
│   │       │
│   │       └── DiscreteSAC
│   │
│   ├── OffPolicyWrapperAlgorithm
│   │   │
│   │   └── ICMOffPolicyWrapper
│   │
│   ├── MultiAgentOffPolicyAlgorithm
│   │
│   └── MARLRandomDiscreteMaskedOffPolicyAlgorithm
│
└── OfflineAlgorithm
    │
    ├── OffPolicyImitationLearning
    │
    ├── OfflineImitationLearning
    │
    ├── BCQ
    │
    ├── CQL
    │
    ├── DiscreteBCQ
    │
    ├── DiscreteCQL (多重继承: OfflineAlgorithm + QRDQN)
    │
    └── DiscreteCRR
```

## Policy 继承关系

```
Policy (基类)
│
├── RandomActionPolicy
│
├── ProbabilisticActorPolicy
│   │
│   └── DiscreteActorPolicy
│
├── DiscreteQLearningPolicy
│   │
│   ├── C51Policy
│   │
│   ├── QRDQNPolicy
│   │   │
│   │   ├── IQNPolicy
│   │   │
│   │   └── FQFPolicy
│   │
│   ├── BDQNPolicy
│   │
│   └── DiscreteBCQPolicy
│
├── ContinuousPolicyWithExplorationNoise
│   │
│   ├── ContinuousDeterministicPolicy
│   │
│   ├── SACPolicy
│   │
│   └── REDQPolicy
│
├── ImitationPolicy
│   │
│   └── BCQPolicy
│
├── PSRLPolicy
│
├── DiscreteSACPolicy
│
└── MultiAgentPolicy
```

## Mixin 类

```
LaggedNetworkAlgorithmMixin (抽象基类)
│
├── LaggedNetworkFullUpdateAlgorithmMixin
│   │
│   └── (被 QLearningOffPolicyAlgorithm, DiscreteBCQ, DiscreteCRR 使用)
│
└── LaggedNetworkPolyakUpdateAlgorithmMixin
    │
    └── (被 ActorCriticOffPolicyAlgorithm, BCQ, CQL 使用)
```

## 说明

1. **多重继承**：
   - `TD3BC` 同时继承自 `OfflineAlgorithm` 和 `TD3`
   - `DiscreteCQL` 同时继承自 `OfflineAlgorithm` 和 `QRDQN`

2. **Wrapper 模式**：
   - `ICMOnPolicyWrapper` 和 `ICMOffPolicyWrapper` 使用包装器模式，包装其他算法

3. **多智能体**：
   - `MultiAgentOffPolicyAlgorithm` 和 `MultiAgentOnPolicyAlgorithm` 用于多智能体强化学习
   - `MultiAgentPolicy` 包含多个子策略

4. **Mixin 类**：
   - `LaggedNetworkAlgorithmMixin` 及其子类用于实现目标网络（target network）功能
