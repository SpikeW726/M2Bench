# WandB Sweep 调参脚本实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 创建单文件 WandB Sweep 调参脚本，与 main.py 训练逻辑完全一致，支持超参数自动搜索。

**Architecture:**
- 单文件脚本 `sweep_mappo.py`，包含训练函数 + Sweep 配置
- 使用 WandB Sweep Controller 进行分布式超参数搜索
- 保持与 main.py 完全相同的训练流程（环境创建、网络初始化、采集、更新、日志）
- 通过 sweep config 覆盖可调参数，其他参数使用默认值

**Tech Stack:**
- WandB Sweep (wandb.sdk)
- PyTorch (与 main.py 相同)
- 项目现有模块: MASUPEnv, MAPPOAlgo, MACollector, MultiAgentPolicy

---

## 文件结构

```
MAP-imitation-framework/
├── sweep_mappo.py          # 新建：WandB Sweep 调参脚本
└── docs/plans/2026-02-10-wandb-sweep-script.md  # 本计划文档
```

---

## Task 1: 创建 sweep_mappo.py 基础框架

**Files:**
- Create: `sweep_mappo.py`

**Step 1: 编写基础框架（导入和主函数结构）**

```python
"""WandB Sweep 超参数搜索脚本 - MAPPO 训练"""

import time
from datetime import datetime
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import wandb

from envs.mdps.masup import MASUPEnv
from envs.venvs import DummyVectorEnv, SubprocVectorEnv
from networks.mlp import ActorMLP, CriticMLP
from policies.rl.rl_base import ActorPolicy
from policies.marl.marl_base import MultiAgentPolicy
from algorithms.marl.mappo import MAPPOAlgo
from data.collector import MACollector
from utils.model_io import save_model


def train():
    """
    WandB Sweep 训练函数
    由 wandb.agent() 调用，config 由 sweep 配置注入
    """
    # 从 wandb.config 获取超参数（由 sweep 设置）
    config = wandb.config

    # ========== 固定配置（不在 sweep 中调参） ==========
    num_envs = 12
    use_subproc = True

    # 网络隐藏层
    actor_hidden = [256, 256]
    critic_hidden = [256, 256]

    # 预训练权重路径（可选，从 sweep config 或固定值）
    actor_path = config.get("actor_path", "")
    critic_path = config.get("critic_path", "")

    # 保存路径配置
    algo_name = "mappo-sweep"
    graph_name = "TSP12"
    now = datetime.now()
    run_name = f"{graph_name}_sweep_{now:%Y-%m-%d_%H-%M-%S}"
    save_dir = Path(f"models/{algo_name}-{graph_name}/{run_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 日志初始化
    log_dir = Path(f"runs/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 打印 sweep 配置
    print(f"[Sweep] Starting run with config:")
    for key, value in config.items():
        if key not in ["actor_path", "critic_path"]:
            print(f"  {key}: {value}")

    # ========== 环境创建（与 main.py 相同） ==========
    # TODO: Task 2 实现

    # ========== 网络创建（与 main.py 相同） ==========
    # TODO: Task 3 实现

    # ========== 训练循环（与 main.py 相同） ==========
    # TODO: Task 4 实现

    writer.close()
    vec_env.close()


def main():
    """启动 WandB Sweep"""
    sweep_config = {
        "method": "bayes",  # 贝叶斯优化
        "metric": {
            "name": "env/iwi",  # 优化目标：最小化 IWI (越小越好)
            "goal": "minimize"
        },
        "parameters": {
            # TODO: Task 5 定义完整参数空间
        }
    }

    # 初始化 sweep
    sweep_id = wandb.sweep(sweep_config, project="MAP-Sweep")

    # 启动 agent
    wandb.agent(sweep_id, train, count=50)  # 运行 50 次试验


if __name__ == "__main__":
    main()
```

**Step 2: 验证文件创建成功**

Run: `python -c "import sweep_mappo; print('Import successful')"`

Expected: 无错误输出

**Step 3: 提交基础框架**

```bash
git add sweep_mappo.py
git commit -m "feat(wandb-sweep): add basic sweep_mappo.py framework"
```

---

## Task 2: 实现环境创建逻辑

**Files:**
- Modify: `sweep_mappo.py` - 在 train() 函数中添加环境创建代码

**Step 1: 添加环境创建代码（与 main.py 第 98-132 行相同）**

在 `train()` 函数中，替换 `# TODO: Task 2 实现` 为：

```python
    # ========== 环境 ==========
    with open("configs/MASUPEnv.yaml", 'r') as f:
        env_config_dict = yaml.safe_load(f)

    def make_env(env_config, custom_config):
        return lambda: MASUPEnv(env_config, **custom_config)

    env_fns = [make_env(env_config_dict["env_config"], env_config_dict["custom_config"])
               for _ in range(num_envs)]

    if use_subproc:
        vec_env = SubprocVectorEnv(env_fns)
        print(f"[Sweep] Using SubprocVectorEnv (parallel, {num_envs} processes)")
    else:
        vec_env = DummyVectorEnv(env_fns)
        print(f"[Sweep] Using DummyVectorEnv (sequential, single process)")

    # 从环境获取维度
    agent_ids = vec_env.agents
    num_agents = len(agent_ids)
    obs_space = vec_env.observation_space[agent_ids[0]]
    action_space = vec_env.action_space[agent_ids[0]]

    obs_dim = obs_space.shape[0]
    action_dim = action_space.n

    # global_state 维度
    temp_env = MASUPEnv(env_config_dict["env_config"], **env_config_dict["custom_config"])
    temp_env.reset()
    state_dim = len(temp_env.state())
    critic_state_dim = state_dim + num_agents
    temp_env.close()

    print(f"[Sweep] Created {num_envs} vectorized MASUPEnv")
    print(f"  Agents: {agent_ids}, Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Critic input dim: {critic_state_dim}")
```

**Step 2: 验证语法正确性**

Run: `python -m py_compile sweep_mappo.py`

Expected: 无语法错误

**Step 3: 提交环境创建代码**

```bash
git add sweep_mappo.py
git commit -m "feat(wandb-sweep): add environment setup logic"
```

---

## Task 3: 实现网络和算法初始化

**Files:**
- Modify: `sweep_mappo.py` - 在 train() 函数中添加网络创建和算法初始化代码

**Step 1: 添加网络和算法初始化代码（与 main.py 第 134-256 行相同，但使用 wandb.config）**

在 `train()` 函数中，替换 `# TODO: Task 3 实现` 为：

```python
    # ========== 创建网络 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor_net = ActorMLP(obs_dim, actor_hidden, action_dim)
    critic_net = CriticMLP(critic_state_dim, critic_hidden, 1)

    # 加载预训练权重（可选）
    value_norm_config = None
    if actor_path and critic_path and Path(actor_path).exists() and Path(critic_path).exists():
        actor_ckpt = torch.load(actor_path, map_location=device, weights_only=True)
        critic_ckpt = torch.load(critic_path, map_location=device, weights_only=True)
        actor_sd = actor_ckpt.get("actor_state_dict", actor_ckpt)
        critic_sd = critic_ckpt.get("critic_state_dict", critic_ckpt)
        actor_net.load_state_dict(actor_sd)
        critic_net.load_state_dict(critic_sd)
        print(f"[Sweep] Loaded pretrained weights from {actor_path}")

        # 读取 value_normalization 配置
        config_dir = Path(actor_path).parent
        config_file = config_dir / 'config.yaml'
        if config_file.exists():
            with open(config_file) as f:
                saved_config = yaml.full_load(f)
            if saved_config.get('value_normalization') is not None:
                value_norm_config = saved_config['value_normalization']
                ret_mean = float(value_norm_config.get('ret_mean', 0.0))
                ret_std = float(value_norm_config.get('ret_std', 1.0))
                value_norm_config['ret_mean'] = ret_mean
                value_norm_config['ret_std'] = ret_std
                print(f"[Sweep] Loaded value_norm config: mean={ret_mean:.4f}, std={ret_std:.4f}")
        else:
            print(f"[Sweep] No config.yaml found, value normalization disabled")
    else:
        print(f"[Sweep] Training from scratch (random initialization)")

    # ========== 构建 Policy 和 Algorithm ==========
    ma_policy = MultiAgentPolicy(
        agent_ids=agent_ids,
        obs_space=obs_space,
        action_space=action_space,
        policy_class=ActorPolicy,
        policy_kwargs={"actor": actor_net},
        shared=True,
    )

    # 从 wandb.config 获取训练超参数
    algorithm = MAPPOAlgo(
        policy=ma_policy,
        critic=critic_net,
        num_envs=num_envs,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        vf_coef=config.vf_coef,
        ent_coef=config.ent_coef,
        num_minibatches=config.num_minibatches,
        update_epochs=config.update_epochs,
        clip_vloss=True,
        use_value_norm=value_norm_config is not None,
        value_norm_config=value_norm_config,
    )

    # ========== Model Config ==========
    actor_config = {
        'type': 'ActorMLP',
        'input_dim': obs_dim,
        'hidden_sizes': actor_hidden,
        'output_dim': action_dim,
    }
    critic_config = {
        'type': 'CriticMLP',
        'input_dim': critic_state_dim,
        'hidden_sizes': critic_hidden,
        'output_dim': 1,
    }

    def get_value_norm_config():
        if algorithm.use_value_norm and algorithm.ret_rms is not None:
            return {
                'enabled': True,
                'ret_mean': float(algorithm.ret_rms.mean.item()),
                'ret_std': float(algorithm.ret_rms.std.item()),
                'ret_count': float(algorithm.ret_rms.count.item()),
            }
        return {'enabled': False}

    def build_extra_info(iteration: int):
        return {
            'iteration': iteration,
            'value_normalization': get_value_norm_config(),
        }
```

**Step 2: 验证语法正确性**

Run: `python -m py_compile sweep_mappo.py`

Expected: 无语法错误

**Step 3: 提交网络和算法初始化代码**

```bash
git add sweep_mappo.py
git commit -m "feat(wandb-sweep): add network and algorithm initialization"
```

---

## Task 4: 实现训练循环

**Files:**
- Modify: `sweep_mappo.py` - 在 train() 函数中添加完整训练循环

**Step 1: 添加训练循环代码（与 main.py 第 258-378 行相同）**

在 `train()` 函数中，替换 `# TODO: Task 4 实现` 为：

```python
    # ========== 训练循环 ==========
    collector = MACollector(algorithm, vec_env)
    collector.reset()

    # 从 wandb.config 获取训练超参数
    num_steps = config.num_steps
    total_timesteps = config.total_timesteps
    save_interval = config.get("save_interval", 1000)

    step_per_epoch = num_envs * num_steps
    num_iterations = total_timesteps // step_per_epoch
    global_step = 0
    start_time = time.time()

    print(f"\n[Sweep] Starting MAPPO training")
    print(f"  Total timesteps: {total_timesteps}, Iterations: {num_iterations}")
    print(f"  Batch size: {step_per_epoch * num_agents}, Device: {device}")

    for iteration in range(1, num_iterations + 1):
        # 0. Checkpoint
        if (iteration + 1) % save_interval == 0:
            ckpt_dir = save_dir / f"iter_{iteration + 1}"
            save_model(
                save_dir=ckpt_dir,
                policy=ma_policy,
                critic=critic_net,
                actor_config=actor_config,
                critic_config=critic_config,
                extra_info=build_extra_info(iteration + 1),
            )

        # 1. 采集数据
        t0 = time.time()
        algorithm.set_training_mode(False)
        result = collector.collect(n_steps=step_per_epoch)
        global_step += result.n_steps
        t_collect = time.time() - t0

        # 2. 计算 GAE 并更新
        t0 = time.time()
        batch = algorithm.prepare_batch(result.batch)
        algorithm.set_training_mode(True)
        stats = algorithm.update(batch)
        collector.reset_buffer()
        t_update = time.time() - t0

        # 3. 获取 episode 指标
        t0 = time.time()
        metrics_list = vec_env.call_env_method("get_episode_metrics")
        finished = [m for m in metrics_list if m is not None]
        if finished:
            env_metrics_igi = np.mean([m["igi"] for m in finished])
            env_metrics_agi = np.mean([m["agi"] for m in finished])
            env_metrics_iwi = np.mean([m["iwi"] for m in finished])
            env_metrics_wi = np.mean([m["wi"] for m in finished])
            env_metrics_wait_ratio = np.mean([m["wait_ratio"] for m in finished])
        else:
            cur_list = vec_env.call_env_method("get_current_metrics")
            m = cur_list[0]
            env_metrics_igi, env_metrics_agi, env_metrics_iwi, env_metrics_wi = m["igi"], m["agi"], m["iwi"], m["wi"]
            env_metrics_wait_ratio = m["wait_ratio"]
        t_metrics = time.time() - t0

        # 4. 记录日志
        sps = int(global_step / (time.time() - start_time))

        log_data = {
            "losses/policy_loss": stats.policy_loss,
            "losses/value_loss": stats.value_loss,
            "losses/entropy": stats.entropy,
            "losses/total_loss": stats.loss,
            "env/igi": env_metrics_igi,
            "env/agi": env_metrics_agi,
            "env/iwi": env_metrics_iwi,
            "env/wi": env_metrics_wi,
            "env/wait_ratio": env_metrics_wait_ratio,
            "charts/SPS": sps,
            "charts/global_step": global_step,
        }

        if stats.extra:
            log_data["losses/clipfrac"] = stats.extra.get("clipfrac", 0)
            log_data["losses/approx_kl"] = stats.extra.get("approx_kl", 0)

        if result.episode_rewards:
            log_data["charts/episode_reward"] = np.mean(result.episode_rewards)
            log_data["charts/episode_length"] = np.mean(result.episode_lengths)

        # TensorBoard
        for key, value in log_data.items():
            writer.add_scalar(key, value, global_step)

        # Wandb
        wandb.log(log_data, step=global_step)

        # 打印进度
        if iteration % 10 == 0 or iteration == 1:
            reward_str = f"{np.mean(result.episode_rewards):.2f}" if result.episode_rewards else "N/A"
            print(f"[Iter {iteration}/{num_iterations}] "
                  f"steps={global_step}, reward={reward_str}, "
                  f"pg_loss={stats.policy_loss:.4f}, v_loss={stats.value_loss:.4f}, "
                  f"iwi={env_metrics_iwi:.2f}, SPS={sps} "
                  f"(collect={t_collect:.1f}s update={t_update:.1f}s)")

    # Save final model
    final_dir = save_dir / "final"
    save_model(
        save_dir=final_dir,
        policy=ma_policy,
        critic=critic_net,
        actor_config=actor_config,
        critic_config=critic_config,
        extra_info=build_extra_info(num_iterations),
    )
    print(f"\n[Sweep] Saved final model to {final_dir}")
```

**Step 2: 验证语法正确性**

Run: `python -m py_compile sweep_mappo.py`

Expected: 无语法错误

**Step 3: 提交训练循环代码**

```bash
git add sweep_mappo.py
git commit -m "feat(wandb-sweep): add training loop"
```

---

## Task 5: 定义 Sweep 参数空间

**Files:**
- Modify: `sweep_mappo.py` - 在 main() 函数中定义完整的 sweep_config

**Step 1: 替换 sweep_config 中的 parameters 定义**

在 `main()` 函数中，替换 `# TODO: Task 5 定义完整参数空间` 为：

```python
        "parameters": {
            # 学习率
            "actor_lr": {
                "min": 1e-5,
                "max": 1e-3
            },
            "critic_lr": {
                "min": 1e-4,
                "max": 1e-3
            },
            # PPO 裁剪参数
            "clip_range": {
                "min": 0.1,
                "max": 0.4
            },
            # 价值函数系数
            "vf_coef": {
                "min": 0.1,
                "max": 2.0
            },
            # 熵系数
            "ent_coef": {
                "min": 0.0,
                "max": 0.3
            },
            # GAE lambda
            "gae_lambda": {
                "min": 0.9,
                "max": 1.0
            },
            # 批次配置
            "num_minibatches": {
                "values": [4, 8, 16, 32]
            },
            "update_epochs": {
                "values": [3, 5, 10]
            },
            "num_steps": {
                "values": [1024, 2048, 4096]
            },
            # 折扣因子
            "gamma": {
                "min": 0.99,
                "max": 0.9999
            },
            # 训练总步数（固定值，用于控制实验长度）
            "total_timesteps": {
                "value": 10000000  # sweep 时使用较短训练用于快速筛选
            },
            # 预训练权重（可选，用于对比实验）
            "actor_path": {
                "values": [
                    "",  # 从头训练
                    # "models/imi-pure-norm-random-init/imi_train__1770181266_final/policy.pt",
                ]
            },
            "critic_path": {
                "values": [
                    "",  # 从头训练
                    # "models/imi-pure-norm-random-init/imi_train__1770181266_final/critic.pt",
                ]
            },
        }
```

**Step 2: 验证 JSON 配置有效性**

Run: `python -c "import yaml; import sweep_mappo; print(yaml.dump(sweep_mappo.main.__code__))"`

Expected: 无语法错误（这步主要是验证脚本可导入）

**更好的验证方式：创建一个临时测试脚本**

```bash
cat > test_sweep_config.py << 'EOF'
import yaml
import json

# 测试 sweep config 是否可以被 wandb 解析
sweep_config = {
    "method": "bayes",
    "metric": {"name": "env/iwi", "goal": "minimize"},
    "parameters": {
        "actor_lr": {"min": 1e-5, "max": 1e-3},
        "critic_lr": {"min": 1e-4, "max": 1e-3},
        "clip_range": {"min": 0.1, "max": 0.4},
        "vf_coef": {"min": 0.1, "max": 2.0},
        "ent_coef": {"min": 0.0, "max": 0.3},
        "gae_lambda": {"min": 0.9, "max": 1.0},
        "num_minibatches": {"values": [4, 8, 16, 32]},
        "update_epochs": {"values": [3, 5, 10]},
        "num_steps": {"values": [1024, 2048, 4096]},
        "gamma": {"min": 0.99, "max": 0.9999},
        "total_timesteps": {"value": 10000000},
    }
}

# 验证 JSON 序列化
try:
    json_str = json.dumps(sweep_config)
    print("Sweep config is valid JSON")
    print(f"Config size: {len(json_str)} bytes")
except Exception as e:
    print(f"Invalid config: {e}")

# 清理
rm test_sweep_config.py
EOF

python test_sweep_config.py && rm test_sweep_config.py
```

Expected: "Sweep config is valid JSON"

**Step 3: 提交 Sweep 参数配置**

```bash
git add sweep_mappo.py
git commit -m "feat(wandb-sweep): define sweep parameter space"
```

---

## Task 6: 添加命令行参数支持

**Files:**
- Modify: `sweep_mappo.py` - 添加 argparse 支持灵活配置

**Step 1: 在 main() 函数开头添加 argparse**

在文件开头的 import 部分添加：

```python
import argparse
```

在 `if __name__ == "__main__":` 部分修改为：

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Sweep for MAPPO training")
    parser.add_argument("--project", type=str, default="MAP-Sweep",
                        help="WandB project name")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of sweep runs")
    parser.add_argument("--method", type=str, default="bayes",
                        choices=["bayes", "random", "grid"],
                        help="Sweep method")
    parser.add_argument("--actor-path", type=str, default="",
                        help="Pretrained actor path (override sweep config)")
    parser.add_argument("--critic-path", type=str, default="",
                        help="Pretrained critic path (override sweep config)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override total timesteps for quick testing")

    args = parser.parse_args()

    # 创建 sweep 配置
    sweep_config = {
        "method": args.method,
        "metric": {
            "name": "env/iwi",
            "goal": "minimize"
        },
        "parameters": {
            # ... (保持 Task 5 中定义的 parameters)
        }
    }

    # 如果命令行指定了预训练权重，覆盖 sweep 配置
    if args.actor_path or args.critic_path:
        sweep_config["parameters"]["actor_path"] = {"value": args.actor_path}
        sweep_config["parameters"]["critic_path"] = {"value": args.critic_path}

    # 如果命令行指定了 total_timesteps，覆盖配置
    if args.total_timesteps is not None:
        sweep_config["parameters"]["total_timesteps"] = {"value": args.total_timesteps}

    # 初始化 sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project)

    # 启动 agent
    wandb.agent(sweep_id, train, count=args.count)
```

**Step 2: 验证命令行参数解析**

Run: `python sweep_mappo.py --help`

Expected: 显示帮助信息，包含所有定义的参数

**Step 3: 提交命令行参数支持**

```bash
git add sweep_mappo.py
git commit -m "feat(wandb-sweep): add command line argument support"
```

---

## Task 7: 创建 Sweep 配置文件（可选）

**Files:**
- Create: `configs/sweep_config.yaml`

**Step 1: 创建 YAML 格式的 sweep 配置文件**

```yaml
# configs/sweep_config.yaml
# WandB Sweep 配置文件

method: bayes

metric:
  name: env/iwi
  goal: minimize

parameters:
  # 学习率
  actor_lr:
    min: 0.00001
    max: 0.001
  critic_lr:
    min: 0.0001
    max: 0.001

  # PPO 参数
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

  # 批次配置
  num_minibatches:
    values: [4, 8, 16, 32]
  update_epochs:
    values: [3, 5, 10]
  num_steps:
    values: [1024, 2048, 4096]

  # 折扣因子
  gamma:
    min: 0.99
    max: 0.9999

  # 训练长度
  total_timesteps:
    value: 10000000  # 10M 步用于快速筛选

  # 预训练权重
  actor_path:
    values: [""]
  critic_path:
    values: [""]
```

**Step 2: 更新 sweep_mappo.py 支持加载 YAML 配置**

在 main() 函数中添加 YAML 配置加载选项：

```python
    parser.add_argument("--sweep-config", type=str, default=None,
                        help="Path to sweep config YAML file")
```

并在解析参数后添加：

```python
    # 如果提供了 YAML 配置文件，从文件加载
    if args.sweep_config:
        with open(args.sweep_config, 'r') as f:
            sweep_config = yaml.safe_load(f)
        print(f"[Main] Loaded sweep config from {args.sweep_config}")
    else:
        # 使用默认配置
        sweep_config = {...}  # 现有代码
```

**Step 3: 验证 YAML 配置加载**

Run: `python -c "import yaml; c = yaml.safe_load(open('configs/sweep_config.yaml')); print(c['method'])"`

Expected: 输出 `bayes`

**Step 4: 提交 YAML 配置文件**

```bash
git add configs/sweep_config.yaml sweep_mappo.py
git commit -m "feat(wandb-sweep): add YAML sweep config file support"
```

---

## Task 8: 编写使用文档

**Files:**
- Create: `docs/wandb_sweep_guide.md`

**Step 1: 编写使用文档**

```markdown
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
```

**Step 2: 提交文档**

```bash
git add docs/wandb_sweep_guide.md
git commit -m "docs(wandb-sweep): add usage guide"
```

---

## Task 9: 验证和测试

**Files:**
- Test: 手动测试脚本运行

**Step 1: 创建快速测试脚本**

创建 `test_sweep_mappo.py`：

```python
"""快速测试 sweep_mappo.py 是否可以正常运行（单次试验）"""

import os
os.environ["WANDB_MODE"] = "disabled"  # 禁用 wandb 上传

import subprocess
import sys

# 运行一次 sweep 试验（使用最短训练）
cmd = [
    sys.executable, "sweep_mappo.py",
    "--project", "MAP-Test",
    "--count", "1",
    "--total-timesteps", "10000",  # 只训练 10k 步用于测试
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
if result.stderr:
    print("STDERR:")
    print(result.stderr)

if result.returncode == 0:
    print("\n✅ Test passed! sweep_mappo.py is ready to use.")
else:
    print(f"\n❌ Test failed with exit code {result.returncode}")
```

**Step 2: 运行测试**

Run: `python test_sweep_mappo.py`

Expected: "✅ Test passed!"

**Step 3: 清理测试文件**

```bash
rm test_sweep_mappo.py
```

**Step 4: 最终提交**

```bash
git add .
git commit -m "feat(wandb-sweep): complete WandB sweep implementation"
```

---

## Task 10: 代码审查和优化

**Files:**
- Review: `sweep_mappo.py`

**Step 1: 完整代码审查**

检查项：
- [ ] 所有 import 与 main.py 一致
- [ ] 训练循环逻辑与 main.py 完全相同
- [ ] Sweep 参数覆盖正确
- [ ] 日志记录完整（TensorBoard + WandB）
- [ ] 模型保存路径正确
- [ ] 命令行参数解析健壮
- [ ] 错误处理充分

**Step 2: 添加错误处理（可选优化）**

在 `train()` 函数中添加 try-except：

```python
def train():
    try:
        # ... 现有代码 ...
    except Exception as e:
        print(f"[Sweep] Error during training: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        raise
```

**Step 3: 格式化代码**

Run: `python -m black sweep_mappo.py` (如果项目使用 black)

**Step 4: 最终验证**

```bash
python -m py_compile sweep_mappo.py
python sweep_mappo.py --help
```

**Step 5: 提交最终版本**

```bash
git add sweep_mappo.py
git commit -m "refactor(wandb-sweep): add error handling and code formatting"
```

---

## 附录：文件依赖关系

```
sweep_mappo.py
├── envs/mdps/masup.py (MASUPEnv)
├── envs/venvs.py (DummyVectorEnv, SubprocVectorEnv)
├── networks/mlp.py (ActorMLP, CriticMLP)
├── policies/rl/rl_base.py (ActorPolicy)
├── policies/marl/marl_base.py (MultiAgentPolicy)
├── algorithms/marl/mappo.py (MAPPOAlgo)
├── data/collector.py (MACollector)
├── utils/model_io.py (save_model)
└── configs/MASUPEnv.yaml (环境配置)
```

---

## 总结

实现完成后，`sweep_mappo.py` 将具备以下功能：

1. **完全兼容 main.py 训练逻辑**：环境、网络、算法、训练循环完全一致
2. **WandB Sweep 集成**：支持贝叶斯/随机/网格搜索
3. **灵活的参数配置**：支持代码默认值、YAML 配置文件、命令行覆盖
4. **完整的日志记录**：TensorBoard + WandB 双重记录
5. **模型自动保存**：每次 sweep 运行独立保存

**预期使用场景：**

```bash
# 场景1: 快速超参数搜索（10M 步快速筛选）
python sweep_mappo.py --count 50

# 场景2: 基于预训练模型微调超参数
python sweep_mappo.py --actor-path models/xxx/policy.pt --critic-path models/xxx/critic.pt --count 20

# 场景3: 使用自定义 sweep 配置
python sweep_mappo.py --sweep-config configs/sweep_config.yaml --count 100
```
