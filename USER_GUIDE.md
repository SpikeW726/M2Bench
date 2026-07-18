# M2Bench User Guide

[English](USER_GUIDE.md) | [简体中文](USER_GUIDE_zh-CN.md) | [README](README.md) | [Paper](https://arxiv.org/abs/2605.09633v2)

This guide describes the current codebase. Commands assume the repository root as the working directory and the `Patrolling` Conda environment is active.

> **TWLO and `masup`.** The paper names our formulation the Tail Worst-case Latency-Optimizing MDP (TWLO-MDP). The earlier code identifier `masup` remains in module names, registry keys, and YAML paths. They refer to the same MDP. This guide uses **TWLO** in prose and `masup` only when a literal code identifier is required.

## Contents

- [Configuration model](#configuration-model)
- [Minimal workflows](#minimal-workflows)
- [Implemented monitoring MDPs](#implemented-monitoring-mdps)
- [Implemented RL and MARL algorithms](#implemented-rl-and-marl-algorithms)
- [Implemented heuristic policies](#implemented-heuristic-policies)
- [Networks and policy mapping](#networks-and-policy-mapping)
- [Adding a map](#adding-a-map)
- [Adding an MDP](#adding-an-mdp)
- [Adding a network](#adding-a-network)
- [Adding an algorithm](#adding-an-algorithm)

## Configuration Model

M2Bench uses three kinds of YAML files:

| Configuration | Location | Purpose |
|---|---|---|
| Experiment | `configs/experiments/<mdp>/` | Environment, algorithm, networks, training budget, logging, and optional post-training evaluation |
| Sweep | `configs/sweep/<mdp>/` | W&B search method, objective, parameter distributions, and optional early termination |
| Evaluation | `configs/eval/<mdp>/` | Evaluation environment, episode count, plots, animation, and action-score logging |
| Heuristic | `configs/heuristic/` | Parameters and default display settings for one heuristic; the runtime environment is supplied by CLI |

An experiment YAML is loaded into `ExperimentConfig` and contains these principal sections:

```yaml
algo_name: mappo          # ALGO_REGISTRY key
env_type: masup           # ENV_REGISTRY key; masup means TWLO
actor_type: masup_mlp     # optional ACTOR_REGISTRY key
critic_type: masup_mlp    # optional CRITIC_REGISTRY key
q_type: null              # optional Q_NETWORK_REGISTRY key

env:                      # EnvConfig
  graph_path: graphs/grid.json
  num_agents: 3
  init_positions: [1, 26, 50]
  episode_len: 500
  custom_configs: {}

algo: {}                  # algorithm-specific parameter dataclass
training: {}              # TrainerConfig / OnPolicyTrainerConfig / OffPolicyTrainerConfig
actor: {}                 # actor architecture parameters
critic: {}                # critic architecture parameters
q_network: {}             # Q-network architecture parameters
```

Unknown fields inside dataclass-backed sections are ignored, so spelling errors may silently leave a default value in effect. Compare new files with an existing configuration from the same MDP and algorithm family.

Important training fields are:

| Field | Meaning |
|---|---|
| `training.num_envs` | Number of parallel simulator instances |
| `training.num_steps` | Transitions collected per environment and iteration |
| `training.total_steps` | Global environment-step budget; overrides the effective iteration count |
| `training.use_subproc` | Use subprocess rather than in-process vector environments |
| `training.minibatch_size`, `update_epochs` | On-policy update schedule |
| `training.batch_size`, `warmup_steps`, `buffer_size` | Off-policy replay schedule |
| `algo.shared_policy` | Share one policy network when supported; otherwise create independent networks |
| `env.custom_configs.truncate_by_time` | Interpret `episode_len` as physical simulation time instead of decision steps |

By default, checkpoints, TensorBoard-compatible run data, and evaluation outputs are placed under `models/`, `runs/`, and `evaluators/results/`. `train.py --save-dir`, `train.py --results-dir`, `sweep.py --models-dir`, `sweep.py --results-dir`, and evaluator output options override these roots.

## Minimal Workflows

### Train one experiment

```bash
python train.py configs/experiments/masup/mappo_masup_grid_a3.yaml
```

The example is MAPPO on TWLO despite the historical `masup` path. The run writes periodic/best/final checkpoints below a timestamped model directory. If `eval_config_path` is present in the experiment YAML, the best available checkpoint is evaluated after training.

Useful overrides:

```bash
python train.py configs/experiments/masup/mappo_masup_grid_a3.yaml \
  --eval-config configs/eval/masup/masup_grid_a3.yaml \
  --save-dir /data/checkpoints/grid-a3 \
  --results-dir /data/evaluation
```

Existing configurations are full research runs. Create a separate smoke-test YAML with a small `total_steps`, `num_envs`, and `num_steps`; do not infer runtime from the short command itself.

### Run a W&B sweep

Authenticate once:

```bash
wandb login
```

Create and execute a sweep in the current process:

```bash
python sweep.py \
  --base-config configs/experiments/masup/mappo_masup_grid_a3.yaml \
  --sweep-config configs/sweep/masup/mappo_masup_grid_a3.yaml \
  --count 20 \
  --eval-config configs/eval/masup/masup_grid_a3.yaml \
  --top-n 5
```

The base experiment supplies all fixed values. Sweep keys matching algorithm, trainer, or top-level experiment fields are applied automatically; environment-specific values use `custom_configs.<key>`.

To distribute trials across terminals or machines, create the sweep first:

```bash
python sweep.py \
  --base-config configs/experiments/masup/mappo_masup_grid_a3.yaml \
  --sweep-config configs/sweep/masup/mappo_masup_grid_a3.yaml \
  --create-only
```

Then start one or more agents with the printed ID and the same W&B project:

```bash
python sweep.py \
  --base-config configs/experiments/masup/mappo_masup_grid_a3.yaml \
  --sweep-config configs/sweep/masup/mappo_masup_grid_a3.yaml \
  --sweep-id <SWEEP_ID> \
  --project <WANDB_PROJECT> \
  --count 10
```

### Evaluate a learned policy

`evaluators/test.py` reconstructs the policy from checkpoint metadata. `--model` must be a directory containing `config.yaml` and `policy.pt`; neural actor-critic checkpoints may also contain `critic.pt`.

```bash
python evaluators/test.py \
  --model models/mappo-masup-grid/<timestamp>/best \
  --env_config configs/eval/masup/masup_grid_a3.yaml \
  --num_episodes 10 \
  --episode_time 500 \
  --results-dir evaluators/results \
  --no_show
```

Common optional outputs:

```bash
python evaluators/test.py \
  --model <CHECKPOINT_DIR> \
  --env_config <EVAL_YAML> \
  --animation \
  --max_frames 400 \
  --log_action_logits \
  --action_logits_csv evaluators/results/action_scores.csv \
  --no_show
```

The evaluator reports IGI, AGI, IWI, and WI under the same simulator metric tracker used during training. TWLO configurations additionally expose `wi_fromT` when the transient cutoff `T` is enabled.

### Evaluate heuristic policies

The heuristic evaluator receives environment parameters directly from the terminal; heuristic YAML files contain only policy and display defaults.

```bash
python evaluators/heuristic_evaluator.py island 6 500 \
  --policy HPCC \
  --init-positions 0 10 20 30 40 49 \
  --speeds 1 1 1 1 1 1 \
  --num_episodes 10 \
  --results-dir evaluators/results \
  --no_show
```

The positional syntax is:

```text
heuristic_evaluator.py MAP NUM_AGENTS EPISODE_LEN [options]
```

`MAP` may be `island`, `island.json`, or an explicit graph path. The length of `--init-positions` and `--speeds` must equal `NUM_AGENTS`, and every initial node ID must occur in the graph's `nodes` array. Omit initial positions for random starts.

Additional environment options include `--enable-wait`, `--deltaT`, `--truncate-by-steps`, edge-time jitter controls, and repeatable generic overrides such as `--env edge_time_jitter_frac=0.2`.

Run every heuristic with one shared map/team/horizon setup:

```bash
python run_all_heuristics.py island 6 500 \
  --init-positions 0 10 20 30 40 49
```

## Implemented Monitoring MDPs

The table lists conceptual formulations rather than counting API adapters as separate MDPs. Links identify the source formulation; implementations adapt each method to the shared weighted-graph simulator and metric protocol.

| MDP | Registry key(s) | Time/API | Main implementation | Reference |
|---|---|---|---|---|
| **TWLO** | `masup` | Event-driven, PettingZoo parallel | Tail-worst-latency state and duration-aware reward; historical code class `MASUPEnv` | [Wang et al., 2026](https://arxiv.org/abs/2605.09633v2) |
| **TWLO graph observation** | `masup_gnn` | Event-driven, PettingZoo parallel | TWLO dynamics with static graph nodes, virtual robot nodes, and edge features | [Wang et al., 2026](https://arxiv.org/abs/2605.09633v2) |
| BBLA | `bbla` | Fixed-step, PettingZoo parallel | Black-Box Learner Agent state and visited-node idleness reward | [Santana et al., 2004](https://ieeexplore.ieee.org/document/1373634) |
| GBLA | `gbla` | Fixed-step, PettingZoo parallel | BBLA plus other robots' adjacent target intentions | [Santana et al., 2004](https://ieeexplore.ieee.org/document/1373634) |
| Extended-GBLA | `ex_gbla` | Fixed-step, PettingZoo parallel | Ordered adjacent idleness and coordination-aware reward | [Lauri and Koukam, 2014](https://link.springer.com/chapter/10.1007/978-3-319-12970-9_18) |
| NEP | `nep` | Event-driven, Gymnasium joint | Node-edge-position state with tabular Q-learning interface | [Hu and Zhao, 2010](https://ieeexplore.ieee.org/document/5599681) |
| S4R1 | `s4r1` | Fixed-step, PettingZoo parallel | Source/target and neighborhood-idleness state used with deep Q-learning | [Jana et al., 2022](https://doi.org/10.1007/s41315-022-00235-1) |
| BEAU | `beau` | Event-driven, PettingZoo parallel | Graph state for autoregressive MAT execution; adapted from the original grid/visibility setting | [Guo et al., 2023](https://doi.org/10.1109/ICRA48891.2023.10160923) |
| MAGEC | `magec` | Event-driven, PettingZoo parallel | GraphSAGE-compatible patrol and virtual-agent graph features | [Goeckner et al., 2024](https://arxiv.org/abs/2403.13093) |
| SUNS | `suns`, `suns_gym` | Event-driven; parallel or single-agent Gym adapter | Full-graph idleness/distance features with SUN actor/critic | [Ward et al., 2025](https://arxiv.org/abs/2412.11916) |
| OUCS | `oucs` | Fixed-step, PettingZoo parallel | Agent locations, neighbor visit counts, priorities, and cooperative reward | [Palma-Borda et al., 2026](https://doi.org/10.1016/j.engappai.2025.113706) |

`suns_gym` is restricted to one robot. `masup_gnn` changes the TWLO observation representation, not its monitoring objective. BEAU preserves the decision-step collection concept but is not step-for-step identical to the original implementation; see `envs/mdps/beau.py` for the documented adaptations.

## Implemented RL and MARL Algorithms

`configs/registry.py` is the source of truth for callable identifiers.

| Registry key | Family | Policy organization | Reference / implementation note |
|---|---|---|---|
| `a2c` | On-policy actor-critic | Single or shared actor | Synchronous A2C based on [Mnih et al., 2016](https://arxiv.org/abs/1602.01783) |
| `maa2c` | MARL actor-critic | Shared actor, centralized critic | Repository CTDE extension of A2C |
| `ppo` | On-policy PPO | Actor and critic | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) |
| `mappo` | MARL PPO | Shared actor, centralized critic | [Yu et al., 2022](https://arxiv.org/abs/2103.01955) |
| `ippo` | Independent PPO | Independent by default; sharing configurable | [de Witt et al., 2020](https://arxiv.org/abs/2011.09533) |
| `vdppo` | Value-decomposition PPO | PPO actor plus decomposed Q functions | VDPPO baseline described by [Palma-Borda et al., 2026](https://doi.org/10.1016/j.engappai.2025.113706); repository uses a QPLEX-style mixer |
| `d3qn` | Off-policy value learning | Q-network | Double DQN with optional [dueling architecture](https://arxiv.org/abs/1511.06581) |
| `iql` | Independent Q-learning | Independent Q-network per robot | [Tan, 1993](https://doi.org/10.1145/168871.168872) |
| `vdn` | Value decomposition | Shared per-agent Q-network and sum mixer | [Sunehag et al., 2017](https://arxiv.org/abs/1706.05296) |
| `qmix` | Monotonic value decomposition | Shared Q-network and state-conditioned mixer | [Rashid et al., 2020](https://jmlr.org/papers/v21/20-081.html) |
| `qtable` | Tabular Q-learning | Independent Q-table per robot | Q-learning from [Watkins and Dayan, 1992](https://doi.org/10.1007/BF00992698) |
| `mappo_mat` | Autoregressive multi-agent PPO | GAT encoder and MAT decoder | MAT architecture from [Wen et al., 2022](https://arxiv.org/abs/2205.14953) |
| `happo` | BEAU compatibility key | Same `MAPPOMATAlgo` implementation as `mappo_mat` | Used by BEAU experiment YAMLs; it is not a separate canonical HAPPO implementation |

On-policy training supports action masks, active-decision masks, recurrent chunks, value normalization, and transparent GAE for event-driven inactive steps. Off-policy training supports flat or sequential replay, target networks, recurrent burn-in, and decision-epoch synchronization options where configured.

## Implemented Heuristic Policies

These are the exact names accepted by `--policy`. Reference links indicate the conceptual source. Several controllers are simulator-adapted or lightweight repository variants, so consult the module docstring before claiming exact reproduction of an original algorithm.

| CLI name | Policy | Reference / note |
|---|---|---|
| `RANDOM` | Uniform random neighbor | Reference-free baseline |
| `CR` | Conscientious Reactive | [Portugal and Rocha, 2013](https://doi.org/10.1080/01691864.2013.763722) |
| `HCR` | Heuristic Conscientious Reactive | Algorithm 2 in [Portugal and Rocha, 2013](https://doi.org/10.1080/01691864.2013.763722) |
| `CC` | Conscientious Cognitive | Repository local idleness-distance variant; related comparison in [Portugal and Rocha, 2013](https://doi.org/10.1080/01691864.2013.763722) |
| `HPCC` | Heuristic Pathfinder Conscientious Cognitive | Algorithm 3 in [Portugal and Rocha, 2013](https://doi.org/10.1080/01691864.2013.763722) |
| `ER` | Expected Idleness / Expected Reactive | [Yan and Zhang, 2016](https://doi.org/10.1177/1729881416663666) |
| `GBS` | Greedy Bayesian Strategy | [Portugal and Rocha, 2013](https://doi.org/10.1016/j.robot.2013.06.011) |
| `SEBS` | State Exchange Bayesian Strategy | [Portugal and Rocha, 2013](https://doi.org/10.1016/j.robot.2013.06.011) |
| `BAPS` | Bayesian Ant Patrolling Strategy | [Chen et al., 2015](https://doi.org/10.5194/isprsannals-II-4-W2-103-2015) |
| `CBLS` | Cycle-Based strategy with tabu lists | Lightweight repository variant; related Bayesian patrolling background in [Portugal's thesis](https://ap.isr.uc.pt/archive/DPortugal_PhDThesis_2014.pdf) |
| `MSP` | Configured cyclic-route policy | Related partitioning method: [Portugal and Rocha, 2010](https://ap.isr.uc.pt/archive/PR10_ACM_SAC2010.pdf); this implementation consumes precomputed routes |
| `DTAGREEDY` | Distributed task-assignment greedy variant | [Farinelli et al., 2017](https://doi.org/10.1007/s10514-016-9579-8) |
| `DTASSI` | Sequential-single-item-inspired DTA variant | Related DTA auction method in [Farinelli et al., 2017](https://doi.org/10.1007/s10514-016-9579-8); current code uses noisy local scores rather than a full auction protocol |
| `AHPA` | Adaptive Heuristic Patrolling Agent | [Goeckner et al., 2024](https://arxiv.org/abs/2304.01386) |

## Networks and Policy Mapping

Actor, critic, and Q-network selection is independent. This is what allows one MDP and one optimization algorithm to be combined with MLP, recurrent, graph, or formulation-specific encoders without rewriting the trainer.

| Registry | Available keys |
|---|---|
| Actor | `mlp`, `rnn`, `sun`, `mpnn`, `sage`, `masup_mlp`, `masup_rnn` |
| Critic | `mlp`, `rnn`, `sun`, `masup_mlp`, `masup_rnn` |
| Q-network | `mlp`, `rnn`, `masup_q_mlp`, `masup_q_rnn`, `masup_vdppo_mlp`, `masup_vdppo_rnn` |

The historical `masup_*` network keys denote TWLO-specific observation encoders. MAT uses its own `GATEncoder` and `MATDecoder` construction path.

`ActorPolicy` and `ValuePolicy` define reusable observation-to-action behavior. `MultiAgentPolicy` maps robot IDs either to separate policy instances or to one shared instance. In shared mode, robot batches are stacked for one network call and then mapped back to agent-keyed outputs. This separation lets an algorithm change how parameters are optimized without changing how a policy performs action masking, recurrence, or inference.

## Adding a Map

### 1. Create the graph JSON

Maps use `G=(V,E,W,phi)`:

```json
{
  "nodes": [0, 1, 2],
  "edges": [
    {"from": 0, "to": 1, "weight": 1.0},
    {"from": 1, "to": 0, "weight": 1.0},
    {"from": 1, "to": 2, "weight": 2.0},
    {"from": 2, "to": 1, "weight": 2.0}
  ],
  "phi": {"0": 1.0, "1": 2.0, "2": 1.0}
}
```

- `nodes` contains integer node IDs. Contiguous IDs are strongly recommended because some baseline state encodings assume them.
- Each `weight` is a positive traversal distance. The simulator derives travel time from distance and robot speed.
- `phi` assigns every node a positive monitoring-priority weight.
- The parser stores directed adjacency entries. For an undirected monitoring graph, include both directions with equal weight as existing maps do.
- The graph should be connected for shortest-path-based policies and MDPs.

### 2. Generate display coordinates

```bash
python utils/graph_layout.py graphs/my_map.json
```

This creates `graphs/my_map_coords.json` with normalized coordinates. Coordinate files are used by visualization and graph-based BEAU/MAT components; those components can generate missing coordinates, but committing a fixed file makes plots reproducible.

### 3. Add configurations

Copy an experiment and evaluation YAML from the same MDP family. Update at least:

- `graph_name` and every `graph_path` in environment/network sections.
- `num_agents`, `init_positions`, and `episode_len`.
- TWLO-specific `num_nodes`, `num_agents`, role information, and cutoff `T` in its network sections.
- Any graph-size-dependent batch or model parameters.

Validate every initial node against the literal `nodes` array. A node count of 50 does not imply that IDs `0` through `49` all exist.

## Adding an MDP

### 1. Select the environment API

- Subclass `FixedStepEnv` or `EventDrivenEnv` for homogeneous PettingZoo parallel agents.
- Subclass `JointFixedStepEnv` or `JointEventDrivenEnv` for a single Gymnasium action controlling the joint process.
- Reuse `PatrolWorld` for graph dynamics, arrivals, waiting, idleness, priorities, and metric tracking.

### 2. Implement the contract

Parallel environments provide `observation_space`, `action_space`, `state`, `_build_obs`, `_build_info`, `_compute_rewards`, `_compute_truncations`, and `_action_to_target`. Joint environments provide Gymnasium spaces and the corresponding singular dispatch, observation, info, reward, and truncation methods.

For asynchronous environments, expose an `active_mask` and give non-ready robots only a no-op action in `action_mask`. This prevents optimization on decisions a robot did not actually make.

### 3. Register and configure it

Add the module and class to `ENV_REGISTRY` in `configs/registry.py`, then create matching experiment and evaluation YAMLs. Environment constructor fields shared across MDPs belong in `EnvConfig`; formulation-specific values belong in `env.custom_configs` and arrive as keyword arguments.

### 4. Validate behavior

Check reset/step space conformance, action masks for ready and moving robots, physical-time truncation at `episode_len`, metric finalization, deterministic seeding, and learned-policy evaluation reconstruction.

## Adding a Network

### 1. Define its configuration

Add a dataclass under `configs/network_configs.py` for architecture parameters. Keep actor, critic, and Q-network types separate even when they share a base configuration.

### 2. Implement reconstruction

A checkpointable network must expose stable `input_dim`/`output_dim` metadata and implement:

```python
def get_config_dict(self, input_dim: int, output_dim: int) -> dict: ...

@classmethod
def from_config_dict(cls, cfg: dict): ...
```

The saved dictionary must include a unique `type` name and every parameter needed to reconstruct the module before loading its state dict. Recurrent networks should also provide the hidden-state and sequence-forward interfaces used by the policy wrappers.

### 3. Register both construction paths

Add the YAML key to `ACTOR_REGISTRY`, `CRITIC_REGISTRY`, or `Q_NETWORK_REGISTRY` in `configs/registry.py`. If the constructor is not covered by an existing factory branch, extend `create_actor`, `create_critic`, or `create_q_network` narrowly.

Also add the class name to `utils/model_io._get_class_registry`; otherwise training can save the model but evaluation cannot rebuild it.

### 4. Exercise save and load

Train or instantiate once, save a checkpoint, load it through `load_policy_for_eval`, and compare deterministic outputs on the same observation and action mask.

## Adding an Algorithm

### 1. Add parameters and implementation

Create an algorithm parameter dataclass in `configs/algo_configs.py`. Implement the algorithm under `algorithms/rl/`, `algorithms/marl/`, or `algorithms/tabular/`, normally by extending `BaseAlgorithm`, `ActorCriticOnPolicyAlgo`, `PPOBase`, or `QLearningOffPolicyAlgo`.

The trainer-facing implementation must prepare the collector's batch format, perform an update, expose training statistics, and preserve target/recurrent state where applicable.

### 2. Register the algorithm

Add an `ALGO_REGISTRY` entry containing:

```python
"my_algo": {
    "module": "algorithms.marl.my_algo",
    "class_name": "MyAlgo",
    "params_class": MyAlgoParams,
    "trainer_type": "on_policy",  # or off_policy / tabular path
    "policy_type": "actor",       # actor, value, or mat
}
```

`create_algorithm` inspects the constructor and forwards matching values from the context assembled in `train.py`. Add a new context value there only if the algorithm requires information not already available.

### 3. Connect policy and collector semantics

Choose whether the algorithm needs actor or value policies, independent or shared parameters, centralized state, synchronized replay indices, recurrent sequences, and active masks. If these requirements differ from existing families, extend `_build_policy` or `_build_collector` explicitly rather than hiding the behavior in a YAML field that no component consumes.

### 4. Add runnable configurations and verify

Provide at least one experiment/evaluation pair, run a short collection/update smoke test, verify checkpoint reconstruction, then test the full evaluator. For a MARL method, test both action-mask handling and agent ordering; for an event-driven method, also test inactive-step treatment.
