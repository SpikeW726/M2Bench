from polocies.heuritic.HeuristicBase import HeuriticBasePolicy
from polocies.heuritic.ER import ERPolicy
from envs.BaseEnvs import EventDrivenEnv
from envs.MASUPEnv import MASUPEnv
import yaml

class HeuristicSampler:
    """暂时只支持 ERPolicy + MASUPEnv """
    def __init__(self, policy:HeuriticBasePolicy, env:EventDrivenEnv) -> None:
        self.policy = policy
        self.env = env


if __name__ == "__main__":
    ER_config = yaml.load()
    MASUP_config = yaml.load()
    num_agents = MASUP_config.get("num_agents", 3)

    policy = ERPolicy(num_agents, ER_config)
    env = MASUPEnv(MASUP_config)