"""
Paper Implementation: Reinforcement Learning for Multi-agent Patrol Policy
Authors: Zhaohui Hu, Dongbin Zhao
Year: 2010
Venue: IEEE International Conference on Cognitive Informatics (ICCI)
Link: https://ieeexplore.ieee.org/document/5599681

Description:
    This script implements the NEP(short for Node-Edge Position) mdp design 
    described in Section 3.C of the paper.
"""

from typing import Dict, Optional
import numpy as np
import gymnasium
from gymnasium.spaces import Box, Discrete

from envs.mdps.patrol_core import PatrolWorld, TickResult

class NEPEnv(gymnasium.Env):
    

