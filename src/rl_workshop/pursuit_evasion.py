"""Two-agent simultaneous pursuit/evasion with discrete heading commands.

Teaching simulation only; agents move instantaneously and have no real aircraft dynamics.
"""
from __future__ import annotations
from collections.abc import Mapping
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from .navigation import DIRECTIONS


class PursuitEvasionEnv(MultiAgentEnv):
    def __init__(self, config: Mapping | None = None):
        super().__init__()
        cfg = dict(config or {})
        self.max_steps = int(cfg.get("max_steps", 35))
        self.step_sizes = {
            "pursuer": float(cfg.get("pursuer_step_size", 0.50)),
            "evader": float(cfg.get("evader_step_size", 0.35)),
        }
        self.possible_agents = ["pursuer", "evader"]
        self.agents = self.possible_agents.copy()
        self._agent_ids = set(self.possible_agents)
        self.observation_spaces = {a: spaces.Box(-1, 1, (6,), dtype=np.float32)
                                   for a in self.possible_agents}
        self.action_spaces = {a: spaces.Discrete(len(DIRECTIONS)) for a in self.possible_agents}
        self.capture_radius = 0.55
        self.safe_radius = 0.65
        self.safe_zone = np.array([9.0, 5.0], dtype=np.float32)
        self.positions = {}
        self.steps = 0

    def _observation(self, who):
        mine = self.positions[who]
        other = self.positions["evader" if who == "pursuer" else "pursuer"]
        return np.concatenate((mine / 10, (other - mine) / 10,
                               (self.safe_zone - mine) / 10)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.agents = self.possible_agents.copy()
        self.steps = 0
        self.positions = {
            "pursuer": np.array([5.0, 3.9], dtype=np.float32) + self.np_random.uniform(-0.5, 0.5, 2),
            "evader": np.array([6.0, 5.0], dtype=np.float32) + self.np_random.uniform(-0.5, 0.5, 2),
        }
        self.positions = {k: v.astype(np.float32) for k, v in self.positions.items()}
        return {agent: self._observation(agent) for agent in self.agents}, {a: {} for a in self.agents}

    def step(self, action_dict):
        if set(action_dict) != set(self.agents):
            raise ValueError(f"Need simultaneous actions for {self.agents}; received {list(action_dict)}")
        before_sep = float(np.linalg.norm(self.positions["pursuer"] - self.positions["evader"]))
        before_safe = float(np.linalg.norm(self.safe_zone - self.positions["evader"]))
        for agent in self.agents:
            action = int(action_dict[agent])
            if not self.action_spaces[agent].contains(action):
                raise ValueError(f"Bad {agent} action: {action}")
            self.positions[agent] = np.clip(self.positions[agent] + DIRECTIONS[action] * self.step_sizes[agent], 0, 10)
        self.steps += 1
        sep = float(np.linalg.norm(self.positions["pursuer"] - self.positions["evader"]))
        safe_dist = float(np.linalg.norm(self.safe_zone - self.positions["evader"]))
        caught = sep <= self.capture_radius
        escaped = safe_dist <= self.safe_radius and not caught
        done = bool(caught or escaped)
        timeout = bool(self.steps >= self.max_steps and not done)
        rewards = {
            "pursuer": 1.5 * (before_sep - sep) - 0.01 + (12 if caught else -12 if escaped else 0),
            "evader": 1.5 * (before_safe - safe_dist) - 0.01 + (12 if escaped else -12 if caught else 0),
        }
        infos = {a: {"caught": caught, "escaped": escaped,
                     "separation": sep, "safe_distance": safe_dist, "steps": self.steps}
                 for a in self.agents}
        terminated = {a: done for a in self.agents}
        terminated["__all__"] = done
        truncated = {a: timeout for a in self.agents}
        truncated["__all__"] = timeout
        observations = {a: self._observation(a) for a in self.agents}
        if done or timeout:
            self.agents = []
        return observations, rewards, terminated, truncated, infos

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]
