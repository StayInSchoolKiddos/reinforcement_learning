"""Simplified mission-choice MDP with a separate supervisory veto layer.

Abstract toy dynamics. Nothing in this file should command a real aircraft.
"""
from __future__ import annotations
from collections.abc import Mapping
import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTIONS = ("PROCEED", "REPLAN", "HOLD", "RETURN")
PROCEED, REPLAN, HOLD, RETURN = range(4)


def safety_supervisor(proposed: int, observation: np.ndarray) -> int:
    """Always applied independently of the learned policy."""
    progress, fuel, threat, link = map(float, observation)
    if proposed == PROCEED:
        if fuel <= 0.12:
            return RETURN
        if threat >= 0.75:
            return REPLAN
        if link <= 0.18:
            return HOLD
    return proposed


def rule_based_commander(observation: np.ndarray) -> int:
    progress, fuel, threat, link = map(float, observation)
    if fuel < 0.13:
        return RETURN
    if threat > 0.58:
        return REPLAN
    if link < 0.32:
        return HOLD
    return PROCEED


class MissionCommanderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Mapping | None = None):
        super().__init__()
        cfg = dict(config or {})
        self.max_steps = int(cfg.get("max_steps", 18))
        self.supervisor_enabled = bool(cfg.get("supervisor_enabled", True))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.state = np.array([
            self.np_random.uniform(0.0, 0.15),
            self.np_random.uniform(0.85, 1.0),
            self.np_random.uniform(0.15, 0.70),
            self.np_random.uniform(0.42, 0.95),
        ], dtype=np.float32)
        if options and "state" in options:
            self.state = np.asarray(options["state"], dtype=np.float32).copy()
        return self.state.copy(), {"note": "Toy mission state [progress,fuel,threat,link]"}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid mission action: {action}")
        proposed = int(action)
        executed = safety_supervisor(proposed, self.state) if self.supervisor_enabled else proposed
        progress, fuel, threat, link = map(float, self.state)
        self.steps += 1
        reward = -0.03  # decision-cost baseline
        success = failure = returned = False
        if executed == PROCEED:
            progress += 0.14
            fuel -= 0.075
            threat += 0.09
            link -= 0.10
            reward += 1.6
            if threat > 0.92 or link < 0.08:
                failure = True
                reward -= 10.0
        elif executed == REPLAN:
            fuel -= 0.045
            threat -= 0.36
            link -= 0.02
            reward -= 0.25
        elif executed == HOLD:
            fuel -= 0.025
            link += 0.30
            threat -= 0.04
            reward -= 0.18
        elif executed == RETURN:
            returned = True
            reward -= 2.5  # safe early return is preferable to mission failure
        self.state = np.clip([progress, fuel, threat, link], 0.0, 1.0).astype(np.float32)
        if self.state[1] <= 0.0:
            failure = True
            reward -= 10.0
        if progress >= 1.0 and not failure:
            success = True
            reward += 15.0
        if proposed != executed:
            reward -= 0.4  # explicit feedback for an unsafe proposed decision
        terminated = bool(success or failure or returned)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        return self.state.copy(), float(reward), terminated, truncated, {
            "proposed": ACTIONS[proposed], "executed": ACTIONS[executed],
            "intervened": proposed != executed, "success": success,
            "failure": failure, "returned": returned, "steps": self.steps,
        }
