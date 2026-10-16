"""Independent evaluation of agents, not RLlib training metrics."""
from __future__ import annotations
from collections.abc import Callable
import numpy as np


def run_episode(env, policy: Callable, seed: int) -> dict:
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    ret, info = 0.0, {}
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(policy(env, observation))
        observation, reward, terminated, truncated, info = env.step(action)
        ret += reward
    return {"return": float(ret), "steps": int(env.steps),
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "truncated": bool(truncated)}


def evaluate(env_factory: Callable, policy: Callable, episodes: int = 30, seed: int = 1000):
    records = []
    for i in range(episodes):
        env = env_factory()
        try:
            records.append(run_episode(env, policy, seed + i))
        finally:
            env.close()
    return {"episodes": episodes,
            "mean_return": round(float(np.mean([x["return"] for x in records])), 3),
            "success_rate": round(float(np.mean([x["success"] for x in records])), 3),
            "collision_rate": round(float(np.mean([x["collision"] for x in records])), 3),
            "mean_steps": round(float(np.mean([x["steps"] for x in records])), 2)}
