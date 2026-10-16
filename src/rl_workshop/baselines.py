"""Pure-Python baseline controllers used throughout the workshop."""
from __future__ import annotations
import numpy as np
from .navigation import DIRECTIONS, NavigationEnv


def random_action(env, observation=None) -> int:
    return int(env.action_space.sample())


def greedy_action(env: NavigationEnv, observation=None) -> int:
    """Head directly to target; intentionally ignores obstacles."""
    vector = env.goal - env.position
    return int(np.argmax(DIRECTIONS @ vector))


def avoid_obstacles_action(env: NavigationEnv, observation=None) -> int:
    """One-step model-based heuristic; still not a full planner."""
    scores = []
    for direction in DIRECTIONS:
        next_pos = env.position + direction * env.step_size
        score = -float(np.linalg.norm(env.goal - next_pos))
        if np.any(next_pos < 0) or np.any(next_pos > 10):
            score -= 100.0
        if env.obstacles_enabled:
            clearance = np.linalg.norm(env.centers - next_pos, axis=1) - env.obstacle_radius
            if np.any(clearance < 0.4):
                score -= 50.0
            score += 0.02 * float(np.clip(clearance.min(), 0, 3))
        scores.append(score)
    return int(np.argmax(scores))
