"""Minimal, deliberately simplified 2-D heading-command navigation MDP.

No vehicle inertia, bank constraints, wind, or autopilot: this is NOT a flight model.
Coordinates are (east, north), in a 10 x 10 arbitrary-unit square.
"""
from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# East, northeast, north, northwest, west, southwest, south, southeast.
DIRECTIONS = np.array(
    [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0],
     [-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
    dtype=np.float32,
)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)


class NavigationEnv(gym.Env):
    """Choose an 8-way heading at each step to reach a goal without collisions.

    obs = [x/10, y/10, (goal - pos)/10, (obs1 - pos)/10, (obs2 - pos)/10]
    action = Discrete(8), indexed by DIRECTIONS.
    rewards/termination are fully exposed for class exercises.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 6}

    def __init__(self, config: Mapping | None = None):
        super().__init__()
        cfg = dict(config or {})
        self.reward_mode = cfg.get("reward_mode", "dense")
        if self.reward_mode not in {"dense", "sparse", "risky"}:
            raise ValueError("reward_mode must be dense, sparse, or risky")
        self.obstacles_enabled = bool(cfg.get("obstacles", True))
        self.max_steps = int(cfg.get("max_steps", 90))
        self.step_size = float(cfg.get("step_size", 0.45))
        self.goal_radius = 0.60
        self.obstacle_radius = 0.90
        self.centers = np.array([[5.0, 5.0], [5.0, 7.4]], dtype=np.float32)
        self.action_space = spaces.Discrete(len(DIRECTIONS))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.position = np.array([1.0, 1.0], dtype=np.float32)
        self.goal = np.array([8.0, 8.0], dtype=np.float32)
        self.steps = 0
        self.path: list[np.ndarray] = []

    def _get_observation(self) -> np.ndarray:
        return np.clip(np.concatenate((self.position / 10.0,
                               (self.goal - self.position) / 10.0,
                               ((self.centers - self.position) / 10.0).ravel())), -1.0, 1.0).astype(np.float32)

    def _distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.position = self.np_random.uniform(0.8, 2.0, size=2).astype(np.float32)
        self.goal = self.np_random.uniform(8.0, 9.1, size=2).astype(np.float32)
        if "start" in options:
            self.position = np.asarray(options["start"], dtype=np.float32).copy()
        if "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=np.float32).copy()
        self.steps = 0
        self.path = [self.position.copy()]
        return self._get_observation(), {"distance": self._distance()}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid discrete action: {action}")
        previous_distance = self._distance()
        self.position += DIRECTIONS[int(action)] * self.step_size
        self.position = self.position.astype(np.float32)
        self.steps += 1
        self.path.append(self.position.copy())
        distance = self._distance()
        out_of_bounds = bool(np.any(self.position < 0.0) or np.any(self.position > 10.0))
        collision = self.obstacles_enabled and bool(
            np.any(np.linalg.norm(self.centers - self.position, axis=1) <= self.obstacle_radius)
        )
        success = distance <= self.goal_radius
        terminated = bool(success or collision or out_of_bounds)
        truncated = bool(self.steps >= self.max_steps and not terminated)

        # Lesson 04: modify *one* term at a time and evaluate under common conditions.
        if self.reward_mode == "sparse":
            reward = 0.0
        else:
            progress = previous_distance - distance
            reward = (3.0 if self.reward_mode == "risky" else 2.0) * progress - 0.025
        if success:
            reward += 15.0
        elif collision or out_of_bounds:
            reward -= 0.5 if self.reward_mode == "risky" else 8.0
        return self._get_observation(), float(reward), terminated, truncated, {
            "success": bool(success), "collision": bool(collision),
            "out_of_bounds": bool(out_of_bounds), "distance": distance,
            "steps": self.steps,
        }

    def render(self):
        """Small RGB array; plot_episode() in rl_workshop.viz is easier for class."""
        size = 320
        image = np.full((size, size, 3), 250, dtype=np.uint8)
        def px(p):
            return np.clip((np.asarray(p) * (size - 1) / 10).astype(int), 0, size - 1)
        yy, xx = np.ogrid[:size, :size]
        for center in self.centers if self.obstacles_enabled else []:
            cx, cy = px(center)
            image[(xx - cx) ** 2 + (yy - cy) ** 2 <= (self.obstacle_radius * size / 10) ** 2] = [235, 125, 125]
        for p in self.path:
            x, y = px(p)
            image[max(0, y-2):y+3, max(0, x-2):x+3] = [40, 100, 210]
        x, y = px(self.goal)
        image[max(0, y-5):y+6, max(0, x-5):x+6] = [40, 170, 80]
        return np.flipud(image)
