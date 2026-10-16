"""Lesson 1 grid world: a small finite Markov decision process.

No Ray, Gymnasium, or flight physics required. The API resembles Gymnasium:
reset() returns (state, info); step() returns (state, reward, terminated,
truncated, info). Here an observation *is* an integer state ID.
"""
from dataclasses import dataclass

import numpy as np

# row, column displacements. Row increases down the page.
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
ACTION_SYMBOLS = ("↑", "→", "↓", "←")


@dataclass(frozen=True)
class GridConfig:
    rows: int = 7
    cols: int = 7
    start: tuple[int, int] = (1, 1)
    goal: tuple[int, int] = (1, 5)
    # An almost-solid wall forces a detour through row 4.
    walls: tuple[tuple[int, int], ...] = tuple((r, 3) for r in range(7) if r != 4)
    step_penalty: float = -0.04
    wall_penalty: float = -0.45
    goal_reward: float = 10.0
    max_steps: int = 45


class GridWorld:
    def __init__(self, config: GridConfig | None = None, seed: int = 0):
        self.config = config or GridConfig()
        self.rng = np.random.default_rng(seed)
        self.walls = set(self.config.walls)
        self.valid_starts = [
            (r, c)
            for r in range(self.config.rows)
            for c in range(self.config.cols)
            if (r, c) not in self.walls and (r, c) != self.config.goal
        ]
        assert self.config.start in self.valid_starts
        assert self.config.goal not in self.walls
        assert self.config.max_steps > 0
        self.position = self.config.start
        self.steps = 0

    @property
    def num_states(self) -> int:
        return self.config.rows * self.config.cols

    @property
    def num_actions(self) -> int:
        return len(ACTIONS)

    def encode(self, position: tuple[int, int]) -> int:
        return position[0] * self.config.cols + position[1]

    def decode(self, state: int) -> tuple[int, int]:
        return divmod(int(state), self.config.cols)

    def reset(self, *, seed: int | None = None,
              start: tuple[int, int] | None = None) -> tuple[int, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if start is None:
            start = self.valid_starts[int(self.rng.integers(len(self.valid_starts)))]
        if start not in self.valid_starts:
            raise ValueError(f"Invalid start position: {start}")
        self.position = start
        self.steps = 0
        return self.encode(self.position), {"position": self.position}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        if not 0 <= int(action) < self.num_actions:
            raise ValueError(f"Action must be 0..{self.num_actions - 1}: {action}")
        dr, dc = ACTIONS[int(action)]
        r, c = self.position
        proposed = (r + dr, c + dc)
        nr, nc = proposed
        bumped = not (0 <= nr < self.config.rows and 0 <= nc < self.config.cols)
        bumped = bumped or proposed in self.walls
        if not bumped:
            self.position = proposed
        self.steps += 1
        terminated = self.position == self.config.goal
        truncated = self.steps >= self.config.max_steps and not terminated
        reward = self.config.step_penalty
        if bumped:
            reward += self.config.wall_penalty
        if terminated:
            reward += self.config.goal_reward
        return self.encode(self.position), float(reward), terminated, truncated, {
            "success": terminated,
            "bumped": bumped,
            "position": self.position,
            "steps": self.steps,
        }

    def render_ascii(self, path: list[tuple[int, int]] | None = None) -> str:
        """Human-readable map. '*' marks visited cells except start, goal, walls."""
        visited = set(path or [])
        lines = []
        for r in range(self.config.rows):
            cells = []
            for c in range(self.config.cols):
                pos = (r, c)
                if pos == self.config.start:
                    glyph = "S"
                elif pos == self.config.goal:
                    glyph = "G"
                elif pos in self.walls:
                    glyph = "#"
                elif pos == self.position and path is None:
                    glyph = "A"
                elif pos in visited:
                    glyph = "*"
                else:
                    glyph = "."
                cells.append(glyph)
            lines.append(" ".join(cells))
        return "\n".join(lines)
