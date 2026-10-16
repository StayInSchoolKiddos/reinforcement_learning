"""Tabular Q-learning for Lesson 1. NumPy only; no black-box RL library."""
from dataclasses import dataclass

import numpy as np

from .gridworld import GridWorld


@dataclass
class TrainingResult:
    q: np.ndarray
    episode_returns: np.ndarray
    successes: np.ndarray
    lengths: np.ndarray
    epsilons: np.ndarray
    wall_bumps: np.ndarray


def choose_action(q: np.ndarray, state: int, rng: np.random.Generator,
                  epsilon: float = 0.0) -> int:
    """Epsilon-greedy: random with probability epsilon, otherwise argmax Q.

    Ties are intentionally broken by the lowest action index to make
    the 'no exploration' experiment transparent. This is NOT optimal practice.
    """
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    return int(np.argmax(q[state]))

def train_agent(
    episodes: int = 1200,
    seed: int = 7,
    alpha: float = 0.25,
    gamma: float = 0.97,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    env: GridWorld | None = None,
) -> TrainingResult:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if not (0 <= epsilon_start <= 1 and 0 <= epsilon_end <= 1):
        raise ValueError("epsilon must be in [0,1]")
    if env is None:
        env = GridWorld(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    q = np.zeros((env.num_states, env.num_actions), dtype=np.float64)
    returns = np.empty(episodes, dtype=np.float64)
    successes = np.zeros(episodes, dtype=bool)
    lengths = np.empty(episodes, dtype=np.int64)
    epsilons = np.empty(episodes, dtype=np.float64)
    bumps = np.zeros(episodes, dtype=np.int64)
    anneal_over = max(1, int(0.75 * episodes))

    for episode in range(episodes):
        # Linear exploration schedule over the first 75% of training.
        progress = min(1.0, episode / anneal_over)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        epsilons[episode] = epsilon
        state, _ = env.reset()
        episode_return = 0.0
        for t in range(env.config.max_steps):
            action = choose_action(q, state, rng, epsilon=epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Important: a time limit is truncation, not a terminal goal state;
            # bootstrap through truncations for the continuing MDP approximation.
            future_value = 0.0 if terminated else gamma * float(np.max(q[next_state]))
            target = reward + future_value
            q[state, action] += alpha * (target - q[state, action])

            episode_return += reward
            bumps[episode] += int(info["bumped"])
            state = next_state
            if terminated or truncated:
                successes[episode] = terminated
                lengths[episode] = t + 1
                break
        returns[episode] = episode_return
    return TrainingResult(q, returns, successes, lengths, epsilons, bumps)


def rollout(env: GridWorld, q: np.ndarray | None, start: tuple[int, int],
            *, seed: int = 0, random_policy: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    state, _ = env.reset(start=start)
    path = [env.position]
    total_reward = 0.0
    bumps = 0
    for t in range(env.config.max_steps):
        if random_policy:
            action = int(rng.integers(env.num_actions))
        else:
            if q is None:
                raise ValueError("q table is required for greedy policy")
            action = choose_action(q, state, rng)
        state, reward, terminated, truncated, info = env.step(action)
        path.append(env.position)
        bumps += int(info["bumped"])
        total_reward += reward
        if terminated or truncated:
            break
    return {
        "success": bool(terminated), "return": float(total_reward),
        "wall_bumps": bumps, "steps": len(path) - 1,
        "path": path, "start": start,
    }


def evaluate(q: np.ndarray | None, *, starts: list[tuple[int, int]],
             seed: int = 1, random_policy: bool = False,
             env: GridWorld | None = None) -> dict:
    """Evaluate on identical fixed start positions for fair policy comparison."""
    if env is None:
        env = GridWorld()
    outcomes = [
        rollout(env, q, start=start, seed=seed + i, random_policy=random_policy)
        for i, start in enumerate(starts)
    ]
    return {
        "episodes": len(outcomes),
        "success_rate": float(np.mean([e["success"] for e in outcomes])),
        "mean_return": float(np.mean([e["return"] for e in outcomes])),
        "mean_steps": float(np.mean([e["steps"] for e in outcomes])),
        "mean_wall_bumps": float(np.mean([e["wall_bumps"] for e in outcomes])),
    }


def evaluation_starts(env: GridWorld, *, count: int = 60,
                      seed: int = 2026) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    return [env.valid_starts[int(i)] for i in rng.integers(len(env.valid_starts), size=count)]
