"""NumPy-only tests for revised Lesson 1; run without Gymnasium or Ray."""
import numpy as np
import pytest
from lessons.lesson01_agent_environment.gridworld import GridConfig, GridWorld
from lessons.lesson01_agent_environment.q_learning import (
    choose_action, train_agent, evaluate, evaluation_starts, rollout,
)


def test_grid_state_transition_wall_and_goal():
    env = GridWorld()
    s, _ = env.reset(start=(1, 2))
    assert s == env.encode((1, 2))
    new_state, reward, done, trunc, info = env.step(1)  # wall at (1,3)
    assert new_state == s and info["bumped"] and not done and not trunc
    assert np.isclose(reward, -.49)
    env.reset(start=(1, 4))
    next_state, reward, done, trunc, info = env.step(1)
    assert next_state == env.encode((1, 5))
    assert done and not trunc and info["success"]
    assert np.isclose(reward, 9.96)


def test_short_episode_truncates_without_success():
    env = GridWorld(GridConfig(max_steps=1))
    env.reset(start=(1, 1))
    _, _, done, trunc, info = env.step(0)
    assert not done and trunc and not info["success"]


def test_greedy_is_deterministic_on_ties():
    q = np.zeros((49, 4))
    action = choose_action(q, 0, np.random.default_rng(5), epsilon=0)
    assert action == 0


def test_learning_update_and_reproducibility():
    r1 = train_agent(episodes=180, seed=7)
    r2 = train_agent(episodes=180, seed=7)
    assert np.array_equal(r1.q, r2.q)
    assert np.any(np.abs(r1.q) > 0)
    assert r1.q.shape == (49, 4)
    starts = evaluation_starts(GridWorld(), count=60, seed=2026)
    before = evaluate(np.zeros_like(r1.q), starts=starts)
    after = evaluate(r1.q, starts=starts)
    assert after["success_rate"] > before["success_rate"]
    assert after["mean_return"] > before["mean_return"]


def test_rollout_reaches_goal_with_pretrained_table():
    result = train_agent(episodes=400, seed=7)
    env = GridWorld()
    rollout_data = rollout(env, result.q, start=(1, 1))
    assert rollout_data["success"] is True
    assert rollout_data["steps"] <= 14


def test_invalid_configuration():
    with pytest.raises(ValueError):
        train_agent(episodes=0)
    with pytest.raises(ValueError):
        train_agent(episodes=1, epsilon_start=1.5)
