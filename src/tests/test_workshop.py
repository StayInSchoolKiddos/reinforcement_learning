"""Run from repo root: uv run --no-sync python -m pytest tests -q"""
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env
from rl_workshop.navigation import NavigationEnv, DIRECTIONS
from rl_workshop.mission import MissionCommanderEnv, safety_supervisor, PROCEED, REPLAN, HOLD, RETURN
from rl_workshop.baselines import greedy_action
from rl_workshop.rollouts import run_episode


def test_navigation_gym_contract():
    env = NavigationEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    for _ in range(120):
        obs, reward, terminated, truncated, info = env.step(greedy_action(env, obs))
        assert obs.shape == (8,)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_navigation_seed_replay():
    e1, e2 = NavigationEnv(), NavigationEnv()
    x1, _ = e1.reset(seed=123)
    x2, _ = e2.reset(seed=123)
    assert np.array_equal(x1, x2)
    for a in [0, 1, 2, 2, 1]:
        a1 = e1.step(a)
        a2 = e2.step(a)
        assert np.array_equal(a1[0], a2[0])
        assert a1[1:] == a2[1:]


def test_navigation_goal_terminal():
    env = NavigationEnv({"obstacles": False})
    env.reset(seed=1, options={"start": [8.0, 8.0], "goal": [8.4, 8.0]})
    _, reward, term, trunc, info = env.step(0)
    assert term and not trunc and info["success"] and reward > 0


def test_navigation_rollout_has_metrics():
    result = run_episode(NavigationEnv(), lambda e, o: greedy_action(e, o), seed=101)
    assert result["steps"] >= 1
    assert isinstance(result["success"], bool)


def test_mission_supervisor_vetos():
    assert safety_supervisor(PROCEED, np.array([.2, .8, .8, .8])) == REPLAN
    assert safety_supervisor(PROCEED, np.array([.2, .8, .2, .1])) == HOLD
    assert safety_supervisor(PROCEED, np.array([.2, .1, .2, .8])) == RETURN


def test_mission_contract_and_proposal_tracking():
    env = MissionCommanderEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(options={"state": [.2, .8, .8, .1]}, seed=3)
    obs, reward, term, trunc, info = env.step(PROCEED)
    assert info["intervened"] and info["proposed"] == "PROCEED"
    assert info["executed"] == "REPLAN"
    assert env.observation_space.contains(obs)
    assert not term


def test_pursuit_contract():
    pytest.importorskip("ray")
    from rl_workshop.pursuit_evasion import PursuitEvasionEnv
    env = PursuitEvasionEnv({"max_steps": 2})
    obs, infos = env.reset(seed=2)
    assert set(obs) == {"pursuer", "evader"}
    for agent, o in obs.items():
        assert env.get_observation_space(agent).contains(o)
    for _ in range(2):
        obs, rewards, term, trunc, infos = env.step({"pursuer": 0, "evader": 0})
    assert term["__all__"] or trunc["__all__"]
    assert env.agents == []
