# Reinforcement Learning Lunch & Learn — Autonomous Decision-Making

**Six project-based, 60-minute workshops for Python-capable engineers.**

A single learning arc: **watch tabular Q-learning discover a route through a grid maze**, specify and validate a richer Gymnasium MDP, train PPO, challenge reward design, progress to multi-agent pursuit/evasion, and build a *toy* high-level mission commander. Emphasis is on **reasoning about experiments**, not copying PPO code.

## Setup: use your existing dev container

This folder is an **overlay** for an existing Python 3.12 + `uv` development container; it does **not** replace your Dockerfile, `pyproject.toml`, CUDA drivers, or installed PyTorch build. From your existing repository, copy `rl_workshop/`, `lessons/`, and `tests/` into its root; alternatively, put this whole folder anywhere *inside* that repository and `cd` into it. Run commands from the folder containing `rl_workshop/`.

Dependencies expected from your existing container: `gymnasium`, `numpy`, `matplotlib`, `ray[rllib]==2.59.0`, `torch`, and `pytest`. **Lesson 1 only needs NumPy + Matplotlib** and does not import Gymnasium or Ray. Only lessons **3–6 training/evaluation** and lesson **5** require Ray. Existing PyTorch CPU or NVIDIA CUDA wheels are fine; scripts explicitly train on CPU so one workstation can run the exercises. Training is slower than the non-Ray demos.

```bash
# From the directory containing rl_workshop/ and lessons/:
uv run --no-sync python -c "import gymnasium, ray, torch; print('Ray',ray.__version__)"
uv run --no-sync python -m pytest tests -q
uv run --no-sync python -m lessons.lesson01_agent_environment.demo --episodes 1200
uv run --no-sync python -m lessons.lesson02_design_environment.inspect_env
uv run --no-sync python -m lessons.lesson03_train_ppo.train --iterations 12
uv run --no-sync python -m lessons.lesson03_train_ppo.evaluate
uv run --no-sync python -m lessons.lesson04_reward_engineering.experiment --iterations 10
uv run --no-sync python -m lessons.lesson05_multi_agent.demo
uv run --no-sync python -m lessons.lesson05_multi_agent.train --iterations 15
uv run --no-sync python -m lessons.lesson05_multi_agent.evaluate
uv run --no-sync python -m lessons.lesson06_mission_commander.demo
uv run --no-sync python -m lessons.lesson06_mission_commander.train --iterations 15
uv run --no-sync python -m lessons.lesson06_mission_commander.evaluate
```

If running this *standalone outside* your configured project, install the dependencies in your own environment first and use `python -m ...` instead of `uv run --no-sync ...`. The scripts deliberately do not install or choose a CUDA version.

**Do not run all PPO training lessons in a single class period.** Precompute checkpoints on the instructor workstation, then use live runs of `--iterations 2` and show precomputed results for the full discussion. Demo/training results are stochastic; short runs might not converge.

## Teaching roadmap

| Lesson | Concept / engineering question | Execute | Artifact |
|:---|:---|:---|:---|
| [01](lessons/lesson01_agent_environment/README.md) | **Can an agent learn a route without us coding it?** Tabular Q-learning, exploration, Q-update | `...lesson01_agent_environment.demo` or [notebook](lessons/lesson01_agent_environment/Lesson_01_Q_Learning.ipynb) | Before/after route, training curve, learned Q-table, evaluation metrics |
| [02](lessons/lesson02_design_environment/README.md) | How do we define and validate an MDP? | `...lesson02_design_environment.inspect_env` | Tested Gymnasium env + diagram |
| [03](lessons/lesson03_train_ppo/README.md) | Does learning outperform hand-written rules? | `...lesson03_train_ppo.train`, `.evaluate` | Checkpoint, learning curve, held-out score |
| [04](lessons/lesson04_reward_engineering/README.md) | What does the reward actually encourage? | `...lesson04_reward_engineering.experiment` | Controlled reward-mode comparison CSV |
| [05](lessons/lesson05_multi_agent/README.md) | What changes when both agents learn? | `...lesson05_multi_agent.demo`, `.train`, `.evaluate` | Pursuit/evasion outcomes |
| [06](lessons/lesson06_mission_commander/README.md) | How can policy choose *mission actions*, not actuator values? | `...lesson06_mission_commander.demo`, `.train`, `.evaluate` | Mission outcome/veto-rate comparison |

Every lesson directory contains a student `README.md`, runnable script(s), and `INSTRUCTOR.md` with a minute-by-minute outline and teaching notes. Lesson 1 also includes a full guided `.ipynb` notebook. The original scripted-controller comparison is now an **optional** Lesson 2 script: `lessons.lesson02_design_environment.compare_controllers`. [INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md) is the teaching overview.

## Shared files

- `lessons/lesson01_agent_environment/gridworld.py`, `q_learning.py`: **self-contained NumPy tabular Q-learning** over a 7×7 grid. Do not confuse this discrete example with the later Gymnasium environment.
- `rl_workshop/navigation.py`: Lesson 2+ Gymnasium MDP, deliberately simple 8-way instantaneous heading moves.
- `rl_workshop/baselines.py`: random, direct-to-goal, short-horizon obstacle-avoidance reference strategies.
- `rl_workshop/rollouts.py`: fixed-seed evaluation; tracks successes, collisions and returns.
- `rl_workshop/viz.py`: trajectory and learning-curve image output (headless).
- `rl_workshop/rllib_helpers.py`: **RLlib 2.59 new API** PPOConfig, RLModule inference, checkpoint helpers.
- `rl_workshop/pursuit_evasion.py`: simultaneous two-policy multi-agent MDP.
- `rl_workshop/mission.py`: abstract mission-state MDP and independent action veto.
- `tests/`: API/shape, repeatability, termination, and safety checks.

## If your repository uses `src/lessons/`

From your project root (`/workspace`), use:

```bash
PYTHONPATH=src uv run --no-sync python -m lessons.lesson01_agent_environment.demo --episodes 1200
```

If you copy the overlay under `src/`, the module path must include `PYTHONPATH=src` (or a package install exposing `lessons`); do **not** run `uv run src/lessons/lesson01_agent_environment/`, which tries to execute a directory.

## Ground rules for experiments

1. **Predict before running**; change one variable at a time.
2. Report at least **success rate, failure rate, mean return and episode length** for fixed held-out seeds. Do not use training return as a substitute for success.
3. Use equivalent evaluation states and environment conditions for policy comparisons; distinguish training rewards from evaluation rewards.
4. A training curve is *not* evidence of generalization. Compare multiple random seeds when time permits.
5. Simulation simplifications are explicit: no 6-DOF, autopilot, communications delays, aircraft limits, operational validation, or deployment path. Mission commander actions are *abstract* and never leave the simulator.

## RLlib version/API notes

Built against Ray/RLlib **2.59.0 new API stack**: `PPOConfig().env_runners(...)`, `.learners(...)`, `train_batch_size_per_learner`, `Algorithm.save_to_path()`, `Algorithm.from_checkpoint()` and `RLModule.forward_inference()`. These differ from old `.rollouts()`, `num_workers`, or `compute_single_action()` examples. The code assumes flat numeric observations and discrete actions; the simple `argmax` action inference helper is not a universal inference implementation. If your container has a different version of Ray, pin/check API first.

### Primary references

- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (Ch. 1, **6.5** for tabular Q-learning; Ch. 3 and 13 for later lessons): http://incompleteideas.net/book/the-book-2nd.html
- Gymnasium environment creation: https://gymnasium.farama.org/introduction/create_custom_env/
- RLlib environments: https://docs.ray.io/en/latest/rllib/rllib-env.html
- RLlib new API migration: https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html
- RLlib multi-agent environments: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html

`outputs/` is created by examples and intentionally ignored in Git. Trained checkpoints are **not** bundled; training must run first or your instructor must provide them.
