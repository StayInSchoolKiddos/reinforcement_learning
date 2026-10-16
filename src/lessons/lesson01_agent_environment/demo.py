"""Lesson 01 — Learn (not hand-code) a policy with tabular Q-learning.

From repository root:
    uv run --no-sync python -m lessons.lesson01_agent_environment.demo
    uv run --no-sync python -m lessons.lesson01_agent_environment.demo --episodes 300
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from .gridworld import GridConfig, GridWorld
from .plots import plot_before_after, plot_policy, plot_training
from .q_learning import evaluate, evaluation_starts, rollout, train_agent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=1200, help="number of training episodes")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--alpha", type=float, default=0.25, help="Q-learning step size")
    p.add_argument("--gamma", type=float, default=0.97, help="future reward discount")
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--step-penalty", type=float, default=-0.04)
    p.add_argument("--wall-penalty", type=float, default=-0.45)
    p.add_argument("--goal-reward", type=float, default=10.0)
    p.add_argument("--output", type=Path, default=Path("outputs/lesson01"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = replace(GridConfig(), step_penalty=args.step_penalty,
                     wall_penalty=args.wall_penalty, goal_reward=args.goal_reward)
    env = GridWorld(config=config, seed=args.seed)
    initial_q = np.zeros((env.num_states, env.num_actions))
    print("\nLESSON 01: Can a machine learn to navigate by trial and error?")
    print("\nMap: S=start, G=goal, #=wall. A passage is open at row 4.")
    print(env.render_ascii())
    print("\nObservation: one integer position ID, action: UP/RIGHT/DOWN/LEFT")
    print("Goal bonus", config.goal_reward, "wall penalty", config.wall_penalty,
          "step penalty", config.step_penalty)
    print("\nPREDICT: How will the untrained policy compare to a learned one?\n")

    starts = evaluation_starts(env, count=60, seed=2026)
    before_metrics = evaluate(initial_q, starts=starts, env=env)
    random_metrics = evaluate(None, starts=starts, env=env, random_policy=True)
    print("Training, please wait...")
    trained = train_agent(episodes=args.episodes, seed=args.seed, alpha=args.alpha,
                          gamma=args.gamma, epsilon_start=args.epsilon_start,
                          epsilon_end=args.epsilon_end, env=env)
    after_metrics = evaluate(trained.q, starts=starts, env=env)
    before = rollout(env, initial_q, start=config.start, seed=100)
    after = rollout(env, trained.q, start=config.start, seed=100)
    print("\nRepresentative BEFORE (untrained, always chooses first tied action):")
    print(env.render_ascii(before["path"]))
    print("Before outcome:", {k: v for k, v in before.items() if k != "path"})
    print("\nRepresentative AFTER (learned greedy policy):")
    print(env.render_ascii(after["path"]))
    print("After outcome:", {k: v for k, v in after.items() if k != "path"})
    print("\nEVALUATION on the SAME 60 start positions (not training returns)")
    results = [("untrained", before_metrics), ("random", random_metrics),
               ("trained", after_metrics)]
    for name, metrics in results:
        print(f"{name:11}  success={metrics['success_rate']:6.1%}  "
              f"return={metrics['mean_return']:7.2f}  "
              f"steps={metrics['mean_steps']:6.2f}  "
              f"bumps={metrics['mean_wall_bumps']:6.2f}")
    print("Last 100 training episodes: success=",
          f"{np.mean(trained.successes[-100:]):.1%}")

    with (args.output / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", *before_metrics.keys()])
        for name, metrics in results:
            writer.writerow([name, *metrics.values()])
    with (args.output / "training_history.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "epsilon", "return", "success", "steps", "wall_bumps"])
        for i in range(args.episodes):
            writer.writerow([i + 1, trained.epsilons[i], trained.episode_returns[i],
                             int(trained.successes[i]), trained.lengths[i], trained.wall_bumps[i]])
    np.save(args.output / "q_table.npy", trained.q)
    plot_training(trained, args.output / "learning_curve.png")
    plot_before_after(env, before, after, args.output / "before_after.png")
    plot_policy(env, trained.q, args.output / "learned_policy.png")
    print("\nFiles saved:", args.output.resolve())
    for name in ("learning_curve.png", "before_after.png", "learned_policy.png",
                 "metrics.csv", "training_history.csv", "q_table.npy"):
        print(" -", name)
    print("\nQUESTION: Which line of code encoded a path around the wall? None. "
          "The environment encodes the wall; experience updates Q.")


if __name__ == "__main__":
    main()
