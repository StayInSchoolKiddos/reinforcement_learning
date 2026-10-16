"""Static plots written under outputs/; no display server needed."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_navigation(env, path: str | Path, title: str = "Navigation rollout") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    pts = np.asarray(env.path)
    ax.plot(pts[:, 0], pts[:, 1], "-o", linewidth=1.3, markersize=2, label="trajectory")
    ax.scatter(*pts[0], marker="s", s=85, label="start")
    ax.scatter(*env.goal, marker="*", s=150, label="goal")
    if env.obstacles_enabled:
        for center in env.centers:
            ax.add_patch(plt.Circle(center, env.obstacle_radius, fill=False, linewidth=2))
    ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="East (a.u.)", ylabel="North (a.u.)", title=title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_metrics_csv(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_training(csv_path: str | Path, path: str | Path) -> None:
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    x = [int(row["iteration"]) for row in records]
    y = [float(row["episode_return_mean"]) for row in records]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o")
    ax.set(xlabel="PPO training iteration", ylabel="Mean episode return", title="RLlib training curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
