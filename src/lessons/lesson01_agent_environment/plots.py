"""Static lesson visuals, headless so Docker needs no display server."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .gridworld import ACTION_SYMBOLS, GridWorld


def rolling(values, window=50):
    values = np.asarray(values, dtype=float)
    c = np.cumsum(np.insert(values, 0, 0))
    means = (c[window:] - c[:-window]) / window
    return np.concatenate((np.full(window - 1, np.nan), means))


def plot_training(result, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rolling(result.successes, 50), label="Success rate, rolling 50")
    ax.plot(result.epsilons, alpha=0.75, label="Exploration ε")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Rate / probability")
    ax.set_title("Q-learning: does success improve with experience?")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=155)
    plt.close(fig)


def draw_grid(ax, env: GridWorld, path=None, *, title=""):
    cfg = env.config
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            if (r, c) in env.walls:
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="0.25"))
    if path:
        xy = np.asarray([(c, r) for r, c in path], dtype=float)
        ax.plot(xy[:, 0], xy[:, 1], linewidth=2, alpha=.75, marker=".", markersize=3)
    sr, sc = cfg.start
    gr, gc = cfg.goal
    ax.plot(sc, sr, "o", markersize=11, label="fixed demo start")
    ax.plot(gc, gr, "*", markersize=15, label="goal")
    ax.set_xlim(-.5, cfg.cols - .5)
    ax.set_ylim(cfg.rows - .5, -.5)
    ax.set_xticks(range(cfg.cols)); ax.set_yticks(range(cfg.rows))
    ax.set_aspect("equal")
    ax.grid(alpha=.25)
    ax.set_title(title)


def plot_before_after(env: GridWorld, before: dict, after: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    draw_grid(axes[0], env, before["path"], title=f"Before: {'reached goal' if before['success'] else 'failed'} ({before['steps']} steps)")
    draw_grid(axes[1], env, after["path"], title=f"After: {'reached goal' if after['success'] else 'failed'} ({after['steps']} steps)")
    fig.suptitle("Same start and map, different Q-tables")
    fig.tight_layout()
    fig.savefig(path, dpi=155)
    plt.close(fig)


def plot_policy(env: GridWorld, q: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_grid(ax, env, title="Learned greedy action by state")
    for r in range(env.config.rows):
        for c in range(env.config.cols):
            if (r, c) in env.walls or (r, c) == env.config.goal:
                continue
            a = int(np.argmax(q[env.encode((r, c))]))
            ax.text(c, r, ACTION_SYMBOLS[a], fontsize=20, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=155)
    plt.close(fig)
