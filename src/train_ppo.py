"""Smoke test: train PPO on CartPole and save a learning-curve plot."""
import os
import tempfile

import matplotlib.pyplot as plt
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from torch.utils.tensorboard import SummaryWriter

ITERS = int(os.environ.get("ITERS", "5"))
RESULTS_DIR = os.environ.get("RAY_RESULTS_DIR", "/workspace/results")

ray.init(include_dashboard=True, dashboard_host="0.0.0.0", dashboard_port=8265)

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .env_runners(num_env_runners=2)
)
algo = config.build_algo()

returns = []
log_root = os.path.join(RESULTS_DIR, "tensorboard")
os.makedirs(log_root, exist_ok=True)
log_dir = tempfile.mkdtemp(prefix="ppo_cartpole_", dir=log_root)
print(f"TensorBoard logs: {log_dir}")
with SummaryWriter(log_dir=log_dir) as writer:
    for i in range(ITERS):
        result = algo.train()
        ret = result["env_runners"]["episode_return_mean"]
        returns.append(ret)
        writer.add_scalar("env_runners/episode_return_mean", ret, i + 1)
        writer.flush()
        print(f"iter {i}: episode_return_mean={ret:.2f}")

ckpt = algo.save_to_path(os.path.join(RESULTS_DIR, "ppo_cartpole"))
print(f"checkpoint: {ckpt}")
algo.stop()

plt.plot(returns, marker="o")
plt.xlabel("training iteration")
plt.ylabel("mean episode return")
plt.title("PPO CartPole-v1")
os.makedirs("/workspace/plots", exist_ok=True)
plt.savefig("/workspace/plots/ppo_cartpole.png", dpi=120)
print("plot: /workspace/plots/ppo_cartpole.png")
