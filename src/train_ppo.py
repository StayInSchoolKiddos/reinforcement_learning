"""Smoke test: train PPO on CartPole and save a learning-curve plot."""
import os
import tempfile

import matplotlib.pyplot as plt
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from torch.utils.tensorboard import SummaryWriter

ITERS = int(os.environ.get("ITERS", "5"))
RESULTS_DIR = os.environ.get("RAY_RESULTS_DIR", "/workspace/results")
# Sampling is the bottleneck for CartPole; use most of the cores for env runners.
NUM_ENV_RUNNERS = int(os.environ.get("NUM_ENV_RUNNERS", str(max(1, (os.cpu_count() or 2) - 2))))
# Only ask Ray for a GPU if torch can actually see one, otherwise the Learner actor
# sits in PENDING_CREATION forever and training appears to hang.
USE_GPU = os.environ.get("USE_GPU", "auto")
if USE_GPU == "auto":
    USE_GPU = torch.cuda.is_available()
else:
    USE_GPU = USE_GPU.lower() in ("1", "true", "yes")
print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | "
      f"use_gpu={USE_GPU} | env_runners={NUM_ENV_RUNNERS}")

ray.init(include_dashboard=True, dashboard_host="0.0.0.0", dashboard_port=8265)

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .env_runners(num_env_runners=NUM_ENV_RUNNERS)
    # New API stack: GPU placement is configured on the Learner, not via resources().
    .learners(
        num_learners=1 if USE_GPU else 0,
        num_gpus_per_learner=1 if USE_GPU else 0,
    )
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
