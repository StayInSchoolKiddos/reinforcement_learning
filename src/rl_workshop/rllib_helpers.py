"""RLlib 2.59 new API stack helpers; import Ray only in RLlib lessons."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def ppo_config(env_cls, env_config=None, seed=42, multi_agent=False, batch_size=512):
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    config = (
        PPOConfig()
        .environment(env=env_cls, env_config=dict(env_config or {}))
        .env_runners(num_env_runners=0)
        .learners(num_learners=0, num_gpus_per_learner=0)
        .training(
            gamma=0.99, lr=3e-4, train_batch_size_per_learner=batch_size,
            minibatch_size=128, num_epochs=3,
        )
        .rl_module(model_config=DefaultModelConfig(
            fcnet_hiddens=[64, 64], fcnet_activation="relu"
        ))
        .debugging(seed=seed)
    )
    if multi_agent:
        config = config.multi_agent(
            policies={"pursuer", "evader"},
            policy_mapping_fn=identity_mapping,
            policies_to_train=["pursuer", "evader"],
        )
    return config


def identity_mapping(agent_id, episode=None, **kwargs):
    return agent_id


def infer_action(algo, observation: np.ndarray, module_id: str = "default_policy") -> int:
    """Inference from RLModule logits, using discrete-action argmax (no exploration).

    The examples have flat Box observations and Discrete action spaces.
    More complex connectors or action distributions need a different helper.
    """
    import torch
    from ray.rllib.core.columns import Columns
    module = algo.get_module(module_id)
    with torch.no_grad():
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
        outputs = module.forward_inference({Columns.OBS: obs})
        if Columns.ACTIONS in outputs:
            return int(outputs[Columns.ACTIONS].reshape(-1)[0].item())
        return int(outputs[Columns.ACTION_DIST_INPUTS].argmax(dim=-1).item())


def checkpoint_path(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser().resolve() / "checkpoint"


def start_ray():
    import ray
    if not ray.is_initialized():
        ray.init(include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)


def stop_ray(algo=None):
    import ray
    if algo is not None:
        algo.stop()
    ray.shutdown()


def metric_mean(result: dict) -> float:
    return float(result.get("env_runners", {}).get("episode_return_mean", float("nan")))
