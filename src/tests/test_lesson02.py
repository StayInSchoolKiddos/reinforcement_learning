"""Behavior checks for the DQN code students execute in Lesson 02.

Run: python -m unittest discover -s src/tests -p test_lesson02.py
The notebook is the source of truth; no second DQN implementation is tested.
"""
from collections import deque
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, replace
import io
import json
from pathlib import Path
import random
import time
import unittest

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class Lesson02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "lessons/lesson02_dqn/Lesson_02_DQN_Gymnasium.ipynb"
        notebook = json.loads(path.read_text())
        cls.ns = dict(deque=deque, deepcopy=deepcopy, dataclass=dataclass,
                      random=random, time=time, gym=gym, np=np, torch=torch,
                      nn=nn, DEVICE=torch.device("cpu"), TRAIN_SEED=7,
                      EVAL_SEEDS=(2026, 2027))
        torch.set_num_threads(1)
        with redirect_stdout(io.StringIO()):
            for cell in notebook["cells"]:
                tags = cell.get("metadata", {}).get("tags", [])
                source = "".join(cell["source"])
                if set(tags) & {"network", "dqn", "training_definition"}:
                    exec(compile(source, str(path), "exec"), cls.ns)
                if "evaluation" in tags:
                    exec(source.split("\nmetrics = ")[0], cls.ns)

    def test_dimensions_replay_and_exploration(self):
        network = self.ns["QNetwork"]()
        self.assertEqual(tuple(network(torch.zeros(8, 4)).shape), (8, 2))
        replay = self.ns["ReplayBuffer"](3)
        obs = np.zeros(4, dtype=np.float32)
        for action in (0, 1, 0, 1):
            replay.add(obs, action, 1.0, obs, False)
        obs[:] = 99  # Replay owns copies, even if a caller mutates the observation.
        batch = replay.sample(3, np.random.default_rng(1))
        self.assertEqual(len(replay), 3)
        self.assertEqual([tuple(t.shape) for t in batch], [(3, 4), (3,), (3,), (3, 4), (3,)])
        self.assertEqual(batch[1].dtype, torch.int64)
        self.assertTrue(torch.equal(batch[0], torch.zeros(3, 4)))
        config = self.ns["DQNConfig"]()
        self.assertAlmostEqual(self.ns["epsilon_at"](0, config), 1.0)
        self.assertAlmostEqual(self.ns["epsilon_at"](config.total_steps, config), 0.05)

    def test_terminal_and_truncated_bootstrapping(self):
        target = nn.Linear(4, 2)
        with torch.no_grad():
            target.weight.zero_()
            target.bias.copy_(torch.tensor([2.0, 4.0]))
        # Ordinary, true terminal, external time limit: only the middle is masked.
        values = self.ns["bellman_targets"](
            target, torch.ones(3), torch.zeros(3, 4), torch.tensor([0., 1., 0.]), .99)
        torch.testing.assert_close(values, torch.tensor([4.96, 1., 4.96]))
        self.assertFalse(values.requires_grad)

    def test_online_update_and_target_synchronization(self):
        online = self.ns["QNetwork"]()
        target = deepcopy(online).requires_grad_(False)
        old_online = [p.clone() for p in online.parameters()]
        old_target = [p.clone() for p in target.parameters()]
        batch = (torch.zeros(8, 4), torch.zeros(8, dtype=torch.int64),
                 torch.ones(8), torch.zeros(8, 4), torch.ones(8))
        optimizer = torch.optim.Adam(online.parameters(), lr=.001)
        loss = self.ns["optimize_batch"](online, target, optimizer, batch, .99)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(p, old) for p, old in zip(online.parameters(), old_online)))
        self.assertTrue(all(torch.equal(p, old) for p, old in zip(target.parameters(), old_target)))
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        self.ns["soft_update"](target, online, .25)
        for p, old, new in zip(target.parameters(), old_target, online.parameters()):
            torch.testing.assert_close(p, old + .25 * (new - old))
        self.ns["soft_update"](target, online, 1.)
        for p, new in zip(target.parameters(), online.parameters()):
            torch.testing.assert_close(p, new)

    def test_short_training_and_frozen_repeatable_evaluation(self):
        config = replace(self.ns["DQNConfig"](), total_steps=128,
                         warmup_steps=32, batch_size=32)
        result = self.ns["train_dqn"](config, seed=7, verbose=False)
        self.assertEqual(len(result["losses"]), 128)
        self.assertTrue(np.isfinite(result["losses"][31:]).all())
        self.assertTrue(any(not torch.equal(p, initial) for p, initial in
                            zip(result["model"].parameters(), result["initial"].parameters())))
        model = result["model"]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
        first = self.ns["evaluate"](model)
        second = self.ns["evaluate"](model)
        self.assertEqual(first, second)
        self.assertEqual(first["episodes"], 2)
        self.assertEqual(first["mean_return"], first["mean_length"])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        for param, grad in zip(model.parameters(), gradients):
            if grad is None:
                self.assertIsNone(param.grad)
            else:
                self.assertTrue(torch.equal(param.grad, grad))


if __name__ == "__main__":
    unittest.main()
