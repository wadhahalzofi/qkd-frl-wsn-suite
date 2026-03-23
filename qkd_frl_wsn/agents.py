from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .config import RLConfig
from .models import DQN, to_numpy_state


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        items = list(self.buffer)
        return [items[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, rl_cfg: RLConfig, device: str = "cpu") -> None:
        self.cfg = rl_cfg
        self.device = torch.device(device)
        self.policy = DQN(rl_cfg.state_dim, rl_cfg.action_dim, rl_cfg.hidden_units).to(self.device)
        self.target = DQN(rl_cfg.state_dim, rl_cfg.action_dim, rl_cfg.hidden_units).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=rl_cfg.learning_rate,
            weight_decay=rl_cfg.weight_decay,
        )
        self.buffer = ReplayBuffer(rl_cfg.replay_buffer_size)
        self.train_steps = 0
        self.epsilon = rl_cfg.epsilon_start
        self.reward_ema = 0.0
        self.participation_history: Deque[int] = deque(maxlen=20)
        self.last_loss = 0.0

    def act(self, state: np.ndarray, action_mask: np.ndarray | None = None, greedy: bool = False) -> int:
        state = to_numpy_state(state)
        eps = 0.0 if greedy else self.epsilon
        if np.random.rand() < eps:
            if action_mask is None:
                return int(np.random.randint(self.cfg.action_dim))
            valid = np.flatnonzero(action_mask > 0.0)
            if len(valid) == 0:
                return int(np.random.randint(self.cfg.action_dim))
            return int(np.random.choice(valid))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy(s).detach().cpu().numpy()[0]
            if action_mask is not None:
                masked = np.where(action_mask > 0.0, q, -1e9)
                return int(np.argmax(masked))
            return int(np.argmax(q))

    def remember(self, state, action, reward, next_state, done) -> None:
        self.reward_ema = 0.95 * self.reward_ema + 0.05 * float(reward)
        self.buffer.push(to_numpy_state(state), int(action), float(reward), to_numpy_state(next_state), float(done))

    def _update_epsilon(self) -> None:
        if self.train_steps >= self.cfg.epsilon_decay_steps:
            self.epsilon = self.cfg.epsilon_end
            return
        span = self.cfg.epsilon_start - self.cfg.epsilon_end
        frac = self.train_steps / max(1, self.cfg.epsilon_decay_steps)
        self.epsilon = self.cfg.epsilon_start - span * frac

    def train_step(self) -> float:
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0
        batch = self.buffer.sample(self.cfg.batch_size)
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.gradient_clip_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_period == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self._update_epsilon()
        self.last_loss = float(loss.item())
        return self.last_loss

    def local_train(self, n_steps: int) -> Dict[str, float]:
        losses = []
        for _ in range(n_steps):
            losses.append(self.train_step())
        return {
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "last_loss": float(losses[-1]) if losses else self.last_loss,
        }

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}

    def set_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.policy.load_state_dict(state_dict)
        self.target.load_state_dict(state_dict)

    def get_delta(self, global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        local = self.get_state_dict()
        return {k: local[k] - global_state[k] for k in global_state.keys()}

    def apply_global(self, global_state: Dict[str, torch.Tensor]) -> None:
        self.set_state_dict(global_state)
