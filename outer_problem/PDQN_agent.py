import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, continuous_action, discrete_index, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(continuous_action, dtype=np.float32),
            int(discrete_index),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, continuous_actions, discrete_indices, rewards, next_states, dones = zip(*transitions)
        return {
            "states": np.asarray(states, dtype=np.float32),
            "continuous_actions": np.asarray(continuous_actions, dtype=np.float32),
            "discrete_indices": np.asarray(discrete_indices, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.asarray(next_states, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }

    def size(self):
        return len(self.buffer)


class ParamActor(nn.Module):
    def __init__(self, state_dim, hidden_dim, continuous_action_dim):
        super(ParamActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, continuous_action_dim)
        self.activate_func = nn.ReLU()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3, gain=0.01)

    def forward(self, state):
        x = self.activate_func(self.fc1(state))
        x = self.activate_func(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class QNetwork(nn.Module):
    def __init__(self, state_dim, continuous_action_dim, hidden_dim, joint_action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + continuous_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, joint_action_dim)
        self.activate_func = nn.ReLU()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3, gain=0.01)

    def forward(self, state, continuous_action):
        x = torch.cat([state, continuous_action], dim=-1)
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        return self.fc3(x)


class PDQN:
    def __init__(self, state_dim, hidden_dim, continuous_action_dim, discrete_action_dims,
                 actor_lr, critic_lr, gamma, tau, epsilon_start, epsilon_end,
                 epsilon_decay_steps, device):
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.continuous_action_dim = int(continuous_action_dim)
        self.discrete_action_dims = np.asarray(discrete_action_dims, dtype=np.int64)
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = max(int(epsilon_decay_steps), 1)
        self.total_steps = 0

        self.action_radices = np.cumprod(
            np.concatenate(([1], self.discrete_action_dims[::-1]))
        )[:-1][::-1].astype(np.int64)
        self.joint_action_dim = int(np.prod(self.discrete_action_dims))

        self.actor = ParamActor(state_dim, hidden_dim, continuous_action_dim).to(device)
        self.actor_target = ParamActor(state_dim, hidden_dim, continuous_action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = QNetwork(state_dim, continuous_action_dim, hidden_dim, self.joint_action_dim).to(device)
        self.critic_target = QNetwork(state_dim, continuous_action_dim, hidden_dim, self.joint_action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

    def _current_epsilon(self):
        progress = min(self.total_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def decode_discrete_index(self, action_index):
        remaining = int(action_index)
        decoded = np.zeros_like(self.discrete_action_dims)
        for i, radix in enumerate(self.action_radices):
            decoded[i] = remaining // radix
            remaining = remaining % radix
        return decoded.astype(np.int64)

    def encode_discrete_action(self, discrete_action):
        discrete_action = np.asarray(discrete_action, dtype=np.int64)
        return int(np.sum(discrete_action * self.action_radices))

    def choose_action(self, state, evaluate=False):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            continuous_action = self.actor(state_tensor)
            q_values = self.critic(state_tensor, continuous_action)

        epsilon = 0.0 if evaluate else self._current_epsilon()
        if (not evaluate) and (np.random.rand() < epsilon):
            discrete_index = np.random.randint(self.joint_action_dim)
        else:
            discrete_index = int(torch.argmax(q_values, dim=-1).item())

        continuous_action_np = continuous_action.squeeze(0).cpu().numpy().astype(np.float32)
        discrete_action_np = self.decode_discrete_index(discrete_index)
        self.total_steps += 0 if evaluate else 1

        return {
            "continuous": continuous_action_np,
            "discrete": discrete_action_np,
            "discrete_index": discrete_index,
            "epsilon": epsilon,
        }

    def update(self, transition_dict, writer=None, step=None, agent_name="BS"):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32, device=self.device)
        continuous_actions = torch.tensor(
            transition_dict["continuous_actions"], dtype=torch.float32, device=self.device
        )
        discrete_indices = torch.tensor(
            transition_dict["discrete_indices"], dtype=torch.long, device=self.device
        ).view(-1, 1)
        rewards = torch.tensor(
            transition_dict["rewards"], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            transition_dict["dones"], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        with torch.no_grad():
            next_continuous_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_continuous_actions)
            next_q_max = next_q_values.max(dim=1, keepdim=True)[0]
            q_target = rewards + self.gamma * (1.0 - dones) * next_q_max

        q_values = self.critic(states, continuous_actions)
        q_pred = q_values.gather(1, discrete_indices)
        critic_loss = F.mse_loss(q_pred, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_continuous_actions = self.actor(states)
        actor_q_values = self.critic(states, actor_continuous_actions)
        actor_loss = -actor_q_values.max(dim=1, keepdim=True)[0].mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        if writer is not None and step is not None:
            writer.add_scalar(f"{agent_name}/Actor_Loss", actor_loss.item(), step)
            writer.add_scalar(f"{agent_name}/Critic_Loss", critic_loss.item(), step)
            writer.add_scalar(f"{agent_name}/Q_Target", q_target.mean().item(), step)
            writer.add_scalar(f"{agent_name}/Q_Pred", q_pred.mean().item(), step)
            writer.add_scalar(f"{agent_name}/Epsilon", self._current_epsilon(), step)

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
