import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Beta, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


# orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Gaussian(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high, device):
        super(Actor_Gaussian, self).__init__()
        self.max_action = action_high
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim,
                                       action_dim)  # 让 std 也随状态变化，或者用 Parameter 但初始化要小  # We use 'nn.Parameter' to train log_std automatically

        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

        # 计算缩放参数: action = (tanh(gaussian) + 1) / 2 * (high - low) + low
        self.scale = (self.action_high - self.action_low) / 2.0
        self.bias = (self.action_high + self.action_low) / 2.0

        self.activate_func = nn.Tanh()
        self.device = device
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)
        orthogonal_init(self.log_std_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.mean_layer(s)
        # 限制 log_std 的范围，防止过大或过小导致数值不稳定
        log_std = self.log_std_layer(s)
        log_std = torch.clamp(log_std, -5, 2)
        return mean, log_std

        # mean = (torch.tanh(self.mean_layer(s)) + 1) / 2 * (self.action_high - self.action_low) + self.action_low
        # return mean

    def get_dist(self, s):
        mean, log_std = self.forward(s)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist

        # mean = self.forward(s)
        # log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        # std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        # dist = Normal(mean, std)  # Get the Gaussian distribution
        # return dist

    def sample(self, s):
        dist = self.get_dist(s)
        x_t = dist.rsample()  # rsample 支持重参数化采样 (reparameterization trick)
        y_t = torch.tanh(x_t)  # 挤压到 [-1, 1]

        # 映射到真实物理范围
        action = y_t * self.scale + self.bias

        # 计算 Log Prob (必须包含 Jacobian 修正)
        # log_prob(y) = log_prob(x) - log(1 - tanh(x)^2)
        log_prob = dist.log_prob(x_t)
        # 这里的 sum 是对动作维度求和
        log_prob -= torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # 返回均值用于测试（确定性策略）
        mean_action = torch.tanh(dist.mean) * self.scale + self.bias

        return action, log_prob, mean_action

    def evaluate(self, s, action):
        """
        s: state
        action: 这里的 action 是 replay buffer 里存的动作（已经是物理范围内的值了）
        """
        mean, log_std = self.forward(s)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        # === 逆变换的关键步骤 ===
        # 因为我们存的是物理动作 y (在 [low, high] 之间)
        # 我们需要反推回 Tanh 之前的 x (在 [-1, 1] 之间) 才能算 log_prob

        # 1. 归一化回 [-1, 1]
        action_norm = (action - self.bias) / self.scale

        # 2. 数值稳定性处理（防止 atanh 溢出）
        # Tanh 的反函数是 atanh，如果值是 1.0 或 -1.0，atanh 是无穷大
        action_norm = torch.clamp(action_norm, -0.999999, 0.999999)

        # 3. 计算 Pre-tanh value (这一步其实不需要显式算出数值，只需要用 Jacobian 公式)
        # 但我们需要计算 log_prob

        # 这里的数学逻辑：
        # y = tanh(x) * scale + bias
        # x = atanh( (y - bias) / scale )
        # 我们需要 log_prob(y)

        # 我们可以利用 pytorch 的变换分布工具，或者手动写：
        x_t = torch.atanh(action_norm)

        # 原始高斯的 log_prob
        log_prob = dist.log_prob(x_t)

        # 减去 Jacobian 修正项
        # log_prob(y) = log_prob(x) - log(scale * (1 - tanh(x)^2))
        # 因为 x_t = atanh(action_norm)，所以 tanh(x_t) 就是 action_norm
        log_prob -= torch.log(self.scale * (1 - action_norm.pow(2)) + 1e-6)

        # 求和（多维动作）
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Entropy (近似计算，或直接用高斯的 entropy)
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, dist_entropy


class Actor_Beta(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high, device):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_layer = nn.Linear(hidden_dim, action_dim)
        self.beta_layer = nn.Linear(hidden_dim, action_dim)
        self.activate_func = nn.Tanh()  # Trick10: use tanh
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.alpha_layer, gain=0.01)
        orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activate_func = nn.Tanh()  # use tanh insted of relu
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class IPPO:
    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high,
                 actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes, device,
                 policy_dist="Beta", entropy_coef=0.01):
        self.policy_dist = policy_dist
        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(state_dim, hidden_dim, action_dim, action_low, action_high, device).to(device)
        else:
            self.actor = Actor_Gaussian(state_dim, hidden_dim, action_dim, action_low, action_high, device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.epochs = epochs  # 更新次数
        self.num_episodes = num_episodes  # 总轮次
        self.device = device
        self.entropy_coef = entropy_coef  # 熵策略项
        self.action_low = action_low
        self.action_high = action_high

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)  # shape (1, state_dim)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a).sum(dim=-1, keepdim=True)  # The log probability density of the action
                return a.squeeze(0).cpu().numpy(), a_logprob.squeeze(
                    0).cpu().numpy()  # returns (action_vec,), (1,) or (,) depending
        else:
            with torch.no_grad():
                # 使用新的 Gaussian Actor
                action, log_prob, _ = self.actor.sample(s)

                # 由于在内部已经做了缩放，这里出来的 action 就是物理动作 a_env
                # 为了兼容你的代码结构，a_raw 和 a_env 可以是一样的，或者你不再需要区分它们
                a_out = action.squeeze(0).cpu().detach().numpy()
                log_prob_out = log_prob.squeeze(0).cpu().detach().numpy()

                return a_out, a_out, log_prob_out  # 返回三次：raw(即env), env, log_prob

    def update(self, transition_dict, step=None, writer=None, agent_name="Agent"):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        old_log_probs = torch.tensor(transition_dict['old_log_probs'],
                                     dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        real_dones = torch.tensor(transition_dict['real_dones'],
                                  dtype=torch.float).view(-1, 1).to(self.device)
        # ---------------- 计算TD目标和优势 ----------------
        adv = []
        gae = 0.0
        with torch.no_grad():
            vs = self.critic(states)
            vs_ = self.critic(next_states)
            # 计算 TD-error (td_delta)
            td_target = rewards + self.gamma * vs_ * (1 - real_dones)  # real_dones 始终为 False 因为都是人为截断
            td_delta = td_target - vs

            # 为了循环，转为 numpy
            td_delta = td_delta.cpu().detach().numpy()
            dones = dones.cpu().detach().numpy()

            # 从后向前计算 GAE
            for delta, d in zip(reversed(td_delta), reversed(dones)):
                # 当一个回合结束时 (d=True)，重置 gae 的传播
                gae = delta + self.gamma * self.lmbda * gae * (1.0 - d)
                adv.insert(0, gae)
            # 将 adv 和 v_target 转为 Tensor
            adv = torch.tensor(adv, dtype=torch.float, device=self.device).view(-1, 1)
            v_target = adv + self.critic(states)

            # # --- 加入随机噪声 (效果不好就删了) ---
            # alpha = 0.05  # 噪声权重
            # noise = torch.randn_like(advantage).to(self.device)  # shape 与 advantage 一致
            # advantage = (1 - alpha) * advantage + alpha * noise

            # # ++++++++++  优势函数归一化  ++++++++++
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            # # +++++++++++++++++++++++++++++++++++

        # tensorboard 中可视化
        actor_losses = []
        critic_losses = []

        # ++++++++++ 对同一批数据进行多轮优化 ++++++++++
        for epoch in range(self.epochs):
            if self.policy_dist == "Beta":
                # ---------------- 连续动作 log_prob (使用当前策略) ----------------
                dist_now = self.actor.get_dist(states)
                dist_entropy = dist_now.entropy().sum(dim=1, keepdim=True)  # 多维动作空间求和
                log_probs = dist_now.log_prob(actions).sum(dim=1, keepdim=True)
            else:
                # Gaussian 分布直接调用 evaluate，传入 state 和 buffer 里的 action
                log_probs, dist_entropy = self.actor.evaluate(states, actions)

            # ---------------- PPO 损失 ----------------
            # 在第二轮及以后, log_probs 和 old_log_probs 将不同
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv

            # # ++++++++++ 加上策略熵 ++++++++++
            # actor_loss = torch.mean(-torch.min(surr1, surr2) - self.entropy_coef * dist_entropy)
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            critic_loss = torch.mean(F.mse_loss(self.critic(states), v_target.detach()))

            # ---------------- 梯度更新 ----------------
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            # ++++++++++ 梯度裁剪 ++++++++++
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.lr_decay(step)

        # --------- TensorBoard 记录指标 ---------
        if writer is not None and step is not None:
            writer.add_scalar(f"{agent_name}/Actor_Loss", np.mean(actor_losses), step)
            writer.add_scalar(f"{agent_name}/Critic_Loss", np.mean(critic_losses), step)
            kl = (old_log_probs - log_probs).mean().item()
            writer.add_scalar(f"{agent_name}/KL_Divergence", kl, step)


    def lr_decay(self, total_steps):
        lr_a_now = self.actor_optimizer.defaults['lr'] * (1 - total_steps / self.num_episodes)
        lr_a_now = max(lr_a_now, 1e-6)  # 防止学习率为负
        lr_c_now = self.critic_optimizer.defaults['lr'] * (1 - total_steps / self.num_episodes)
        lr_c_now = max(lr_c_now, 1e-6)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now
