import os
import random
import pandas as pd
import torch
import numpy as np
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from my_env import MyEnv
import warnings
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.normalization import Normalization, RewardScaling
from IPPO_agent import IPPO, orthogonal_init
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# ==========================================
# Task 2: Hybrid Action Space Support
# ==========================================

class HybridDistribution:
    """
    Custom Distribution for Hybrid Action Space (Continuous + Discrete)
    """
    def __init__(self, alpha, beta, discrete_logits):
        self.cont_dist = Beta(alpha, beta)
        self.disc_dist = Categorical(logits=discrete_logits)

    def sample(self):
        # Sample from both distributions
        a_cont = self.cont_dist.sample()
        a_disc = self.disc_dist.sample()
        
        # a_disc is (batch,), unsqueeze to (batch, 1) for concatenation
        a_disc = a_disc.unsqueeze(-1).float()
        
        # Concatenate: [continuous_actions, discrete_action]
        return torch.cat([a_cont, a_disc], dim=-1)

    def log_prob(self, action):
        """
        Calculate log_prob for hybrid action.
        Args:
            action: (batch, cont_dim + 1) tensor
        Returns:
            log_prob: (batch, 2) tensor where [:, 0] is cont_log_prob, [:, 1] is disc_log_prob
        """
        # Split action into continuous and discrete parts
        a_cont = action[:, :-1]
        a_disc = action[:, -1].long()
        
        # Calculate log_probs
        # Sum continuous log_prob over dimensions (assuming independent)
        log_prob_cont = self.cont_dist.log_prob(a_cont).sum(dim=-1, keepdim=True)
        
        # Discrete log_prob
        log_prob_disc = self.disc_dist.log_prob(a_disc).unsqueeze(-1)
        
        # Return concatenated log_probs to be summed by IPPO.update (which uses sum(dim=1))
        return torch.cat([log_prob_cont, log_prob_disc], dim=-1)

    def entropy(self):
        # Sum of entropies
        ent_cont = self.cont_dist.entropy().sum(dim=-1, keepdim=True)
        ent_disc = self.disc_dist.entropy().unsqueeze(-1)
        return ent_cont + ent_disc


class Actor_Hybrid(nn.Module):
    """
    Actor Network for Hybrid Action Space
    """
    def __init__(self, state_dim, hidden_dim, action_dim_cont, action_dim_disc, device):
        super(Actor_Hybrid, self).__init__()
        # Shared Feature Extractor
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Continuous Action Head (Beta Distribution)
        self.alpha_layer = nn.Linear(hidden_dim, action_dim_cont)
        self.beta_layer = nn.Linear(hidden_dim, action_dim_cont)
        
        # Discrete Action Head (Categorical Distribution)
        self.discrete_layer = nn.Linear(hidden_dim, action_dim_disc)
        
        self.activate_func = nn.Tanh()
        self.device = device
        
        # Initialization
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.alpha_layer, gain=0.01)
        orthogonal_init(self.beta_layer, gain=0.01)
        orthogonal_init(self.discrete_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        
        # Continuous parameters (Beta: alpha > 1, beta > 1)
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        
        # Discrete parameters (Logits)
        discrete_logits = self.discrete_layer(s)
        
        return alpha, beta, discrete_logits

    def get_dist(self, s):
        alpha, beta, discrete_logits = self.forward(s)
        return HybridDistribution(alpha, beta, discrete_logits)


class IPPO_Hybrid(IPPO):
    """
    IPPO Agent adapted for Hybrid Action Space
    """
    def __init__(self, state_dim, hidden_dim, action_dim_cont, action_dim_disc, 
                 actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes, device):
        # Initialize parent to set basic params, but we will overwrite actor
        # Passing dummy action dims to parent as we handle them manually
        super().__init__(state_dim, hidden_dim, action_dim_cont, np.zeros(1), np.zeros(1),
                         actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes, device, policy_dist="Beta")
        
        # Overwrite Actor with Hybrid Actor
        self.actor = Actor_Hybrid(state_dim, hidden_dim, action_dim_cont, action_dim_disc, device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        
        # Critic remains the same (State Value Function)
        
    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()
            # log_prob from HybridDistribution is (batch, 2), sum to get total log_prob
            a_logprob = dist.log_prob(a).sum(dim=-1, keepdim=True)
            
            return a.squeeze(0).cpu().numpy(), a_logprob.squeeze(0).cpu().numpy()


def setSeed(seed):
    """
    设置随机种子，确保仿真可复现
    """
    seed = 42
    # Python 内置随机
    random.seed(seed)
    # NumPy 随机
    np.random.seed(seed)
    # PyTorch 随机
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # 确保 PyTorch 的卷积操作确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tensorboard_writer(log_dir):
    """
    创建 TensorBoard 写入器
    """
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def get_madrl_args():
    madrl_parser = argparse.ArgumentParser(description="MADRL 超参数")
    madrl_parser.add_argument("--actor_lr", type=int, default=3e-4, help="actor 学习率")
    madrl_parser.add_argument("--critic_lr", type=int, default=3e-4, help="critic 学习率")
    madrl_parser.add_argument("--lmbda", type=float, default=0.95, help="GAE (lambda)")
    madrl_parser.add_argument("--eps", type=float, default=0.2, help="PPO 裁剪因子 (epsilon)")
    madrl_parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子 (gamma)")
    madrl_parser.add_argument("--epochs", type=int, default=10, help="每次更新迭代次数")


def get_base_args():
    base_parser = argparse.ArgumentParser(description="场景的基本参数")

    # 仿真场景参数
    base_parser.add_argument("--num_cases", type=int, default=30, help="随机案例数量")
    base_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    base_parser.add_argument("--targets_num", type=int, default=8, help="目标数量")
    base_parser.add_argument("--uavs_num", type=int, default=8, help="UAV 数量")
    base_parser.add_argument("--cus_num", type=int, default=10, help="CU 数量")
    base_parser.add_argument("--uav_height", type=float, default=100, help="UAV 高度 (m)")
    base_parser.add_argument("--radius", type=float, default=200, help="区域半径 (m)")

    # 信道参数
    base_parser.add_argument("--ref_path_loss_db", type=float, default=-30, help="1m 参考路径损耗 (dB)")
    base_parser.add_argument("--frac_d_lambda", type=float, default=0.5, help="天线间距与波长比例")
    base_parser.add_argument("--alpha_uav_link", type=float, default=2, help="UAV 链路路径损耗指数")
    base_parser.add_argument("--alpha_cu_link", type=float, default=2.5, help="CU 链路路径损耗指数")
    base_parser.add_argument("--rician_factor_db", type=float, default=10, help="Rician 因子 (dB)")
    base_parser.add_argument("--antenna_nums", type=int, default=8, help="UAV 天线数量")
    base_parser.add_argument("--radar_rcs", type=float, default=10, help="雷达 RCS (m^2)")
    base_parser.add_argument("--noise_power_density_dbm", type=float, default=-174, help="噪声功率谱密度 (dBm/Hz)")
    base_parser.add_argument("--bandwidth", type=float, default=10e6, help="带宽 (Hz)")

    # 算法/物理参数
    base_parser.add_argument("--uav_c1", type=float, default=0.00614, help="UAV 飞行参数 c1")
    base_parser.add_argument("--uav_c2", type=float, default=15.976, help="UAV 飞行参数 c2")
    base_parser.add_argument("--kappa", type=float, default=1e-28, help="BS CPU 有效开关电容")
    base_parser.add_argument("--bs_max_freq", type=float, default=20e9, help="BS 最大工作频率 (Hz)")
    base_parser.add_argument("--bs_cycles_per_bit", type=float, default=1000, help="BS 处理 1bit 需要的周期数")
    base_parser.add_argument("--time_slot_duration", type=float, default=0.5, help="时隙长度 (s)")
    base_parser.add_argument("--uav_sen_duration", type=float, default=0.1, help="UAV 感知时长 (s)")
    base_parser.add_argument("--cu_max_power_dbm", type=float, default=23, help="CU 最大发射功率 (dBm)")
    base_parser.add_argument("--uav_max_power", type=float, default=10, help="UAV 最大功率 (W) = 40dBm")
    base_parser.add_argument("--cu_max_delay", type=float, default=0.5, help="娱乐任务最大延迟 (s)")
    base_parser.add_argument("--uav_max_delay", type=float, default=0.2, help="感知任务最大延迟 (s)")
    base_parser.add_argument("--uav_max_speed", type=float, default=10.0, help="UAV 最大移动速度 (m/s)")
    base_parser.add_argument("--uav_min_speed", type=float, default=5.0, help="UAV 最小移动速度 (m/s)")
    base_parser.add_argument("--uav_safe_distance", type=float, default=5.0, help="UAV 安全距离 (m)")
    base_parser.add_argument("--sen_sinr", type=float, default=2, help="感知门限阈值 (dB)")

    # 权重参数
    base_parser.add_argument("--omega_weight_1", type=float, default=0.2, help="BS 权重")
    base_parser.add_argument("--omega_weight_2", type=float, default=0.4, help="UAV 权重")
    base_parser.add_argument("--omega_weight_3", type=float, default=0.4, help="CU 权重")

    # 雷达参数
    base_parser.add_argument("--radar_duty_ratio", type=float, default=0.01, help="雷达占空比")
    base_parser.add_argument("--var_range_fluctuation", type=float, default=1e-14, help="范围波动过程方差")
    base_parser.add_argument("--radar_impulse_duration", type=float, default=2e-5, help="雷达脉冲持续时间")
    base_parser.add_argument("--radar_spectrum_shape", type=float, default=pi / sqrt(3), help="雷达频谱形状参数")

    # 基于惩罚的 CCCP 算法参数
    base_parser.add_argument("--max_iterations", type=int, default=30, help="CCCP 算法最大迭代次数")
    base_parser.add_argument("--cccp_threshold", type=float, default=1e-5, help="CCCP 算法收敛阈值")
    base_parser.add_argument("--penalty_factor", type=float, default=1e-1, help="罚因子")
    base_parser.add_argument("--zoom_factor", type=float, default=1.5, help="缩放系数")
    base_parser.add_argument("--constraint_include_groups", type=str, default="4.5,4.12,4.23,4.25,4.27,4.28,4.29,4.32, 4.39,4.40, 4.44, 4.45, auxiliary_t", help="启用约束组，逗号分隔")
    base_parser.add_argument("--constraint_exclude_groups", type=str, default="", help="禁用约束组，逗号分隔")

    return base_parser.parse_args()
    

if __name__ == "__main__":
    setSeed(seed = 42)  # 设置全局唯一种子 保证仿真可复现

    writer = get_tensorboard_writer(log_dir = f"runs/beta-ippo/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_beta-ippo_result")  # 利用 tensorboard 可视化训练结果
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")  # 获取 device

    base_args = get_base_args()  # 获取系统环境参数
    madrl_args = get_madrl_args()  # 获取 MADRL 参数

    # 创建环境
    env = MyEnv(base_args = base_args, madrl_args = madrl_args)  # 创建环境
    uavs_pos, cus_pos, targets_pos \
        = env.generate_pos(uavs_num = base_args.uavs_num,  # UAVs 数量和 targets 数量设置相同
                       cus_num = base_args.cus_num,  # CUs 的数量多于 UAVs
                       targets_num = base_args.targets_num,
                       center = (0, 0),
                       radius = base_args.radius,
                       uav_height = base_args.uav_height
                       )  # 根据圆心和半径生成 UAVs 和 CUs 位置
    
    # uavs_2_cus_channel: UAVs -> CUs 信道 (I * J * N)
    # uavs_2_bs_channel: UAVs -> BS 信道 (I * 1 * N)
    # cus_2_bs_channel: CUs -> BS 信道 (J * 1)
    uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels \
        = env.compute_com_channel_gain(uavs_pos = uavs_pos,
                                   cus_pos = cus_pos,
                                   ref_path_loss = db_2_watt(base_args.ref_path_loss_db),  # 1m 下参考路径损耗
                                   frac_d_lambda = base_args.frac_d_lambda,  # 天线间距为半波长
                                   alpha_uav_link = base_args.alpha_uav_link,  # 与 UAV 有关链路的路径损耗系数
                                   alpha_cu_link = base_args.alpha_cu_link,  # 与 CU 有关的路径损耗系数
                                   rician_factor = db_2_watt(base_args.rician_factor_db),  # Rician 因子
                                   antenna_nums = base_args.antenna_nums  # UAV 天线数量
                                   )  # 计算通信信道增益
    
    # uavs_2_targets_channels: UAVs -> targets 信道响应矩阵 (I * I * N * N)
    # I 个 UAV 对 I 个 TARGET 的信道 I * I * (a * a^{H}) = I * I * N * N
    uavs_2_targets_channels \
        = env.compute_sen_channel_gain(radar_rcs = base_args.radar_rcs,  # 目标 RCS
                                   frac_d_lambda = base_args.frac_d_lambda,  # 天线间距为半波长
                                   uavs_pos = uavs_pos,
                                   targets_pos = targets_pos,
                                   antenna_nums = base_args.antenna_nums,  # UAV 天线数量
                                   ref_path_loss = db_2_watt(base_args.ref_path_loss_db)  # 1m 下参考路径损耗
                                   )  # 计算感知信道响应矩阵

    # ---------------- 仿真时隙 ----------------
    total_time_slots = 50

    state_dim_uav = env.observation_space["uav"]["uav_0"].shape[0]
    state_dim_ris = env.observation_space["ris"].shape[0]

    # --- Task 2: Hybrid Action Space Adaptation ---
    uav_action_space = env.action_space["uav"]["uav_0"]
    is_hybrid_uav = False
    
    if isinstance(uav_action_space, gym.spaces.Dict):
        # Hybrid: Continuous + Discrete
        # Assuming structure: Dict(continuous=Box(N), discrete=Discrete(M))
        action_dim_uav_cont = uav_action_space["continuous"].shape[0]
        action_dim_uav_disc = uav_action_space["discrete"].n
        is_hybrid_uav = True
        print(f"Detected Hybrid Action Space for UAV: Cont={action_dim_uav_cont}, Disc={action_dim_uav_disc}")
        # Placeholder for action bounds (handled by actor internally or env)
        action_uav_low = np.zeros(action_dim_uav_cont)
        action_uav_high = np.ones(action_dim_uav_cont)
    else:
        # Standard Continuous
        action_dim_uav = uav_action_space.shape[0]
        action_dim_uav_cont = action_dim_uav
        action_dim_uav_disc = 0 # No discrete action
        action_uav_low = uav_action_space.low
        action_uav_high = uav_action_space.high

    action_dim_ris = env.action_space["ris"].shape[0]
    action_ris_low = env.action_space["ris"].low
    action_ris_high = env.action_space["ris"].high

    # UAV 和 RIS Agent
    agents_uav = {}
    running_norms_uav = {}
    for i in range(uav_num):
        agent_id = f"uav_{i}"
        if is_hybrid_uav:
             agents_uav[agent_id] = IPPO_Hybrid(state_dim_uav, hidden_dim, action_dim_uav_cont, action_dim_uav_disc,
                                        actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes, device)
        else:
             agents_uav[agent_id] = IPPO(state_dim_uav, hidden_dim, action_dim_uav_cont, action_uav_low, action_uav_high,
                                    actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes, device)
        running_norms_uav[agent_id] = Normalization(state_dim_uav)  # 为每个 agent 各建一个动态归一化状态空间

    agent_ris = IPPO(state_dim_ris, hidden_dim, action_dim_ris, action_ris_low, action_ris_high,
                     actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes, device)
    running_norm_ris = Normalization(state_dim_ris)  # 为每个 agent 各建一个动态归一化状态空间

    reward_scalers_uav = [RewardScaling(shape=1, gamma=gamma) for _ in range(uav_num)]
    reward_scaler_ris = RewardScaling(shape=1, gamma=gamma)

    reward_res = []
    all_agents_rewards = []  # 每个agent的奖励记录：[[uav_0, uav_1, ..., ris], ...]
    max_avg_reward = -np.inf  # 用于记录最大的平均奖励
    best_uav_trajectory = None
    user_trajectory = None

    with tqdm(total=int(num_episodes), desc='Training Progress') as pbar:
        for i_episode in range(int(num_episodes)):
            # --- 每个episode存储各UAV奖励 ---
            episode_rewards_total = []  # 总奖励（环境返回的 total_reward）
            episode_rewards_uav = np.zeros(uav_num)  # 每个UAV奖励累计
            episode_reward_ris = 0.0  # RIS 奖励累计

            # --- 为每个 UAV agent 创建独立的 transition_dict ---
            transition_dicts_uav = {
                f"uav_{j}": {'states': [],
                             'actions': [],
                             'next_states': [],
                             'rewards': [],
                             'old_log_probs': [],
                             'dones': [],
                             'real_dones': []}
                for j in range(uav_num)}
            # --- ADDED END ---
            transition_dict_ris = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'old_log_probs': [],
                'dones': [],
                'real_dones': []}

            s = env.reset()
            terminal = False
            uav_positions_episode = []  # 初始化当前 episode 的 UAV 轨迹存储
            user_positions_episode = []

            # reset reward scaling
            for rs in reward_scalers_uav:
                rs.reset()
            reward_scaler_ris.reset()

            while not terminal:
                # --- 为每个 UAV agent 独立处理状态、动作 ---
                actions_uav_dict = {}
                old_log_probs_uav_dict = {}
                states_uav_norm_dict = {}  # 临时存储归一化后的状态
                uav_positions_episode.append(env.getPosUAV())  # --- 存储当前时隙 UAV 坐标 ---
                user_positions_episode.append(env.getPosUser())  # --- 存储当前时隙用户坐标 ---

                # --- UAV agents 选择动作 ---
                for uav_i in range(uav_num):
                    agent_id = f"uav_{uav_i}"
                    # 更新并归一化该 UAV 的状态
                    s_uav_norm = running_norms_uav[agent_id](np.array(s["uav"][agent_id]))
                    states_uav_norm_dict[agent_id] = s_uav_norm

                    # 该 UAV agent 采取动作
                    a_uav, old_log_probs_uav = agents_uav[agent_id].choose_action(s_uav_norm)
                    # 存储 UAV
                    actions_uav_dict[agent_id] = a_uav
                    old_log_probs_uav_dict[agent_id] = old_log_probs_uav

                # --- RIS agent 选择动作 ---
                s_ris_norm = running_norm_ris(np.array(s["ris"]))
                a_ris, old_log_probs_ris = agent_ris.choose_action(s_ris_norm)
                # --- 环境执行 ---
                next_s, total_reward, r_dict, done = env.step({"uav": actions_uav_dict, "ris": a_ris}, i_episode)
                # =====================================================
                #  r 是 dict： r["uav"], r["ris"]
                # =====================================================
                # 解析奖励
                r_uav_list = [np.mean(r_val) if len(r_val) > 0 else 0.0 for r_val in r_dict["uav"]]
                r_ris = np.mean(r_dict["ris"]) if len(r_dict["ris"]) > 0 else 0.0

                # 奖励归一化
                r_uav_norm = [reward_scalers_uav[j](r_uav_list[j]) for j in range(uav_num)]
                r_ris_norm = reward_scaler_ris(r_ris)

                # 累计各agent奖励
                episode_rewards_total.append(total_reward)
                episode_rewards_uav += np.array(r_uav_list)
                episode_reward_ris += r_ris

                # --- 为每个 UAV agent 存储经验 ---
                for uav_i in range(uav_num):
                    agent_id = f"uav_{uav_i}"
                    # 对 next_state 同样进行 normalize
                    next_s_uav_norm = running_norms_uav[agent_id](next_s["uav"][agent_id])

                    # 存储已归一化的数据到对应的字典
                    transition_dicts_uav[agent_id]['states'].append(states_uav_norm_dict[agent_id])
                    transition_dicts_uav[agent_id]['actions'].append(actions_uav_dict[agent_id])
                    transition_dicts_uav[agent_id]['next_states'].append(next_s_uav_norm)
                    transition_dicts_uav[agent_id]['rewards'].append(r_uav_norm[uav_i])  # 共享奖励
                    transition_dicts_uav[agent_id]['old_log_probs'].append(old_log_probs_uav_dict[agent_id])
                    transition_dicts_uav[agent_id]['dones'].append(bool(done))
                    transition_dicts_uav[agent_id]['real_dones'].append(False)

                # --- 为 RIS agent 存储经验 ---
                next_s_ris_norm = running_norm_ris(next_s["ris"])
                # 存储已归一化的数据
                transition_dict_ris['states'].append(s_ris_norm)
                transition_dict_ris['actions'].append(a_ris)
                transition_dict_ris['next_states'].append(next_s_ris_norm)
                transition_dict_ris['rewards'].append(r_ris_norm)
                transition_dict_ris['old_log_probs'].append(old_log_probs_ris)
                transition_dict_ris['dones'].append(bool(done))
                transition_dict_ris['real_dones'].append(False)

                s = next_s
                terminal = done

            if np.mean(episode_rewards_total) > max_avg_reward:
                max_avg_reward = np.mean(episode_rewards_total)
                best_uav_trajectory = np.array(uav_positions_episode)  # shape: (50, uav_num, 3)
                user_trajectory = np.array(user_positions_episode)

            # ---  更新所有 UAV Agents ---
            for uav_i in range(uav_num):
                agent_id = f"uav_{uav_i}"
                agents_uav[agent_id].update(transition_dicts_uav[agent_id], i_episode, writer,
                                            agent_name=f"UAV_{uav_i}")
            # RIS Agent 更新
            agent_ris.update(transition_dict_ris, i_episode, writer, agent_name="RIS")

            # 统计奖励
            avg_total_reward = np.mean(episode_rewards_total)
            avg_uav_rewards = episode_rewards_uav / len(episode_rewards_total)
            avg_ris_reward = episode_reward_ris / len(episode_rewards_total)

            reward_res.append(np.mean(episode_rewards_total))  # 统计平均奖励
            all_agents_rewards.append(np.concatenate([avg_uav_rewards, [avg_ris_reward]]))

            # Average Reward 写入 TensorBoard
            writer.add_scalar("Reward/episode", avg_total_reward, i_episode)
            for uav_i in range(uav_num):
                writer.add_scalar(f"Reward/UAV_{uav_i}", avg_uav_rewards[uav_i], i_episode)
            writer.add_scalar("Reward/RIS", avg_ris_reward, i_episode)

            # --- 每轮 episode 结束后，立即更新 tqdm 显示平均奖励 ---
            pbar.set_postfix({
                'avg_reward': f'{np.mean(episode_rewards_total):.3f}'
            })
            pbar.update(1)
    # 关闭 TensorBoard writer
    writer.close()

    # ========= 保存总平均奖励 ========= #
    reward_array = np.array(reward_res)
    episodes_list = np.arange(reward_array.shape[0])
    plt.plot(episodes_list, reward_array)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('IPPO training performance')
    plt.show()

    # =========== 保存 reward csv 文件 ===========
    # 构造文件名
    filename = f'{today_str}_IPPO_training_rewards_{noma_group_num}_seed_{seed}.csv'
    # 保存 reward_array 和 episodes_list 到 CSV
    df = pd.DataFrame({
        'episode': episodes_list,
        'reward': reward_array
    })
    df.to_csv(filename, index=False)
    print(f"训练奖励已保存到 {filename} 中.csv")
    # =========== 保存 reward csv 文件 ===========

    # ========= 保存每个Agent奖励 ========= #
    all_agents_rewards = np.array(all_agents_rewards)
    columns = [f"UAV_{i}_reward" for i in range(uav_num)] + ["RIS_reward"]
    df_agents = pd.DataFrame(all_agents_rewards, columns=columns)
    df_agents.insert(0, "episode", np.arange(num_episodes))
    filename_agents = f'{today_str}_IPPO_all_agents_rewards_seed_{seed}.csv'
    df_agents.to_csv(filename_agents, index=False)
    print(f"✅ 所有 Agent 的奖励已保存到 {filename_agents}")

    # =================== 保存最佳 UAV 轨迹到 CSV ===================
    if best_uav_trajectory is not None:
        # ===== 保存 UAV 轨迹 =====
        uav_traj_list = []
        for t in range(best_uav_trajectory.shape[0]):
            for uav_i in range(best_uav_trajectory.shape[1]):
                x, y, z = best_uav_trajectory[t, uav_i]
                uav_traj_list.append([t, uav_i, x, y, z])
        df_uav = pd.DataFrame(uav_traj_list, columns=['time_slot', 'uav_id', 'x', 'y', 'z'])
        df_uav.to_csv(f'{today_str}_SEED{seed}_best_uav_trajectory.csv', index=False)
        print(f"✅ 最佳 UAV 轨迹已保存：{today_str}_best_uav_trajectory.csv")

        # ===== 保存 User 轨迹 =====
        user_traj_list = []
        for t in range(user_trajectory.shape[0]):
            for user_i in range(user_trajectory.shape[1]):
                x, y, z = user_trajectory[t, user_i]
                user_traj_list.append([t, user_i, x, y, z])
        df_user = pd.DataFrame(user_traj_list, columns=['time_slot', 'user_id', 'x', 'y', 'z'])
        df_user.to_csv(f'SEED{seed}_user_trajectory.csv', index=False)
        print(f"✅ 用户轨迹已保存：user_trajectory.csv")

    # =================== 绘制最佳 UAV 轨迹 ===================
    plt.figure(figsize=(8, 6))
    for uav_i in range(best_uav_trajectory.shape[1]):
        traj = best_uav_trajectory[:, uav_i, :]  # shape: (time_slots, 3)
        plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f'UAV {uav_i}')

    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Best UAV trajectories over 50 time slots')
    plt.legend()
    plt.grid(True)
    plt.show()