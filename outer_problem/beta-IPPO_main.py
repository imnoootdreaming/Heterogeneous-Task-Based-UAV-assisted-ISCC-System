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
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.normalization import Normalization, RewardScaling
from IPPO_agent import IPPO, orthogonal_init
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
from math import pi, sqrt # 2026-03-16: Added math imports for radar parameters

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def setSeed(seed):
    """
    设置随机种子，确保仿真可复现
    """
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

def db_2_watt(db):
    """
    将 dB 转换为瓦特
    :param db: 功率值（dB）
    :return: 功率值（瓦特）
    """
    return 10 ** (db / 10)

def get_madrl_args():
    madrl_parser = argparse.ArgumentParser(description="MADRL 超参数")
    madrl_parser.add_argument("--actor_lr", type=float, default=3e-4, help="actor 学习率") # Changed to float
    madrl_parser.add_argument("--critic_lr", type=float, default=3e-4, help="critic 学习率") # Changed to float
    madrl_parser.add_argument("--lmbda", type=float, default=0.95, help="GAE (lambda)")
    madrl_parser.add_argument("--eps", type=float, default=0.2, help="PPO 裁剪因子 (epsilon)")
    madrl_parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子 (gamma)")
    madrl_parser.add_argument("--epochs", type=int, default=10, help="每次更新迭代次数")
    madrl_parser.add_argument("--total_time_slots", type=int, default=30, help="总时隙数量")
    madrl_parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    madrl_parser.add_argument("--episodes", type=int, default=2000, help="迭代轮次")
    return madrl_parser.parse_args()


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
    base_parser.add_argument("--center", type=float, default=[0, 0], help="区域中心坐标 (m)") 

    # 信道参数
    base_parser.add_argument("--ref_path_loss_db", type=float, default=-30, help="1m 参考路径损耗 (dB)")
    base_parser.add_argument("--frac_d_lambda", type=float, default=0.5, help="天线间距与波长比例")
    base_parser.add_argument("--alphaction_uav_link", type=float, default=2, help="UAV 链路路径损耗指数")
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
    base_parser.add_argument("--markov_velocity", type=float, default=[1, 0, 0], help="马尔可夫模型初始速度 (m/s)")
    base_parser.add_argument("--markov_memory_level", type=float, default=0.4, help="马尔可夫模型记忆水平")
    base_parser.add_argument("--markov_asymptotic_mean_of_velocity", type=float, default=[1, 0, 0], help="马尔可夫模型渐进平均值")
    base_parser.add_argument("--markov_standard_deviation_of_velocity", type=float, default=2, help="马尔可夫模型速度标准差")

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
    
    # 2026-03-16: Fixed f-string format for python compatibility if needed
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = get_tensorboard_writer(log_dir = f"runs/beta-ippo/{current_time_str}_beta-ippo_result")  # 利用 tensorboard 可视化训练结果
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")  # 获取 device

    base_args = get_base_args()  # 获取系统环境参数
    madrl_args = get_madrl_args()  # 获取 MADRL 参数

    # 创建环境
    env = MyEnv(base_args = base_args, madrl_args = madrl_args) 

    # 获取 UAV 和 BS 的状态维度
    state_dim_uav = env.observation_space["uav"]["uav_0"].shape[0]
    state_dim_bs = env.observation_space["bs"].shape[0]

    # 获取 UAV 和 BS 的动作维度
    action_space_uav = env.action_space["uav"]["uav_0"]
    action_space_bs = env.action_space["bs"]
    
    # 2026-03-16: Reverted to single continuous action space logic
    action_dim_bs = action_space_bs.shape[0]
    action_dim_uav = action_space_uav["uav_action_continuous"].shape[0]
    
    # 获取 UAV 和 BS 的最高最低值
    action_uav_low = action_space_uav["uav_action_continuous"].low
    action_uav_high = action_space_uav["uav_action_continuous"].high
    bs_action_low = action_space_bs.low
    bs_action_high = action_space_bs.high

    # UAV 和 RIS Agent
    agents_uav = {}
    running_norms_uav = {}
    for i in range(base_args.uavs_num):
        agent_id = f"uav_{i}"
        agents_uav[agent_id] = IPPO(state_dim_uav, madrl_args.hidden_dim, action_dim_uav,
                                    action_uav_low, action_uav_high,
                                    madrl_args.actor_lr, madrl_args.critic_lr, madrl_args.lmbda, 
                                    madrl_args.eps, madrl_args.gamma, madrl_args.epochs, madrl_args.num_episodes, device,
                                    policy_dist = "Beta", entropy_coef = 0.01)
        running_norms_uav[agent_id] = Normalization(state_dim_uav)  # 为每个 agent 各建一个动态归一化状态空间

    agent_bs = IPPO(state_dim_bs, madrl_args.hidden_dim, action_dim_bs, 
                    bs_action_low, bs_action_high,
                    madrl_args.actor_lr, madrl_args.critic_lr, madrl_args.lmbda, 
                    madrl_args.eps, madrl_args.gamma, madrl_args.epochs, madrl_args.num_episodes, device,
                    policy_dist = "Beta", entropy_coef = 0.01)
    
    running_norm_bs = Normalization(state_dim_bs)  # 为每个 agent 各建一个动态归一化状态空间

    reward_scalers_uav = [RewardScaling(shape=1, gamma=madrl_args.gamma) for _ in range(base_args.uavs_num)]
    reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

    reward_res = []
    all_agents_rewards = []  # 每个agent的奖励记录：[[uav_0, uav_1, ..., ris], ...]
    max_avg_reward = -np.inf  # 用于记录最大的平均奖励
    best_uav_trajectory = None
    cu_trajectory = None

    with tqdm(total=int(madrl_args.episodes), desc='Training Progress') as pbar:
        for i_episode in range(int(madrl_args.episodes)):
            # 每个episode存储各UAV奖励
            episode_rewards_total = []  # 总奖励（环境返回的 total_reward）
            episode_rewards_uav = np.zeros(base_args.uavs_num)  # 每个UAV奖励累计
            episode_reward_bs = 0.0  # BS 奖励累计

            #  为每个 UAV agent 创建独立的 transition_dict
            transition_dicts_uav = {
                f"uav_{uav_i}": {'states': [],
                             'actions': [],
                             'next_states': [],
                             'rewards': [],
                             'old_log_probs': [],
                             'dones': [],
                             'real_dones': []}
                for uav_i in range(base_args.uavs_num)}
            #  为 BS agent 创建独立的 transition_dict
            transition_dict_bs = {
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
            cu_positions_episode = []  # 初始化当前 episode 的 CU 轨迹存储
            target_positions_episode = []  # 初始化当前 episode 的 Target 轨迹存储

            # reset reward scaling
            for rs in reward_scalers_uav:
                rs.reset()
            reward_scaler_bs.reset()

            while not terminal:
                # --- 为每个 UAV agent 独立处理状态、动作 ---
                actions_uav_dict = {} # For Buffer (Concatenated)
                # actions_uav_env_dict = {} # Removed: Now passing actions directly
                old_log_probs_uav_dict = {}
                state_uav_norm_dict = {}  # 临时存储归一化后的状态
                
                
                uav_positions_episode.append(env.getPosUAV())  # --- 存储当前时隙 UAV 坐标 ---
                
                cu_positions_episode.append(env.getPosCU())  # --- 存储当前时隙用户坐标 ---
                
                target_positions_episode.append(env.getPosTarget())  # --- 存储当前时隙 Target 坐标 ---

                # UAV agents 选择动作
                for uav_i in range(base_args.uavs_num):
                    agent_id = f"uav_{uav_i}"
                    # 更新并归一化该 UAV 的状态
                    state_uav_norm = running_norms_uav[agent_id](np.array(s["uav"][agent_id]))
                    state_uav_norm_dict[agent_id] = state_uav_norm

                    # 该 UAV agent 采取动作
                    action_uav, old_log_probs_uav = agents_uav[agent_id].choose_action(state_uav_norm)
                    
                    # 存储 UAV
                    actions_uav_dict[agent_id] = action_uav
                    old_log_probs_uav_dict[agent_id] = old_log_probs_uav
                    
                # BS agent 选择动作
                state_bs_norm = running_norm_bs(np.array(s["bs"]))
                action_bs, old_log_probs_bs = agent_bs.choose_action(state_bs_norm)
                
                # 环境执行
                
                next_s, total_reward, r_dict, done = env.step({"uav": actions_uav_dict, "bs": action_bs}, i_episode)
                # =====================================================
                #  r 是 dict： r["uav"], r["bs"]
                # =====================================================
                # 解析奖励
                r_uav_list = [np.mean(r_val) if np.ndim(r_val) > 0 and len(r_val) > 0 else float(r_val) for r_val in r_dict["uav"]]
                r_bs = np.mean(r_dict["bs"]) if np.ndim(r_dict["bs"]) > 0 and len(r_dict["bs"]) > 0 else float(r_dict["bs"])

                # 奖励归一化
                r_uav_norm = [reward_scalers_uav[j](r_uav_list[j]) for j in range(base_args.uavs_num)]
                r_bs_norm = reward_scaler_bs(r_bs)

                # 累计各agent奖励
                episode_rewards_total.append(total_reward)
                episode_rewards_uav += np.array(r_uav_norm)
                episode_reward_bs += r_bs_norm

                # --- 为每个 UAV agent 存储经验 ---
                for uav_i in range(base_args.uavs_num):
                    agent_id = f"uav_{uav_i}"
                    # 对 next_state 同样进行 normalize
                    next_state_uav_norm = running_norms_uav[agent_id](next_s["uav"][agent_id])

                    # 存储已归一化的数据到对应的字典
                    transition_dicts_uav[agent_id]['states'].append(state_uav_norm_dict[agent_id])
                    transition_dicts_uav[agent_id]['actions'].append(actions_uav_dict[agent_id]) # Store Concatenated
                    transition_dicts_uav[agent_id]['next_states'].append(next_state_uav_norm)
                    transition_dicts_uav[agent_id]['rewards'].append(r_uav_norm[uav_i])  # 共享奖励
                    transition_dicts_uav[agent_id]['old_log_probs'].append(old_log_probs_uav_dict[agent_id])
                    transition_dicts_uav[agent_id]['dones'].append(bool(done))
                    transition_dicts_uav[agent_id]['real_dones'].append(False)

                # --- 为 BS agent 存储经验 ---
                next_state_bs_norm = running_norm_bs(next_s["bs"])
                # 存储已归一化的数据
                transition_dict_bs['states'].append(state_bs_norm)
                transition_dict_bs['actions'].append(action_bs)
                transition_dict_bs['next_states'].append(next_state_bs_norm)
                transition_dict_bs['rewards'].append(r_bs_norm)
                transition_dict_bs['old_log_probs'].append(old_log_probs_bs)
                transition_dict_bs['dones'].append(bool(done))
                transition_dict_bs['real_dones'].append(False)

                s = next_s
                terminal = done

            if np.mean(episode_rewards_total) > max_avg_reward:
                max_avg_reward = np.mean(episode_rewards_total)
                if len(uav_positions_episode) > 0:
                    best_uav_trajectory = np.array(uav_positions_episode)  # shape: (50, uav_num, 3)
                if len(cu_positions_episode) > 0:
                    cu_trajectory = np.array(cu_positions_episode)
                if len(target_positions_episode) > 0:
                    target_trajectory = np.array(target_positions_episode) # 2026-03-16: Store target trajectory

            # ---  更新所有 UAV Agents ---
            for uav_i in range(base_args.uavs_num):
                agent_id = f"uav_{uav_i}"
                agents_uav[agent_id].update(transition_dicts_uav[agent_id], i_episode, writer,
                                            agent_name=f"UAV_{uav_i}")
            # BS Agent 更新
            agent_bs.update(transition_dict_bs, i_episode, writer, agent_name="BS")

            # 统计奖励
            avg_total_reward = np.mean(episode_rewards_total)
            avg_uav_rewards = episode_rewards_uav / len(episode_rewards_total)
            avg_bs_reward = episode_reward_bs / len(episode_rewards_total)

            reward_res.append(np.mean(episode_rewards_total))  # 统计平均奖励
            all_agents_rewards.append(np.concatenate([avg_uav_rewards, [avg_bs_reward]]))

            # Average Reward 写入 TensorBoard
            writer.add_scalar("Reward/episode", avg_total_reward, i_episode)
            for uav_i in range(base_args.uavs_num):
                writer.add_scalar(f"Reward/UAV_{uav_i}", avg_uav_rewards[uav_i], i_episode)
            writer.add_scalar("Reward/BS", avg_bs_reward, i_episode)

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
    plt.title('Beta-IPPO training performance')
    plt.show()

    # =========== 保存 reward csv 文件 ===========
    # 构造文件名
    filename = f'{current_time_str}_IPPO_training_rewards_seed_{base_args.seed}.csv'
    # 保存 reward_array 和 episodes_list 到 CSV
    df = pd.DataFrame({
        'episode': episodes_list,
        'reward': reward_array
    })
    df.to_csv(filename, index=False)
    print(f"训练奖励已保存到 {filename}")
    # =========== 保存 reward csv 文件 ===========

    # ========= 保存每个Agent奖励 ========= #
    all_agents_rewards = np.array(all_agents_rewards)
    columns = [f"UAV_{i}_reward" for i in range(base_args.uavs_num)] + ["BS_reward"]
    df_agents = pd.DataFrame(all_agents_rewards, columns=columns)
    df_agents.insert(0, "episode", np.arange(madrl_args.episodes))
    filename_agents = f'{current_time_str}_IPPO_all_agents_rewards_seed_{base_args.seed}.csv'
    df_agents.to_csv(filename_agents, index=False)
    print(f"✅ 所有 Agent 的奖励已保存到 {filename_agents}")

    # =================== 保存最佳 UAV 轨迹到 CSV ===================
    if best_uav_trajectory is not None:
        # ===== 保存 UAV 轨迹 =====
        uav_traj_list = []
        for t in range(best_uav_trajectory.shape[0]):
            for uav_i in range(base_args.uavs_num):
                x, y, z = best_uav_trajectory[t, uav_i]
                uav_traj_list.append([t, uav_i, x, y, z])
        df_uav = pd.DataFrame(uav_traj_list, columns=['time_slot', 'uav_id', 'x', 'y', 'z'])
        df_uav.to_csv(f'{current_time_str}_SEED{base_args.seed}_best_uav_trajectory.csv', index=False)
        print(f"✅ 最佳 UAV 轨迹已保存：{current_time_str}_SEED{base_args.seed}_best_uav_trajectory.csv")

    # ===== 保存 CU 轨迹 =====
    if cu_trajectory is not None:
        cu_traj_list = []
        for t in range(cu_trajectory.shape[0]):
            for cu_i in range(base_args.cus_num):
                x, y, z = cu_trajectory[t, cu_i]
                cu_traj_list.append([t, cu_i, x, y, z])
        df_cu = pd.DataFrame(cu_traj_list, columns=['time_slot', 'cu_id', 'x', 'y', 'z'])
        df_cu.to_csv(f'{current_time_str}_SEED{base_args.seed}_cu_trajectory.csv', index=False)
        print(f"✅ CU 轨迹已保存：{current_time_str}_SEED{base_args.seed}_cu_trajectory.csv")

    # ===== 保存 TARGET 位置 =====
    if target_trajectory is not None:
        target_traj_list = []
        for t in range(target_trajectory.shape[0]):
            for target_i in range(base_args.targets_num):
                x, y, z = target_trajectory[t, target_i]
                target_traj_list.append([t, target_i, x, y, z])
        df_target = pd.DataFrame(target_traj_list, columns=['time_slot', 'target_id', 'x', 'y', 'z'])
        df_target.to_csv(f'{current_time_str}_SEED{base_args.seed}_target_trajectory.csv', index=False)
        print(f"✅ TARGET 轨迹已保存：{current_time_str}_SEED{base_args.seed}_target_trajectory.csv")
