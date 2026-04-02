import os
import random
from datetime import datetime
from math import pi, sqrt

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from HPPO_agent import HPPO
from my_env import MyEnv
from utils.normalization import Normalization, RewardScaling


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tensorboard_writer(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def get_madrl_args():
    madrl_parser = argparse.ArgumentParser(description="MADRL 超参数")
    madrl_parser.add_argument("--actor_lr", type=float, default=3e-4, help="Actor 学习率") 
    madrl_parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic 学习率") 
    madrl_parser.add_argument("--lmbda", type=float, default=0.95, help="GAE (lambda)")
    madrl_parser.add_argument("--eps", type=float, default=0.2, help="PPO 裁剪因子 (epsilon)")
    madrl_parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子 (gamma)")
    madrl_parser.add_argument("--epochs", type=int, default=10, help="每次更新迭代次数")
    madrl_parser.add_argument("--total_time_slots", type=int, default=30, help="总时隙数量")
    madrl_parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    madrl_parser.add_argument("--episodes", type=int, default=5000, help="迭代轮次")
    return madrl_parser.parse_args()


def get_base_args():
    base_parser = argparse.ArgumentParser(description="场景的基本参数")

    # 仿真场景参数
    base_parser.add_argument("--num_cases", type=int, default=30, help="随机案例数量")
    base_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    base_parser.add_argument("--targets_num", type=int, default=40, help="目标数量")
    base_parser.add_argument("--uavs_num", type=int, default=4, help="UAV 数量")
    base_parser.add_argument("--cus_num", type=int, default=10, help="CU 数量")
    base_parser.add_argument("--uav_height", type=float, default=100, help="UAV 高度 (m)")
    base_parser.add_argument("--radius", type=float, default=600, help="区域半径 (m)")
    base_parser.add_argument("--center", type=float, default=[0, 0], help="区域中心坐标 (m)") 

    # 信道参数
    base_parser.add_argument("--ref_path_loss_db", type=float, default=-30, help="1m 参考路径损耗 (dB)")
    base_parser.add_argument("--frac_d_lambda", type=float, default=0.5, help="天线间距与波长比例")
    base_parser.add_argument("--alpha_uav_link", type=float, default=2, help="UAV 链路路径损耗指数")
    base_parser.add_argument("--alpha_cu_link", type=float, default=2.5, help="CU 链路路径损耗指数")
    base_parser.add_argument("--rician_factor_db", type=float, default=10, help="Rician 因子 (dB)")
    base_parser.add_argument("--antenna_nums", type=int, default=6, help="UAV 天线数量")
    base_parser.add_argument("--radar_rcs", type=float, default=10, help="雷达 RCS (mm^2)")
    base_parser.add_argument("--noise_power_density_dbm", type=float, default=-174, help="噪声功率谱密度 (dBm/Hz)")
    base_parser.add_argument("--bandwidth", type=float, default=10e6, help="带宽 (Hz)")

    # 算法/物理参数
    base_parser.add_argument("--uav_c1", type=float, default=0.00614, help="UAV 飞行参数 c1")
    base_parser.add_argument("--uav_c2", type=float, default=15.976, help="UAV 飞行参数 c2")
    base_parser.add_argument("--kappa", type=float, default=1e-28, help="BS CPU 有效开关电容")
    base_parser.add_argument("--bs_max_freq", type=float, default=10e9, help="BS 最大工作频率 (Hz)")
    base_parser.add_argument("--freq_scale", type=float, default=1e9, help="频率归一化尺度")
    base_parser.add_argument("--z_scale", type=float, default=1e5, help="z 变量归一化尺度")  
    base_parser.add_argument("--bs_cycles_per_bit", type=float, default=1000, help="BS 处理 1bit 需要的周期数")
    base_parser.add_argument("--time_slot_duration", type=float, default=0.6, help="时隙长度 (s)")
    base_parser.add_argument("--uav_sen_duration", type=float, default=0.1, help="UAV 感知时长 (s)")
    base_parser.add_argument("--cu_max_power_dbm", type=float, default=23, help="CU 最大发射功率 (dBm)")
    base_parser.add_argument("--uav_max_power", type=float, default=10, help="UAV 最大功率 (W) = 40dBm")
    base_parser.add_argument("--cu_max_delay", type=float, default=0.6, help="娱乐任务最大延迟 (s)")
    base_parser.add_argument("--uav_max_delay", type=float, default=0.2, help="感知任务最大延迟 (s)")
    base_parser.add_argument("--uav_max_speed", type=float, default=40.0, help="UAV 最大移动速度 (m/s)")
    base_parser.add_argument("--uav_min_speed", type=float, default=5.0, help="UAV 最小移动速度 (m/s)")
    base_parser.add_argument("--uav_safe_distance", type=float, default=5.0, help="UAV 安全距离 (m)")
    base_parser.add_argument("--sen_sinr", type=float, default=20, help="感知门限阈值 (dB)")
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
    base_parser.add_argument("--max_iterations", type=int, default=10, help="CCCP 算法最大迭代次数")
    base_parser.add_argument("--cccp_threshold", type=float, default=1e-4, help="CCCP 算法目标函数收敛阈值")
    base_parser.add_argument("--rank1_threshold", type=float, default=1e-4, help="CCCP 算法秩一约束收敛阈值")
    base_parser.add_argument("--penalty_factor", type=float, default=0.1, help="罚因子")
    base_parser.add_argument("--zoom_factor", type=float, default=2, help="缩放系数")
    base_parser.add_argument("--enable_cccp_diagnostics", default="True", help="启用 CCCP 诊断，检查下一轮线性化后的贴合性和可行性")
    base_parser.add_argument("--diagnostic_violation_tol", type=float, default=1e-7, help="CCCP 诊断时的违约容差")
    base_parser.add_argument("--diagnostic_top_k", type=int, default=5, help="CCCP 诊断时打印的最大违约约束组数量")
    base_parser.add_argument("--constraint_include_groups", type=str, default="4.5,4.12,4.23,4.25,4.27,4.28,4.29,4.32,4.39,4.40,4.44,4.45,auxiliary_t,var", help="启用约束组，逗号分隔")
    base_parser.add_argument("--constraint_exclude_groups", type=str, default="", help="禁用约束组，逗号分隔")

    base_parser.add_argument("--linearization_psi_floor", type=float, default=1e-10, help="CCCP 线性化里 Psi 的数值下界，避免 1/Psi 和 Psi^{-1} 爆大")
    base_parser.add_argument("--enable_first_iter_rank_boost", type=lambda x: str(x).lower() == "true", default=False,
                             help="开关参数")
    base_parser.add_argument("--first_iter_rank_boost_eps", type=float, default=0.1,
                             help="强度参数")
    base_parser.add_argument("--solver_backend", type=str, default="fusion", choices=["fusion", "cvxpy"],
                             help="是否采用 Fusion 求解器，默认为 True（使用 Mosek Fusion），否则使用 CVXPY（默认使用 Mosek 作为 CVXPY 的求解器）")
    base_parser.add_argument("--enable_initial_anchor", type=lambda x: str(x).lower() == "true", default=False,
                             help="是否启用初始化描点/锚点可行性检查，默认为 False")
    
    return base_parser.parse_args()


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = get_tensorboard_writer(log_dir=f"runs/beta-hppo/{current_time_str}_beta-hppo_result")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args = get_base_args()
    madrl_args = get_madrl_args()
    setSeed(seed=base_args.seed)

    env = MyEnv(base_args=base_args, madrl_args=madrl_args)

    state_dim_bs = env.observation_space["bs"].shape[0]
    bs_continuous_action_space = env.action_space["bs"]["continuous"]
    bs_discrete_action_space = env.action_space["bs"]["discrete"]

    agent_bs = HPPO(
        state_dim=state_dim_bs,
        hidden_dim=madrl_args.hidden_dim,
        continuous_action_splits=env.bs_continuous_action_splits,
        discrete_action_dims=bs_discrete_action_space.nvec,
        action_low=bs_continuous_action_space.low,
        action_high=bs_continuous_action_space.high,
        actor_lr=madrl_args.actor_lr,
        critic_lr=madrl_args.critic_lr,
        lmbda=madrl_args.lmbda,
        eps=madrl_args.eps,
        gamma=madrl_args.gamma,
        epochs=madrl_args.epochs,
        num_episodes=madrl_args.episodes,
        device=device,
        entropy_coef=0.01
    )

    running_norm_bs = Normalization(state_dim_bs)
    reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

    reward_res = []
    all_agents_rewards = []
    max_avg_reward = -np.inf
    best_uav_trajectory = None
    cu_trajectory = None
    target_trajectory = None

    with tqdm(total=int(madrl_args.episodes), desc="Training Progress") as pbar:
        for i_episode in range(int(madrl_args.episodes)):
            episode_rewards_total = []
            episode_reward_bs = 0.0

            transition_dict_bs = {
                "states": [],
                "continuous_actions": [],
                "discrete_actions": [],
                "next_states": [],
                "rewards": [],
                "old_cont_log_probs": [],
                "old_disc_log_probs": [],
                "dones": [],
                "real_dones": [],
            }

            s = env.reset()
            terminal = False
            uav_positions_episode = []
            cu_positions_episode = []
            target_positions_episode = []
            reward_scaler_bs.reset()

            while not terminal:
                uav_positions_episode.append(env.getPosUAV())
                cu_positions_episode.append(env.getPosCU())
                target_positions_episode.append(env.getPosTarget())

                state_bs_norm = running_norm_bs(np.array(s["bs"], dtype=np.float32))
                action_bs, old_log_probs_bs, old_con_log_probs_bs, old_dis_log_probs_bs = agent_bs.choose_action(state_bs_norm)

                next_s, total_reward, r_dict, done = env.step({"bs": action_bs}, i_episode)

                r_bs = float(r_dict["bs"])
                r_bs_norm = float(np.asarray(reward_scaler_bs(r_bs)).item())

                episode_rewards_total.append(float(total_reward))
                episode_reward_bs += r_bs

                next_state_bs_norm = running_norm_bs(np.array(next_s["bs"], dtype=np.float32))
                transition_dict_bs["states"].append(state_bs_norm)
                transition_dict_bs["continuous_actions"].append(action_bs["continuous"])
                transition_dict_bs["discrete_actions"].append(action_bs["discrete"])
                transition_dict_bs["next_states"].append(next_state_bs_norm)
                transition_dict_bs["rewards"].append(r_bs_norm)
                transition_dict_bs["old_cont_log_probs"].append(float(np.asarray(old_con_log_probs_bs).item()))
                transition_dict_bs["old_disc_log_probs"].append(float(np.asarray(old_dis_log_probs_bs).item()))
                transition_dict_bs["dones"].append(bool(done))
                transition_dict_bs["real_dones"].append(False)

                s = next_s
                terminal = done

            if np.mean(episode_rewards_total) > max_avg_reward:
                max_avg_reward = np.mean(episode_rewards_total)
                if len(uav_positions_episode) > 0:
                    best_uav_trajectory = np.array(uav_positions_episode, copy=True)
                if len(cu_positions_episode) > 0:
                    cu_trajectory = np.array(cu_positions_episode, copy=True)
                if len(target_positions_episode) > 0:
                    target_trajectory = np.array(target_positions_episode, copy=True)

            agent_bs.update(transition_dict_bs, i_episode, writer, agent_name="BS")

            avg_total_reward = np.mean(episode_rewards_total)
            avg_bs_reward = episode_reward_bs / len(episode_rewards_total)

            reward_res.append(avg_total_reward)
            all_agents_rewards.append([avg_bs_reward])

            writer.add_scalar("Reward/episode", avg_total_reward, i_episode)

            pbar.set_postfix({"avg_reward": f"{avg_total_reward:.3f}"})
            pbar.update(1)

    writer.close()

    reward_array = np.array(reward_res)
    episodes_list = np.arange(reward_array.shape[0])
    plt.plot(episodes_list, reward_array)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Single-Agent Beta-PPO training performance")
    plt.show()

    filename = f"{current_time_str}_HPPO_training_rewards_seed_{base_args.seed}.csv"
    df = pd.DataFrame({
        "episode": episodes_list,
        "reward": reward_array
    })
    df.to_csv(filename, index=False)
    print(f"HPPO 训练文件已保存至 {filename}")

    all_agents_rewards = np.array(all_agents_rewards)
    df_agents = pd.DataFrame(all_agents_rewards, columns=["BS_reward"])
    df_agents.insert(0, "episode", np.arange(madrl_args.episodes))
    filename_agents = f"{current_time_str}_HPPO_all_agents_rewards_seed_{base_args.seed}.csv"
    df_agents.to_csv(filename_agents, index=False)
    print(f"HPPO BS Agent 训练奖励已保存至 {filename_agents}")

    if best_uav_trajectory is not None:
        uav_traj_list = []
        for t in range(best_uav_trajectory.shape[0]):
            for uav_i in range(base_args.uavs_num):
                x, y, z = best_uav_trajectory[t, uav_i]
                uav_traj_list.append([t, uav_i, x, y, z])
        df_uav = pd.DataFrame(uav_traj_list, columns=["time_slot", "uav_id", "x", "y", "z"])
        df_uav.to_csv(f"{current_time_str}_SEED{base_args.seed}_best_uav_trajectory.csv", index=False)
        print(f"HPPO 最佳UAV轨迹已保存至 {current_time_str}_SEED{base_args.seed}_best_uav_trajectory.csv")

    if cu_trajectory is not None:
        cu_traj_list = []
        for t in range(cu_trajectory.shape[0]):
            for cu_i in range(base_args.cus_num):
                x, y, z = cu_trajectory[t, cu_i]
                cu_traj_list.append([t, cu_i, x, y, z])
        df_cu = pd.DataFrame(cu_traj_list, columns=["time_slot", "cu_id", "x", "y", "z"])
        df_cu.to_csv(f"{current_time_str}_SEED{base_args.seed}_cu_trajectory.csv", index=False)
        print(f"HPPO CU轨迹已保存至 {current_time_str}_SEED{base_args.seed}_cu_trajectory.csv")

    if target_trajectory is not None:
        target_traj_list = []
        for t in range(target_trajectory.shape[0]):
            for target_i in range(base_args.targets_num):
                x, y, z = target_trajectory[t, target_i]
                target_traj_list.append([t, target_i, x, y, z])
        df_target = pd.DataFrame(target_traj_list, columns=["time_slot", "target_id", "x", "y", "z"])
        df_target.to_csv(f"{current_time_str}_SEED{base_args.seed}_target_trajectory.csv", index=False)
        print(f"HPPO TARGET轨迹已保存至 {current_time_str}_SEED{base_args.seed}_target_trajectory.csv")
