import argparse
from math import pi, sqrt
import os
import time
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
from datetime import datetime  # 新增
import csv  # 新增：用于将迭代数据写入 CSV 文件
from scipy.optimize import linear_sum_assignment  # 引入线性求和分配函数
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('error', category=RuntimeWarning)  # 把 RuntimeWarning 当作异常处理
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # 新增


def get_base_args():
    base_parser = argparse.ArgumentParser(description="场景的基本参数")

    # 仿真场景参数
    base_parser.add_argument("--num_cases", type=int, default=30, help="随机案例数量")
    base_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    base_parser.add_argument("--targets_num", type=int, default=8, help="目标数量")
    base_parser.add_argument("--uavs_num", type=int, default=8, help="UAV 数量")
    base_parser.add_argument("--cus_num", type=int, default=16, help="CU 数量")
    base_parser.add_argument("--uav_height", type=float, default=100, help="UAV 高度 (m)")
    base_parser.add_argument("--radius", type=float, default=200, help="区域半径 (m)")

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
    base_parser.add_argument("--bs_cycles_per_bit", type=float, default=1000, help="BS 处理 1bit 需要的周期数")
    base_parser.add_argument("--time_slot_duration", type=float, default=0.6, help="时隙长度 (s)")
    base_parser.add_argument("--uav_sen_duration", type=float, default=0.1, help="UAV 感知时长 (s)")
    base_parser.add_argument("--cu_max_power_dbm", type=float, default=23, help="CU 最大发射功率 (dBm)")
    base_parser.add_argument("--uav_max_power", type=float, default=10, help="UAV 最大功率 (W) = 40dBm")
    base_parser.add_argument("--cu_max_delay", type=float, default=0.6, help="娱乐任务最大延迟 (s)")
    base_parser.add_argument("--uav_max_delay", type=float, default=0.3, help="感知任务最大延迟 (s)")
    base_parser.add_argument("--uav_max_speed", type=float, default=10.0, help="UAV 最大移动速度 (m/s)")
    base_parser.add_argument("--uav_min_speed", type=float, default=5.0, help="UAV 最小移动速度 (m/s)")
    base_parser.add_argument("--uav_safe_distance", type=float, default=5.0, help="UAV 安全距离 (m)")
    base_parser.add_argument("--sen_sinr", type=float, default=20, help="感知门限阈值 (dB)")

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
    base_parser.add_argument("--max_iterations", type=int, default=15, help="CCCP 算法最大迭代次数")
    base_parser.add_argument("--cccp_threshold", type=float, default=1e-6, help="CCCP 算法目标函数收敛阈值")
    base_parser.add_argument("--rank1_threshold", type=float, default=1e-3, help="CCCP 算法秩一约束收敛阈值")
    base_parser.add_argument("--penalty_factor", type=float, default=1, help="罚因子")
    base_parser.add_argument("--zoom_factor", type=float, default=2, help="缩放系数")
    base_parser.add_argument("--constraint_include_groups", type=str, default="4.5,4.12,4.23,4.25,4.27,4.28,4.29,4.32,4.39,4.40,4.44,4.45，auxiliary_t,var", help="启用约束组，逗号分隔")
    base_parser.add_argument("--constraint_exclude_groups", type=str, default="", help="禁用约束组，逗号分隔")

    return base_parser.parse_args()


def compute_largest_eigenvector(matrix):
    """
    计算矩阵中最大特征值对应的特征向量

    :param matrix: 需要计算最大特征值对应的特征向量的矩阵
    :return: 矩阵中最大特征值对应的特征向量
    """
    # 确保矩阵的对称性，防止数值截断误差
    matrix = (matrix + matrix.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # eigh 返回特征值升序排列，取最后一个
    v_max = eigenvectors[:, -1]
    # 返回最大特征值对应的特征向量
    return v_max


def generate_pos(uavs_num, cus_num, targets_num, center, radius, uav_height):
    """
    根据圆心和半径生成 UAVs、CUs 和 targets 的随机位置

    :param uavs_num: UAVs 的数量
    :param cus_num: CUs 的数量
    :param targets_num: targets 的数量
    :param center: 圆心坐标 (x, y)
    :param radius: 半径
    :param uav_height: UAVs 的高度
    :return: UAVs, CUs 和 targets 的随机位置
    """
    # 生成 UAVs 位置
    r_uav = radius * np.sqrt(np.random.rand(uavs_num))
    theta_uav = np.random.rand(uavs_num) * 2 * np.pi
    uavs_pos = np.zeros((uavs_num, 3))
    uavs_pos[:, 0] = center[0] + r_uav * np.cos(theta_uav)
    uavs_pos[:, 1] = center[1] + r_uav * np.sin(theta_uav)
    uavs_pos[:, 2] = uav_height

    # 生成 CUs 位置 (假设在地面 z = 0)
    r_cu = radius * np.sqrt(np.random.rand(cus_num))
    theta_cu = np.random.rand(cus_num) * 2 * np.pi
    cus_pos = np.zeros((cus_num, 3))
    cus_pos[:, 0] = center[0] + r_cu * np.cos(theta_cu)
    cus_pos[:, 1] = center[1] + r_cu * np.sin(theta_cu)
    # cus_pos[:, 2] = 0  # 默认为 0

    # 生成 targets 位置 (假设在地面 z = 0)
    r_target = radius * np.sqrt(np.random.rand(targets_num))
    theta_target = np.random.rand(targets_num) * 2 * np.pi
    targets_pos = np.zeros((targets_num, 3))
    targets_pos[:, 0] = center[0] + r_target * np.cos(theta_target)
    targets_pos[:, 1] = center[1] + r_target * np.sin(theta_target)
    # targets_pos[:, 2] = 0  # 默认为 0

    return uavs_pos, cus_pos, targets_pos


def compute_com_channel_gain(uavs_pos, cus_pos, ref_path_loss, frac_d_lambda, alpha_uav_link, alpha_cu_link, rician_factor, antenna_nums):
    """
    UAVs, CUs 和 BS 之间的均建模为莱斯路径损耗模型
    根据莱斯路径损耗模型计算 UAVs、CUs 和 BS 之间的通信信道增益

    :param uavs_pos: UAVs 的位置 (I * 3)
    :param cus_pos: CUs 的位置 (J * 3)
    :param ref_path_loss: 1m 路径参考路径损耗
    :param frac_d_lambda: 天线间距与波长的比例
    :param alpha_uav_link: 与 UAV 有关链路的路径损耗系数
    :param alpha_cu_link: 与 CU 有关的路径损耗系数
    :param rician_factor: Rician 因子
    :param antenna_nums: 天线数量 N
    :return: UAVs -> CUs 信道 (I * J * N), UAVs -> BS 信道 (I * 1 * N), CUs -> BS 信道 (J * 1)
    """
    bs_pos = np.array([0, 0, 0])  # 假设 BS 位于原点

    def get_rician_channel(pos1, pos2, alpha, K, is_mimo=True):
        # 计算距离
        # pos1: (N_pos1, 3), pos2: (N_pos2, 3) -> dist: (N_pos1, N_pos2)
        diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # 路径损耗
        path_loss = ref_path_loss * (dist ** -alpha)
        
        if is_mimo:
            # 计算方位角 phi (AoD)
            # diff shape: (N_pos1, N_pos2, 3), diff[..., 0] is dx, diff[..., 1] is dy
            dx = diff[..., 0]
            dy = diff[..., 1]
            phi = np.arctan2(dy, dx)
            
            # LoS 分量 (Array Response Vector)
            # h_los shape: (N_pos1, N_pos2, antenna_nums)
            # array_response: [1, exp(j*2*pi*d/lambda*sin(phi)), ..., exp(j*2*pi*d/lambda*(N-1)*sin(phi))]
            n_range = np.arange(antenna_nums)
            exponent = 1j * 2 * np.pi * frac_d_lambda * np.sin(phi)[..., np.newaxis] * n_range
            h_los = np.exp(exponent)
            
            # NLoS 分量 (Rayleigh)
            # h_nlos shape: (N_pos1, N_pos2, antenna_nums)
            h_nlos = (np.random.randn(*dist.shape, antenna_nums) + 1j * np.random.randn(*dist.shape, antenna_nums)) / np.sqrt(2)
            
            # 扩展 path_loss 维度以匹配 MIMO 信道
            path_loss_expanded = path_loss[..., np.newaxis]
            
            h = np.sqrt(path_loss_expanded) * (np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos)
            
        else:
            # SISO 情况 (CU -> BS)
            # LoS 分量 = 1
            h_los = 1.0
            
            # NLoS 分量 (Rayleigh)
            h_nlos = (np.random.randn(*dist.shape) + 1j * np.random.randn(*dist.shape)) / np.sqrt(2)
            
            h = np.sqrt(path_loss) * (np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos)

        return h

    # UAVs -> CUs (MIMO, N antennas at UAV)
    uavs_2_cus_channels = get_rician_channel(uavs_pos, cus_pos, alpha_uav_link, rician_factor, is_mimo=True)
    
    # UAVs -> BS (MIMO, N antennas at UAV)
    uavs_2_bs_channels = get_rician_channel(uavs_pos, bs_pos[np.newaxis, :], alpha_uav_link, rician_factor, is_mimo=True)
    
    # CUs -> BS (SISO, 1 antenna at CU)
    cus_2_bs_channels = get_rician_channel(cus_pos, bs_pos[np.newaxis, :], alpha_cu_link, rician_factor, is_mimo=False)

    return uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels


def compute_sen_channel_gain(radar_rcs, frac_d_lambda, uavs_pos, targets_pos, antenna_nums, ref_path_loss):
    """
    根据雷达截面积、天线间距与波长的比例，UAVs 的位置和 targets 的位置计算 UAVs 与 targets 之间的信道响应矩阵

    :param radar_rcs: 雷达截面积 (xi)
    :param frac_d_lambda: 天线间距与波长的比例
    :param uavs_pos: UAVs 的位置 (I * 3)
    :param targets_pos: targets 的位置 (J * 3)
    :param antenna_nums: 天线数量 N
    :param ref_path_loss: 1m 参考距离下的路径损耗
    :return: UAVs 与 targets 之间的信道响应矩阵 A (I * J * N * N)
    """
    # 计算距离
    # diff: (I, J, 3)
    diff = uavs_pos[:, np.newaxis, :] - targets_pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # 计算方位角 theta (AoD/AoA)
    dx = diff[..., 0]
    dy = diff[..., 1]
    theta = np.arctan2(dy, dx)
    
    # 生成转向矢量 a_t 和 a_r (假设收发共址，角度相同)
    # n_range: (N,)
    n_range = np.arange(antenna_nums)
    # exponent: (I, J, N)
    exponent = 1j * 2 * np.pi * frac_d_lambda * np.sin(theta)[..., np.newaxis] * n_range
    # a_vec: (I, J, N)
    a_vec = np.exp(exponent)
    
    # 计算信道矩阵 A = sqrt(xi * ref_path_loss * d^-4) * a_r * a_t^H
    # path_gain_amplitude: (I, J)
    # 注意：原文公式为 sqrt(xi * ref_path_loss * d^-4)，其中 ref_path_loss 应为路径损耗常数
    # 通常雷达方程中接收功率 Pr ~ Pt * G^2 * lambda^2 * RCS / ((4pi)^3 * d^4)
    # 这里按照用户给定的公式实现：sqrt(xi * ref_path_loss * d^-4)
    path_gain_amplitude = np.sqrt(radar_rcs * ref_path_loss * (dist ** -4))
    
    # 扩展维度以进行矩阵乘法
    # a_vec_col: (I, J, N, 1) -> a_r
    a_vec_col = a_vec[..., np.newaxis]
    # a_vec_row_conj: (I, J, 1, N) -> a_t^H
    a_vec_row_conj = np.conj(a_vec)[..., np.newaxis, :]
    
    # matrix_term: (I, J, N, N) = a_r * a_t^H
    matrix_term = np.matmul(a_vec_col, a_vec_row_conj)
    
    # uavs_2_targets_channels: (I, J, N, N)
    uavs_2_targets_channels = path_gain_amplitude[..., np.newaxis, np.newaxis] * matrix_term
    
    return uavs_2_targets_channels


def match_uav_targets_nearest(uavs_pos, targets_pos):
    """
    根据距离最近原则，对 UAVs 和 targets 进行一一匹配
    (使用匈牙利算法最小化总距离)

    :param uavs_pos: UAVs 的位置 (I * 3)
    :param targets_pos: targets 的位置 (J * 3)
    :return: UAVs 和 targets 的匹配矩阵 (I * J)
    """
    num_uavs = uavs_pos.shape[0]
    num_targets = targets_pos.shape[0]
    uavs_targets_matched_matrix = np.zeros((num_uavs, num_targets))
    
    # 计算距离矩阵
    diff = uavs_pos[:, np.newaxis, :] - targets_pos[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    # 使用匈牙利算法求解最小权匹配 (距离之和最小)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # 设置匹配矩阵
    uavs_targets_matched_matrix[row_ind, col_ind] = 1
    
    return uavs_targets_matched_matrix


def random_choose_matched_matrix(uavs_pos, cus_pos):
    """
    随机选择 UAVs 和 CUs 的匹配矩阵

    :param uavs_pos: UAVs 的位置 (I * 3)
    :param cus_pos: CUs 的位置 (J * 3)
    :return: UAVs 和 CUs 的匹配矩阵 (I * J)
    """
    num_uavs = uavs_pos.shape[0]
    num_cus = cus_pos.shape[0]
    uavs_cus_matched_matrix = np.zeros((num_uavs, num_cus))
    
    # 生成随机匹配索引
    u_indices = np.arange(num_uavs)
    c_indices = np.arange(num_cus)
    np.random.shuffle(u_indices)
    np.random.shuffle(c_indices)
    
    # 匹配数量取两者的最小值，确保满足 1-to-1 约束
    num_matches = min(num_uavs, num_cus)
    
    for i in range(num_matches):
        # 每个 UAV 索引 u_indices[i] 只会出现一次 -> 每一行最多有一个 1
        # 每个 CU 索引 c_indices[i] 只会出现一次 -> 每一列最多有一个 1
        uavs_cus_matched_matrix[u_indices[i], c_indices[i]] = 1
        
    return uavs_cus_matched_matrix


def extract_matched_sensing_channel(uavs_targets_matched_matrix, uavs_2_targets_channels):
    """
    根据匹配矩阵提取对应的信道响应矩阵

    :param uavs_targets_matched_matrix: UAVs 和 targets 的匹配矩阵 (I * J)
    :param uavs_2_targets_channels: 所有 UAVs 到所有 targets 的信道响应矩阵 (I * J * N * N)
    :return: 匹配后的信道响应矩阵 (I * N * N)
    """
    # 找到匹配的索引
    # row_indices: UAV 索引, col_indices: Target 索引
    row_indices, col_indices = np.where(uavs_targets_matched_matrix == 1)
    
    # 按照 UAV 索引排序 (确保顺序对应 UAV 0, UAV 1, ...)
    sorted_args = np.argsort(row_indices)
    row_indices = row_indices[sorted_args]
    col_indices = col_indices[sorted_args]
    
    # 提取信道
    # matched_uav_sensing_channel: (I, N, N)
    matched_uav_sensing_channel = uavs_2_targets_channels[row_indices, col_indices, :, :]
    
    return matched_uav_sensing_channel


def compute_uav_pos_cur(args, uavs_pos_pre):
    """
    根据前一时刻位置和当前时刻速度计算当前时刻位置

    :param uav_pos_pre: UAVs 前一时刻位置 (I * 3)
    :return: UAVs 当前时刻位置 (I * 3)
    """
    # 根据速度约束生成当前位置
    # 随机生成速度向量，模长在 [min_speed, max_speed] 之间
    # 速度向量 v = [vx, vy, vz], |v| = speed
    # q(t+1) = q(t) + v * tau
    
    # 1. 速度大小
    uav_speed_magnitude = np.random.uniform(args.uav_min_speed, args.uav_max_speed, args.uavs_num)

    # 2. 只生成 xy 方向
    random_direction_xy = np.random.randn(args.uavs_num, 2)
    random_direction_xy /= np.linalg.norm(random_direction_xy, axis=1, keepdims=True)

    # 3. 位移（只在 xy）
    displacement_xy = random_direction_xy * uav_speed_magnitude[:, np.newaxis] * args.time_slot_duration

    # 4. 拼成 3D 位移（z=0）
    displacement = np.zeros_like(uavs_pos_pre)
    displacement[:, :2] = displacement_xy

    # 5. 更新位置（z 不变）
    uavs_pos_cur = uavs_pos_pre + displacement

    return uavs_pos_cur


def initialize_uav_beams(args, matched_uav_sensing_channel, uavs_2_bs_channels):
    hat_uav_sen_beams = []
    hat_uav_off_beams = []

    # ==============================
    # 初始化 UAV 感知波束
    # ==============================
    for i in range(args.uavs_num):
        H_i = matched_uav_sensing_channel[i]  # (N,N)

        w_init = compute_largest_eigenvector(H_i)

        W_init = np.outer(w_init, np.conj(w_init))

        W_init = args.uav_max_power * W_init / np.trace(W_init)

        hat_uav_sen_beams.append(W_init)

    # ==============================
    # 初始化 UAV 卸载感知任务波束
    # ==============================
    for i in range(args.uavs_num):
        h_i = uavs_2_bs_channels[i, 0, :]  # (N,)

        H_i = np.outer(h_i, np.conj(h_i))

        w_init = compute_largest_eigenvector(H_i)

        W_init = np.outer(w_init, np.conj(w_init))

        W_init = args.uav_max_power * W_init / np.trace(W_init)

        hat_uav_off_beams.append(W_init)

    return hat_uav_sen_beams, hat_uav_off_beams


def update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                          rank1_sen_proj_mats, rank1_off_proj_mats,
                                          cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                          rank1_const_terms):
    cur_rho = float(cur_penalty_factor.value)
    for i in range(args.uavs_num):
        v_w = compute_largest_eigenvector(hat_uav_sen_beams[i].value)
        v_b = compute_largest_eigenvector(hat_uav_off_beams[i].value)
        v_w_matrix = np.outer(v_w, np.conj(v_w))
        v_b_matrix = np.outer(v_b, np.conj(v_b))
        v_w_matrix = (v_w_matrix + v_w_matrix.conj().T) / 2
        v_b_matrix = (v_b_matrix + v_b_matrix.conj().T) / 2
        rank1_sen_proj_mats[i].value = v_w_matrix
        rank1_off_proj_mats[i].value = v_b_matrix
        rank1_sen_scaled_proj_mats[i].value = cur_rho * v_w_matrix
        rank1_off_scaled_proj_mats[i].value = cur_rho * v_b_matrix
        rank1_const_terms.value[i] = float(
            cur_rho * (
                - np.linalg.norm(hat_uav_sen_beams[i].value, 2)
                - np.linalg.norm(hat_uav_off_beams[i].value, 2)
                + np.real(np.trace(v_w_matrix @ hat_uav_sen_beams[i].value))
                + np.real(np.trace(v_b_matrix @ hat_uav_off_beams[i].value))
            )
        )


def update_obj5_linearization_parameters(args, hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                         obj5_coef_z, obj5_coef_f, obj5_const_terms):
    alpha = args.omega_weight_1 * args.kappa * args.bs_cycles_per_bit * args.uav_sen_duration
    scale_4 = args.freq_scale ** 4
    for i in range(args.uavs_num):
        hat_z = float(hat_auxiliary_variable_z.value[i])
        hat_f = float(hat_bs_2_uav_freqs_norm.value[i])
        obj5_coef_z.value[i] = -alpha * hat_z
        obj5_coef_f.value[i] = -alpha * 2.0 * (hat_f ** 3) * scale_4
        obj5_const_terms.value[i] = alpha * (0.5 * (hat_z ** 2) + 1.5 * (hat_f ** 4) * scale_4)


def update_constraint_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                               matched_uav_sensing_channel, uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels,
                                               uavs_cus_matched_matrix, uavs_off_duration, cus_off_power,
                                               c44_const_terms, c44_grad_mats,
                                               c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b):
    I = args.uavs_num
    J = args.cus_num
    N = args.antenna_nums
    B = args.bandwidth
    sigma_2 = dbm_2_watt(args.noise_power_density_dbm) * B
    D_sen = args.uav_sen_duration
    xi1 = args.radar_duty_ratio / (2.0 * args.radar_impulse_duration)
    xi2 = (2.0 * args.var_range_fluctuation
           * (args.radar_spectrum_shape ** 2)
           * (args.bandwidth ** 3)
           * args.radar_impulse_duration)

    Delta_list = []
    for i in range(I):
        Delta_i = sigma_2 * np.eye(N, dtype=complex)
        for j in range(J):
            eta_ij = uavs_cus_matched_matrix[i, j]
            if eta_ij > 0:
                h_ij = uavs_2_cus_channels[i, j, :]
                Delta_i = Delta_i + eta_ij * cus_off_power[j] * np.outer(h_ij, np.conj(h_ij))
        Delta_list.append(Delta_i)

    H_ui_BS_list = []
    for i in range(I):
        h_i = uavs_2_bs_channels[i, 0, :]
        H_ui_BS_list.append(np.outer(h_i, np.conj(h_i)))

    for i in range(I):
        A_i = matched_uav_sensing_channel[i]
        Delta_i = Delta_list[i]
        hat_W_i = hat_uav_sen_beams[i].value
        Psi_i_n = Delta_i + xi2 * (A_i @ hat_W_i @ A_i.conj().T)
        _, logdet_Delta_i = np.linalg.slogdet(Delta_i)
        _, logdet_Psi_i_n = np.linalg.slogdet(Psi_i_n)
        AH_Psiinv_A = A_i.conj().T @ np.linalg.inv(Psi_i_n) @ A_i
        grad_mat = (xi1 * xi2 / np.log(2)) * AH_Psiinv_A
        c44_grad_mats[i].value = (grad_mat + grad_mat.conj().T) / 2
        tr_grad_W = np.real(np.trace(c44_grad_mats[i].value @ hat_W_i))
        c44_const_terms.value[i] = float(-xi1 * (logdet_Delta_i / np.log(2)) + xi1 * (logdet_Psi_i_n / np.log(2))) - float(tr_grad_W)

    for j in range(J):
        sum_eta_ij_D_sen = D_sen * float(np.sum(uavs_cus_matched_matrix[:, j]))
        sum_eta_ij_D_off = float(np.sum(uavs_cus_matched_matrix[:, j] * uavs_off_duration))
        Psi_j1_n = sigma_2
        Psi_j2_n = sigma_2
        for i in range(I):
            eta_ij = uavs_cus_matched_matrix[i, j]
            if eta_ij > 0:
                Psi_j1_n += eta_ij * float(np.real(np.trace(H_ui_BS_list[i] @ hat_uav_sen_beams[i].value)))
                Psi_j2_n += eta_ij * float(np.real(np.trace(H_ui_BS_list[i] @ hat_uav_off_beams[i].value)))
        Psi_j1_n_safe = float(np.real(Psi_j1_n))
        Psi_j2_n_safe = float(np.real(Psi_j2_n))
        c45_term4_const.value[j] = sum_eta_ij_D_sen * B * np.log2(Psi_j1_n_safe)
        c45_term5_const.value[j] = sum_eta_ij_D_off * B * np.log2(Psi_j2_n_safe)

        coef_w_j = B * sum_eta_ij_D_sen / (np.log(2) * Psi_j1_n_safe) if sum_eta_ij_D_sen > 0 else 0.0
        coef_b_j = B * sum_eta_ij_D_off / (np.log(2) * Psi_j2_n_safe) if sum_eta_ij_D_off > 0 else 0.0
        tr_hat_w_sum = 0.0
        tr_hat_b_sum = 0.0
        for i in range(I):
            eta_ij = uavs_cus_matched_matrix[i, j]
            if eta_ij > 0:
                tr_hat_w_sum += eta_ij * float(np.real(np.trace(H_ui_BS_list[i] @ hat_uav_sen_beams[i].value)))
                tr_hat_b_sum += eta_ij * float(np.real(np.trace(H_ui_BS_list[i] @ hat_uav_off_beams[i].value)))
        c45_coef_w.value[j] = coef_w_j
        c45_coef_b.value[j] = coef_b_j
        c45_const_w.value[j] = -coef_w_j * tr_hat_w_sum
        c45_const_b.value[j] = -coef_b_j * tr_hat_b_sum


def compute_obj_fun(args, var_uavs_sen_beam, var_uavs_off_beam, var_bs_2_uav_freqs_norm, var_auxiliary_variable_z, var_cus_off_duration, var_t,
                    hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm, hat_uav_sen_beams, hat_uav_off_beams,
                    rank1_sen_proj_mats, rank1_off_proj_mats, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats, rank1_const_terms,
                    obj5_coef_z, obj5_coef_f, obj5_const_terms, cus_entertaining_task_size,
                    uavs_off_duration, cus_off_power, uavs_pos_pre, uavs_pos_cur, cur_penalty_factor, use_penalty_rank1):

    obj_fun = 0

    omega_weight_1 = args.omega_weight_1
    omega_weight_2 = args.omega_weight_2
    omega_weight_3 = args.omega_weight_3
    kappa = args.kappa

    cu_max_delay = args.cu_max_delay

    # ==============================
    # UAV 飞行能耗
    # ==============================
    uavs_pos_cur = np.array(uavs_pos_cur, dtype=float)
    uavs_pos_pre = np.array(uavs_pos_pre, dtype=float)
    # 1. 按行求范数，得到 shape 为 (args.uavs_num,) 的距离数组
    uav_dist_diff = np.linalg.norm(uavs_pos_cur - uavs_pos_pre, axis=1)
    # 2. 引入极小值 epsilon 防止悬停时除以 0
    epsilon = 1e-8
    uav_dist_diff = np.maximum(uav_dist_diff, epsilon)
    uav_fly_energy = ( (args.uav_c1 * (uav_dist_diff ** 3) / (args.time_slot_duration ** 2)) +
                      (args.uav_c2 * (args.time_slot_duration ** 2) / uav_dist_diff))

    # ==============================
    # 第一项
    # ==============================
    for i in range(args.uavs_num):
        obj_fun += omega_weight_1 * kappa * (
            args.bs_cycles_per_bit *
            args.uav_sen_duration *
            (cp.square(var_t[i]) * (args.freq_scale ** 4))
        ) / 2

    # ==============================
    # 第二项
    # ==============================
    for j in range(args.cus_num):
        obj_fun += omega_weight_1 * kappa * ((args.bs_cycles_per_bit * cus_entertaining_task_size[j]) ** 3) * \
                   cp.power(cu_max_delay - var_cus_off_duration[j], -2)

    # ==============================
    # 第三项
    # ==============================
    for i in range(args.uavs_num):
        obj_fun += omega_weight_2 * (
            args.uav_sen_duration * cp.real(cp.trace(var_uavs_sen_beam[i]))
            + uavs_off_duration[i] * cp.real(cp.trace(var_uavs_off_beam[i]))
            + uav_fly_energy[i]
        )

    # ==============================
    # 第四项
    # ==============================
    for j in range(args.cus_num):
        obj_fun += omega_weight_3 * (
            var_cus_off_duration[j] *
            cus_off_power[j]
        )

    # ==============================
    # 第五项 CCCP 线性化
    # ==============================
    for i in range(args.uavs_num):
        obj_fun += (
            obj5_coef_z[i] * var_auxiliary_variable_z[i]
            + obj5_coef_f[i] * var_bs_2_uav_freqs_norm[i]
            + obj5_const_terms[i]
        )
    
    # 采用 rank-1 惩罚项
    if use_penalty_rank1:
        # ==============================
        # 第六项 rank-1 penalty
        # ==============================
        for i in range(args.uavs_num):
            v_w_scaled_matrix = rank1_sen_scaled_proj_mats[i]
            v_b_scaled_matrix = rank1_off_scaled_proj_mats[i]

            rank1_penalty_gap = (
                cur_penalty_factor * (
                    cp.real(cp.trace(var_uavs_sen_beam[i]))
                    + cp.real(cp.trace(var_uavs_off_beam[i]))
                )
                - cp.real(cp.trace(v_w_scaled_matrix @ var_uavs_sen_beam[i]))
                - cp.real(cp.trace(v_b_scaled_matrix @ var_uavs_off_beam[i]))
                + rank1_const_terms[i]
            )
            # obj_fun += cp.pos(rank1_penalty_gap)
            obj_fun += rank1_penalty_gap
    
    return obj_fun


def compute_original_obj_fun_value(args, cur_uavs_sen_beams, cur_uavs_off_beams, cur_bs_2_uav_freqs_norm, cur_auxiliary_variable_z,
                                   cur_cus_off_duration, cus_entertaining_task_size, uavs_off_duration, cus_off_power, uavs_pos_pre,
                                   uavs_pos_cur, cur_penalty_factor, use_penalty_rank1):
    """
    计算原始目标函数值
    """
    obj_fun = 0.0
    omega_weight_1 = args.omega_weight_1
    omega_weight_2 = args.omega_weight_2
    omega_weight_3 = args.omega_weight_3
    kappa = args.kappa
    cu_max_delay = args.cu_max_delay
    uavs_pos_cur = np.array(uavs_pos_cur, dtype=float)
    uavs_pos_pre = np.array(uavs_pos_pre, dtype=float)
    uav_dist_diff = np.linalg.norm(uavs_pos_cur - uavs_pos_pre, axis=1)
    uav_dist_diff = np.maximum(uav_dist_diff, 1e-8)
    uav_fly_energy = ((args.uav_c1 * (uav_dist_diff ** 3) / (args.time_slot_duration ** 2)) +
                      (args.uav_c2 * (args.time_slot_duration ** 2) / uav_dist_diff))
    for i in range(args.uavs_num):
        f_ui = cur_bs_2_uav_freqs_norm[i] * args.freq_scale
        obj_fun += omega_weight_1 * kappa * args.bs_cycles_per_bit * args.uav_sen_duration * cur_auxiliary_variable_z[i] * (f_ui ** 2)
    for j in range(args.cus_num):
        denom = max(cu_max_delay - cur_cus_off_duration[j], 1e-8)
        obj_fun += omega_weight_1 * kappa * ((args.bs_cycles_per_bit * cus_entertaining_task_size[j]) ** 3) / (denom ** 2)
    for i in range(args.uavs_num):
        obj_fun += omega_weight_2 * (
            args.uav_sen_duration * np.real(np.trace(cur_uavs_sen_beams[i]))
            + uavs_off_duration[i] * np.real(np.trace(cur_uavs_off_beams[i]))
            + uav_fly_energy[i]
        )
    for j in range(args.cus_num):
        obj_fun += omega_weight_3 * cur_cus_off_duration[j] * cus_off_power[j]
    if use_penalty_rank1:
        for i in range(args.uavs_num):
            w_gap = np.real(np.trace(cur_uavs_sen_beams[i])) - np.linalg.norm(cur_uavs_sen_beams[i], 2)
            b_gap = np.real(np.trace(cur_uavs_off_beams[i])) - np.linalg.norm(cur_uavs_off_beams[i], 2)
            # NOTE - 防止因为数值误差导致 gap 为负数
            obj_fun += cur_penalty_factor * (max(float(w_gap), 0.0) + max(float(b_gap), 0.0)) 
    return float(np.real(obj_fun))


def compute_pure_energy_value(args, cur_uavs_sen_beams, cur_uavs_off_beams, cur_bs_2_uav_freqs_norm, cur_auxiliary_variable_z,
                              cur_cus_off_duration, cus_entertaining_task_size, uavs_off_duration, cus_off_power, uavs_pos_pre,
                              uavs_pos_cur):
    """
    计算纯能量值，即不考虑罚因子的目标函数值。
    """
    return compute_original_obj_fun_value(
        args=args,
        cur_uavs_sen_beams=cur_uavs_sen_beams,
        cur_uavs_off_beams=cur_uavs_off_beams,
        cur_bs_2_uav_freqs_norm=cur_bs_2_uav_freqs_norm,
        cur_auxiliary_variable_z=cur_auxiliary_variable_z,
        cur_cus_off_duration=cur_cus_off_duration,
        cus_entertaining_task_size=cus_entertaining_task_size,
        uavs_off_duration=uavs_off_duration,
        cus_off_power=cus_off_power,
        uavs_pos_pre=uavs_pos_pre,
        uavs_pos_cur=uavs_pos_cur,
        cur_penalty_factor=0.0,
        use_penalty_rank1=False,
    )


def define_constraint(args, var_uavs_sen_beam, var_uavs_off_beam, var_bs_2_uav_freqs_norm, var_auxiliary_variable_z, var_cus_off_duration, var_t,
                      hat_uav_sen_beams, hat_uav_off_beams, cus_entertaining_task_size, uavs_off_duration, cus_off_power, matched_uav_sensing_channel,
                      uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, uavs_cus_matched_matrix,
                      c44_const_terms, c44_grad_mats, c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b):
    """
    定义问题 P6 的所有约束。
    每条约束均附有与论文公式一一对应的逐行注释，方便核对。
    """
    constraints = []

    # ── 基础标量参数 ──────────────────────────────────────────────────────────
    I        = args.uavs_num
    J        = args.cus_num
    N        = args.antenna_nums
    B        = args.bandwidth                                  # 带宽 (Hz)
    sigma_2  = dbm_2_watt(args.noise_power_density_dbm) * B   # 噪声功率 σ²
    D_sen    = args.uav_sen_duration                           # D̄_sen
    D_max_ui = args.uav_max_delay                              # D^max_{u_i}
    D_max_cj = args.cu_max_delay                               # D^max_{c_j}
    C_bit    = args.bs_cycles_per_bit                          # C_{u_i} = C_{c_j}
    P_max    = args.uav_max_power                              # P^max_UAV (W)，已是瓦特，无需再转换
    F_max    = args.bs_max_freq                                # F^max
    eps_snr  = db_2_watt(args.sen_sinr)                        # 感知 SINR 门限 ε

    # ── 雷达参数 ─────────────────────────────────────────────────────────────
    # ξ1 = δ / (2ν)
    xi1 = args.radar_duty_ratio / (2.0 * args.radar_impulse_duration)
    # ξ2 = 2 σ²_pre γ² B³ ν
    xi2 = (2.0 * args.var_range_fluctuation
           * (args.radar_spectrum_shape ** 2)
           * (args.bandwidth ** 3)
           * args.radar_impulse_duration)

    # ── 预计算 Δ_i ────────────────────────────────────────────
    #
    # Δ_i = Σ_{j=1}^{J} η_{i,j} · p_j · h_{u_i,c_j} · h^H_{u_i,c_j} + σ² I_N
    #
    # uavs_2_cus_channels shape: (I, J, N)，其中第三维为 UAV 天线个数 N
    Delta_list = []
    for i in range(I):
        Delta_i = sigma_2 * np.eye(N, dtype=complex)           # σ² I_N
        for j in range(J):
            eta_ij = uavs_cus_matched_matrix[i, j]             # η_{i,j}
            if eta_ij > 0:
                h_ij   = uavs_2_cus_channels[i, j, :]          # h_{u_i,c_j}，shape (N,)
                Delta_i = (Delta_i
                           + eta_ij                             # η_{i,j}
                           * cus_off_power[j]                  # · p_j
                           * np.outer(h_ij, np.conj(h_ij)))    # · h_{u_i,c_j} h^H_{u_i,c_j}
        Delta_list.append(Delta_i)

    # ── 预计算 H_{u_i,BS} = h_{u_i,BS} h^H_{u_i,BS} ─────────────────────────
    # uavs_2_bs_channels shape: (I, 1, N)
    H_ui_BS_list = []
    for i in range(I):
        h_i = uavs_2_bs_channels[i, 0, :]                      # h_{u_i,BS}，shape (N,)
        H_ui_BS_list.append(np.outer(h_i, np.conj(h_i)))       # H_{u_i,BS}，shape (N, N)

    # ── 预计算 |h_{c_j,BS}|² ─────────────────────────────────────────────────
    # cus_2_bs_channels shape: (J, 1)
    h_cj_BS_sq = np.array([abs(cus_2_bs_channels[j, 0]) ** 2 for j in range(J)])

    # =========================================================================
    # 约束 (4.5)：D̄_sen + D^off_{u_i} - Σ_{j=1}^{J} η_{i,j} D^off_{c_j} ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(
            D_sen                                               # D̄_sen
            + uavs_off_duration[i]                             # + D^off_{u_i}
            - cp.sum(                                          # - Σ_{j=1}^{J}
                uavs_cus_matched_matrix[i, :] *                #   η_{i,j}
                var_cus_off_duration                           #   · D^off_{c_j}
            )
            <= 0
        )

    # =========================================================================
    # 约束 (4.12)：
    #   D̄_sen · f_{u_i} + D^off_{u_i} · f_{u_i} - D^max_{u_i} · f_{u_i}
    #   + C_{u_i} · D̄_sen · z_i ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(
            D_sen * var_bs_2_uav_freqs_norm[i] * args.freq_scale                      # D̄_sen · f_{u_i}
            + uavs_off_duration[i] * var_bs_2_uav_freqs_norm[i] * args.freq_scale    # + D^off_{u_i} · f_{u_i}
            - D_max_ui * var_bs_2_uav_freqs_norm[i] * args.freq_scale                 # - D^max_{u_i} · f_{u_i}
            + C_bit * D_sen * var_auxiliary_variable_z[i]                              # + C_{u_i} · D̄_sen · z_i
            <= 0
        )

    # =========================================================================
    # 约束 (4.23)：W_i(t) ⪰ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(var_uavs_sen_beam[i] >> 0)

    # =========================================================================
    # 约束 (4.25)：B_i(t) ⪰ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(var_uavs_off_beam[i] >> 0)

    # =========================================================================
    # 约束 (4.27)：Tr(W_i) - P^max_UAV ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(
            cp.real(cp.trace(var_uavs_sen_beam[i]))            # Tr(W_i)
            - P_max                                             # - P^max_UAV
            <= 0
        )

    # =========================================================================
    # 约束 (4.28)：ε - Tr(A^H(θ_i) Δ^{-1}_i A(θ_i) W_i) ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        A_i         = matched_uav_sensing_channel[i]            # A(θ_i)，shape (N, N)
        Delta_i     = Delta_list[i]                             # Δ_i，shape (N, N)
        Delta_i_inv = np.linalg.inv(Delta_i)                   # Δ^{-1}_i
        AH_Dinv_A   = A_i.conj().T @ Delta_i_inv @ A_i         # A^H Δ^{-1} A，shape (N, N)
        constraints.append(
            eps_snr                                             # ε
            - cp.real(cp.trace(                                 # - Tr(A^H Δ^{-1} A · W_i)
                AH_Dinv_A @ var_uavs_sen_beam[i]
            ))
            <= 0
        )

    # =========================================================================
    # 约束 (4.29)：Tr(B_i) - P^max_UAV ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        constraints.append(
            cp.real(cp.trace(var_uavs_off_beam[i]))            # Tr(B_i)
            - P_max                                             # - P^max_UAV
            <= 0
        )

    # =========================================================================
    # 约束 (4.32)：
    #   D̄_sen · z_i
    #   - D^off_{u_i} · B · log2( Tr(H_{u_i,BS} B_i) + Σ_j η_{i,j} p_j |h_{c_j,BS}|² + σ² )
    #   + D^off_{u_i} · B · log2( Σ_j η_{i,j} p_j |h_{c_j,BS}|² + σ² )
    #   ≤ 0，∀u_i ∈ U
    # =========================================================================
    for i in range(I):
        H_i = H_ui_BS_list[i]                                  # H_{u_i,BS}，shape (N, N)

        # Σ_j η_{i,j} · p_j · |h_{c_j,BS}|² + σ²（常数）
        interf_plus_noise = float(np.sum(uavs_cus_matched_matrix[i, :] * cus_off_power * h_cj_BS_sq)) + sigma_2

        constraints.append(
            D_sen * var_auxiliary_variable_z[i]                
            - uavs_off_duration[i] * B * cp.log(cp.real(cp.trace(H_i @ var_uavs_off_beam[i]))+ interf_plus_noise) / np.log(2)
            + uavs_off_duration[i] * B * np.log2(interf_plus_noise)                     
            <= 0
        )

    # =========================================================================
    # 约束 (4.39)：D^off_{c_j} - D^max_{c_j} < 0，∀c_j ∈ C
    #   保证 f_{c_j} = C_{c_j} L_j / (D^max_{c_j} - D^off_{c_j}) 始终为正
    # =========================================================================
    eps_strict = 1e-10
    for j in range(J):
        constraints.append(
            var_cus_off_duration[j]                            # D^off_{c_j}
            - D_max_cj + eps_strict                            # - D^max_{c_j}
            <= 0
        )

    # =========================================================================
    # 约束 (4.40)：
    #   Σ_{u_i} f_{u_i} + Σ_{c_j} C_{c_j} L_j / (D^max_{c_j} - D^off_{c_j}) - F^max ≤ 0
    # =========================================================================
    sum_freq = cp.sum(var_bs_2_uav_freqs_norm) * args.freq_scale                      # Σ_{u_i} f_{u_i}
    for j in range(J):
        sum_freq = (
            sum_freq
            + C_bit * cus_entertaining_task_size[j] * cp.inv_pos(D_max_cj - var_cus_off_duration[j]) 
        )
    constraints.append(sum_freq - F_max <= 0)

    # =========================================================================
    # 约束 (4.44)：对约束 (4.30) 进行 CCCP 线性化（第 n+1 次迭代）
    #
    #   - ξ1 · log2det(Δ_i)
    #   - z_i
    #   + ξ1 · log2det(Ψ^(n)_i)
    #   + ξ1 ξ2 / ln2 · Tr( A^H(θ_i) · (Ψ^(n)_i)^{-1} · A(θ_i) · (W_i - W^(n)_i) )
    #   ≤ 0，∀u_i ∈ U
    #
    #   其中 Ψ^(n)_i = Δ_i + ξ2 · A(θ_i) · W^(n)_i · A^H(θ_i)
    # =========================================================================
    for i in range(I):
        lhs_44 = (
            c44_const_terms[i]
            - var_auxiliary_variable_z[i]
            + cp.real(cp.trace(c44_grad_mats[i] @ var_uavs_sen_beam[i]))
        )
        constraints.append(lhs_44 <= 0)

    # =========================================================================
    # 约束 (4.45)：对约束 (4.31) 进行 CCCP 线性化（第 n+1 次迭代）
    #
    #   L_j
    #   - Σ_i η_{i,j} D̄_sen · B · log2( p_j|h_{c_j,BS}|² + Σ_i η_{i,j} Tr(H_{u_i,BS} W_i) + σ² )
    #   - Σ_i η_{i,j} D^off_{u_i} · B · log2( p_j|h_{c_j,BS}|² + Σ_i η_{i,j} Tr(H_{u_i,BS} B_i) + σ² )
    #   - ( D^off_{c_j} - Σ_i η_{i,j} D̄_sen - Σ_i η_{i,j} D^off_{u_i} )
    #     · B · log2( 1 + p_j |h_{c_j,BS}|² / σ² )
    #   + Σ_i η_{i,j} D̄_sen · B · log2( Ψ^(n)_{j,1} )
    #   + Σ_i η_{i,j} D^off_{u_i} · B · log2( Ψ^(n)_{j,2} )
    #   + B · Σ_i η_{i,j} D̄_sen / ( ln2 · Ψ^(n)_{j,1} )
    #     · Σ_i η_{i,j} Tr( H_{u_i,BS} (W_i - W^(n)_i) )
    #   + B · Σ_i η_{i,j} D^off_{u_i} / ( ln2 · Ψ^(n)_{j,2} )
    #     · Σ_i η_{i,j} Tr( H_{u_i,BS} (B_i - B^(n)_i) )
    #   ≤ 0，∀c_j ∈ C
    #
    #   其中 Ψ^(n)_{j,1} = Σ_i η_{i,j} Tr(H_{u_i,BS} W^(n)_i) + σ²
    #        Ψ^(n)_{j,2} = Σ_i η_{i,j} Tr(H_{u_i,BS} B^(n)_i) + σ²
    # =========================================================================
    for j in range(J):
        p_j     = cus_off_power[j]                              # p_j
        h_cj_sq = h_cj_BS_sq[j]                                # |h_{c_j,BS}|²

        # Σ_i η_{i,j} D̄_sen（常数系数）
        sum_eta_ij_D_sen = D_sen * float(np.sum(uavs_cus_matched_matrix[:, j]))
        # Σ_i η_{i,j} D^off_{u_i}（常数系数）
        sum_eta_ij_D_off = float(np.sum(uavs_cus_matched_matrix[:, j] * uavs_off_duration))

        # log2(1 + p_j |h_{c_j,BS}|² / σ²)（常数）
        log2_cu_snr = np.log2(1.0 + p_j * h_cj_sq / sigma_2)

        # ── 构建含优化变量的 CVXPY 表达式 ────────────────────────────────────
        # Σ_i η_{i,j} Tr(H_{u_i,BS} W_i)（含优化变量，用于 log 的参数）
        sum_eta_tr_H_W = 0.0
        for i in range(I):
            eta_ij = uavs_cus_matched_matrix[i, j]
            if eta_ij > 0:
                sum_eta_tr_H_W = (
                    sum_eta_tr_H_W
                    + eta_ij                                     # η_{i,j}
                    * cp.real(cp.trace(                          # · Tr(H_{u_i,BS} W_i)
                        H_ui_BS_list[i] @ var_uavs_sen_beam[i]
                    ))
                )

        # Σ_i η_{i,j} Tr(H_{u_i,BS} B_i)（含优化变量，用于 log 的参数）
        sum_eta_tr_H_B = 0.0
        for i in range(I):
            eta_ij = uavs_cus_matched_matrix[i, j]
            if eta_ij > 0:
                sum_eta_tr_H_B = (
                    sum_eta_tr_H_B
                    + eta_ij                                     # η_{i,j}
                    * cp.real(cp.trace(                          # · Tr(H_{u_i,BS} B_i)
                        H_ui_BS_list[i] @ var_uavs_off_beam[i]
                    ))
                )

        # ── 逐项构建约束 (4.39) ───────────────────────────────────────────────

        # 项①：- Σ_i η_{i,j} D̄_sen · B · log2( p_j|h|² + Σ_i η Tr(H W_i) + σ² )
        if sum_eta_ij_D_sen > 0:
            term_1 = (
                - sum_eta_ij_D_sen * B                          # - Σ_i η_{i,j} D̄_sen · B
                * cp.log(                                        # · log(
                    p_j * h_cj_sq                               #     p_j |h_{c_j,BS}|²
                    + sum_eta_tr_H_W                            #   + Σ_i η Tr(H_{u_i,BS} W_i)
                    + sigma_2                                    #   + σ²
                ) / np.log(2)
            )
        else:
            term_1 = 0.0

        # 项②：- Σ_i η_{i,j} D^off_{u_i} · B · log2( p_j|h|² + Σ_i η Tr(H B_i) + σ² )
        if sum_eta_ij_D_off > 0:
            term_2 = (
                - sum_eta_ij_D_off * B                          # - Σ_i η_{i,j} D^off_{u_i} · B
                * cp.log(                                        # · log(
                    p_j * h_cj_sq                               #     p_j |h_{c_j,BS}|²
                    + sum_eta_tr_H_B                            #   + Σ_i η Tr(H_{u_i,BS} B_i)
                    + sigma_2                                    #   + σ²
                ) / np.log(2)
            )
        else:
            term_2 = 0.0

        # 项③：- ( D^off_{c_j} - Σ_i η D̄_sen - Σ_i η D^off_{u_i} ) · B · log2(1 + p_j|h|²/σ²)
        term_3 = (
            - (var_cus_off_duration[j]                          # -( D^off_{c_j}
               - sum_eta_ij_D_sen                               #    - Σ_i η_{i,j} D̄_sen
               - sum_eta_ij_D_off)                              #    - Σ_i η_{i,j} D^off_{u_i} )
            * B * log2_cu_snr                                   # · B · log2(1 + p_j|h|²/σ²)
        )

        term_4 = c45_term4_const[j]
        term_5 = c45_term5_const[j]

        if sum_eta_ij_D_sen > 0 and not isinstance(sum_eta_tr_H_W, float):
            term_6 = c45_coef_w[j] * sum_eta_tr_H_W + c45_const_w[j]
        else:
            term_6 = 0.0

        if sum_eta_ij_D_off > 0 and not isinstance(sum_eta_tr_H_B, float):
            term_7 = c45_coef_b[j] * sum_eta_tr_H_B + c45_const_b[j]
        else:
            term_7 = 0.0

        constraints.append(
            cus_entertaining_task_size[j]                       # L_j
            + term_1                                            # 项①
            + term_2                                            # 项②
            + term_3                                            # 项③
            + term_4                                            # 项④
            + term_5                                            # 项⑤
            + term_6                                            # 项⑥
            + term_7                                            # 项⑦
            <= 0
        )

    # 追加为了满足 DCP 而引入的变量 t = x + y^{2}
    # var_t 为 epigraph 上镜图
    # 2026-03-17 修改： var_t 成为归一化变量
    for i in range(args.uavs_num):
        constraints.append(
            (var_auxiliary_variable_z[i] / (args.freq_scale ** 2)) + cp.square(var_bs_2_uav_freqs_norm[i]) - var_t[i] <= 0
        )
    
    # 定义变量大于 0 
    for i in range(I):
        constraints += [
            var_auxiliary_variable_z[i] >= 0,
            var_bs_2_uav_freqs_norm[i] >= 0,
            var_t[i] >= 0,
        ]

    for j in range(J):
        constraints += [
            var_cus_off_duration[j] >= 0
        ]

    return constraints


def parse_group_list(group_text):
    """
    解析约束组列表字符串，返回约束组名称列表。
    约束组名称列表中的每个元素为一个字符串，代表一个约束组。
    约束组名称列表中的元素顺序与约束组在问题 P6 中的顺序一致。
    """
    if group_text is None:
        return None
    group_text = str(group_text).strip()
    if group_text == "":
        return None
    # 以","进行输入故以逗号进行分割
    return [item.strip() for item in group_text.split(",") if item.strip()]


def build_constraint_map(args, constraints):
    I = args.uavs_num
    J = args.cus_num
    idx = 0
    constraint_map = {}
    constraint_map["4.5"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.12"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.23"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.25"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.27"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.28"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.29"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.32"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.39"] = constraints[idx:idx + J]
    idx += J
    constraint_map["4.40"] = constraints[idx:idx + 1]
    idx += 1
    constraint_map["4.44"] = constraints[idx:idx + I]
    idx += I
    constraint_map["4.45"] = constraints[idx:idx + J]
    idx += J
    constraint_map["auxiliary_t"] = constraints[idx:idx + I]
    idx += I
    constraint_map["var"] = constraints[idx:idx + I * 3 + J]
    return constraint_map


def select_constraints(constraint_map, include_groups=None, exclude_groups=None):
    all_groups = list(constraint_map.keys())
    unknown_include = [] if include_groups is None else [g for g in include_groups if g not in constraint_map]
    unknown_exclude = [] if exclude_groups is None else [g for g in exclude_groups if g not in constraint_map]
    if len(unknown_include) > 0 or len(unknown_exclude) > 0:
        raise ValueError(f"未知约束组 include={unknown_include}, exclude={unknown_exclude}, 可选组={all_groups}")

    active_groups = all_groups if include_groups is None else [g for g in all_groups if g in include_groups]
    if exclude_groups is not None:
        active_groups = [g for g in active_groups if g not in exclude_groups]

    selected_constraints = []
    for group in active_groups:
        selected_constraints.extend(constraint_map[group])
    return selected_constraints, active_groups


def probe_group_feasibility(constraint_map, active_groups, solver):
    """
    问题的可行性检验
    
    :param constraint_map: 约束组 map 集合，key 为约束组名称，value 为约束列表
    :param active_groups: 活动约束组名称列表
    :param solver: 优化求解器
    :return: 每个约束组的可行性检验结果列表，每个元素为 (约束组名称, 求解状态)
    """
    cumulative_constraints = []
    probe_rows = []
    for group in active_groups:
        cumulative_constraints.extend(constraint_map[group])
        probe_problem = cp.Problem(cp.Minimize(0), cumulative_constraints)
        probe_problem.solve(solver=solver, warm_start=False)
        probe_rows.append((group, probe_problem.status))
        if probe_problem.status in ("infeasible", "infeasible_inaccurate"):
            break
    return probe_rows


def penalty_based_cccp(args,
                       uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, uavs_2_targets_channels,
                       uavs_targets_matched_matrix, uavs_cus_matched_matrix,
                       uavs_pos_pre, uavs_pos_cur, uavs_off_duration, cus_off_power,
                       use_penalty_rank1 = True,
                       cus_entertaining_task_size = None
                       ):
    """
    基于惩罚的 CCCP 算法
    
    :param args: 包含所有基础参数的命名空间
    :param uavs_2_cus_channels: UAVs 到 CUs 的信道响应矩阵 (维度: I * J * N * N)
    :param uavs_2_bs_channels: UAVs 到 BS 的信道响应矩阵 (维度: I * 1 * N)
    :param cus_2_bs_channels: CUs 到 BS 的信道响应矩阵 (维度: J * 1)
    :param uavs_2_targets_channels: UAVs 到 targets 的信道响应矩阵 (维度: I * K * N * N)
    :param uavs_targets_matched_matrix: UAVs 和 targets 的匹配矩阵 (维度: I * K) 
    :param uavs_cus_matched_matrix: UAVs 和 CUs 的匹配矩阵 (维度: I * J) (DRL 输出)
    :param uavs_pos_pre: UAVs 前一时刻位置 (维度: I * 3) (DRL 输出)
    :param uavs_pos_cur: UAVs 当前时刻位置 (维度: I * 3) (DRL 输出)
    :param uavs_off_duration: UAVs 感知任务卸载时长 (维度: I) (DRL 输出)
    :param cus_off_power: CUs 娱乐任务卸载功率 (维度: J) (DRL 输出)
    :return: 目标函数总延迟
    """
    if cus_entertaining_task_size is None:
        cus_entertaining_task_size = np.random.uniform(4e3, 8e3, args.cus_num)
    
    # 根据匹配矩阵提取对应的信道响应矩阵
    # 结果维度: (I, N, N)，其中 I 是 UAV 数量，N 是天线数量
    matched_uav_sensing_channel \
        = extract_matched_sensing_channel(uavs_targets_matched_matrix = uavs_targets_matched_matrix,
                                          uavs_2_targets_channels = uavs_2_targets_channels
                                          )

    # 最优目标函数值初始化
    obj_fun_opt = float('inf')
    # 声明优化变量
    var_uavs_sen_beam = [cp.Variable((args.antenna_nums, args.antenna_nums), hermitian=True) for _ in range(args.uavs_num)]
    var_uavs_off_beam = [cp.Variable((args.antenna_nums, args.antenna_nums), hermitian=True) for _ in range(args.uavs_num)]
    var_bs_2_uav_freqs_norm = cp.Variable(args.uavs_num, nonneg=True)
    var_auxiliary_variable_z = cp.Variable(args.uavs_num, nonneg=True)
    var_cus_off_duration = cp.Variable(args.cus_num, nonneg=True)
    var_t = cp.Variable(args.uavs_num, nonneg=True)  # 辅助变量 t 是为了保障满足 DCP, 该变量没有在文中使用
    # 声明初始值
    hat_auxiliary_variable_z = cp.Parameter(args.uavs_num, nonneg=True, value=np.ones(args.uavs_num) * 1e5)
    hat_bs_2_uav_freqs_norm = cp.Parameter(args.uavs_num, nonneg=True, value=np.ones(args.uavs_num) * (args.bs_max_freq / args.freq_scale / args.uavs_num / 2))
    hat_uav_sen_beams = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    hat_uav_off_beams = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    # 预计算系数保障 DPP 架构
    rank1_sen_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_off_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_sen_scaled_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_off_scaled_proj_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums, args.antenna_nums)) for _ in range(args.uavs_num)]
    rank1_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_coef_z = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_coef_f = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    obj5_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    c44_const_terms = cp.Parameter(args.uavs_num, value=np.zeros(args.uavs_num))
    c44_grad_mats = [cp.Parameter((args.antenna_nums, args.antenna_nums), hermitian=True, value=np.eye(args.antenna_nums)) for _ in range(args.uavs_num)]
    c45_term4_const = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_term5_const = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_coef_w = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_coef_b = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_const_w = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    c45_const_b = cp.Parameter(args.cus_num, value=np.zeros(args.cus_num))
    temp_hat_uav_sen_beams, temp_hat_uav_off_beams = initialize_uav_beams(args = args,
                                                                matched_uav_sensing_channel = matched_uav_sensing_channel,
                                                                uavs_2_bs_channels = uavs_2_bs_channels
                                                                )
    for i in range(args.uavs_num):
        hat_uav_sen_beams[i].value = temp_hat_uav_sen_beams[i]
        hat_uav_off_beams[i].value = temp_hat_uav_off_beams[i]
    # 定义罚因子
    cur_penalty_factor = cp.Parameter(nonneg=True, value=args.penalty_factor)
    # 更新 rank-1 投影矩阵
    update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                          rank1_sen_proj_mats, rank1_off_proj_mats,
                                          cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                          rank1_const_terms)
    update_obj5_linearization_parameters(args, hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                         obj5_coef_z, obj5_coef_f, obj5_const_terms)
    update_constraint_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                               matched_uav_sensing_channel, uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels,
                                               uavs_cus_matched_matrix, uavs_off_duration, cus_off_power,
                                               c44_const_terms, c44_grad_mats,
                                               c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b)
    
    print("-------------------------------------------------------- ")
    print(f"-------------------- CCCP算法开始迭代 -------------------- ")
    print("-------------------------------------------------------- ")
    iter_count = 0  # 记录迭代次数
    # infeasible_probe_done = False  # 在问题求解为 infeasible 的时候记录是否已经完成了一次不可行性检验
    # 定义目标函数
    obj_fun = compute_obj_fun(args = args,
                            var_uavs_sen_beam = var_uavs_sen_beam,
                            var_uavs_off_beam = var_uavs_off_beam,
                            var_bs_2_uav_freqs_norm = var_bs_2_uav_freqs_norm,
                            var_auxiliary_variable_z = var_auxiliary_variable_z,
                            var_cus_off_duration = var_cus_off_duration,
                            var_t = var_t,
                            hat_auxiliary_variable_z = hat_auxiliary_variable_z,
                            hat_bs_2_uav_freqs_norm = hat_bs_2_uav_freqs_norm,
                            hat_uav_sen_beams = hat_uav_sen_beams,
                            hat_uav_off_beams = hat_uav_off_beams,
                            rank1_sen_proj_mats = rank1_sen_proj_mats,
                            rank1_off_proj_mats = rank1_off_proj_mats,
                            rank1_sen_scaled_proj_mats = rank1_sen_scaled_proj_mats,
                            rank1_off_scaled_proj_mats = rank1_off_scaled_proj_mats,
                            rank1_const_terms = rank1_const_terms,
                            obj5_coef_z = obj5_coef_z,
                            obj5_coef_f = obj5_coef_f,
                            obj5_const_terms = obj5_const_terms,
                            cus_entertaining_task_size = cus_entertaining_task_size,
                            uavs_off_duration = uavs_off_duration,
                            cus_off_power = cus_off_power,
                            uavs_pos_pre = uavs_pos_pre,
                            uavs_pos_cur = uavs_pos_cur,
                            cur_penalty_factor = cur_penalty_factor,
                            use_penalty_rank1 = use_penalty_rank1,
                            )
    all_constraints = define_constraint(args=args,
                                        var_uavs_sen_beam=var_uavs_sen_beam,
                                        var_uavs_off_beam=var_uavs_off_beam,
                                        var_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm,
                                        var_auxiliary_variable_z=var_auxiliary_variable_z,
                                        var_cus_off_duration=var_cus_off_duration,
                                        var_t=var_t,
                                        hat_uav_sen_beams=hat_uav_sen_beams,
                                        hat_uav_off_beams=hat_uav_off_beams,
                                        cus_entertaining_task_size=cus_entertaining_task_size,
                                        uavs_off_duration=uavs_off_duration,
                                        cus_off_power=cus_off_power,
                                        matched_uav_sensing_channel=matched_uav_sensing_channel,
                                        uavs_2_cus_channels=uavs_2_cus_channels,
                                        uavs_2_bs_channels=uavs_2_bs_channels,
                                        cus_2_bs_channels=cus_2_bs_channels,
                                        uavs_cus_matched_matrix=uavs_cus_matched_matrix,
                                        c44_const_terms=c44_const_terms,
                                        c44_grad_mats=c44_grad_mats,
                                        c45_term4_const=c45_term4_const,
                                        c45_term5_const=c45_term5_const,
                                        c45_coef_w=c45_coef_w,
                                        c45_coef_b=c45_coef_b,
                                        c45_const_w=c45_const_w,
                                        c45_const_b=c45_const_b)
    constraint_map = build_constraint_map(args, all_constraints)
    include_groups = parse_group_list(args.constraint_include_groups)
    exclude_groups = parse_group_list(args.constraint_exclude_groups)
    constraints, active_groups = select_constraints(constraint_map,
                                                    include_groups=include_groups,
                                                    exclude_groups=exclude_groups)
    # print(f"启用约束组: {active_groups}")
    # NOTE - 定义为 DPP，加速求解
    problem = cp.Problem(cp.Minimize(obj_fun), constraints)
    # print(f"是否定义为 DPP : {problem.is_dcp(dpp=True)}")  # 检测是否为 DPP
    rank1_gap_max = float('inf')
    energy_val_list = []
    rank1_val_list = []
    cur_opt_energy_val = float('inf')
    for outer_iter in range(args.max_iterations):
        pre_penalized_obj_val = float('inf')
        for inner_iter in range(args.max_iterations):
            print(f"-------------------- 第 {iter_count + 1} 轮, rho = {cur_penalty_factor.value:.4e} --------------------")
            problem.solve(solver=cp.MOSEK, warm_start=True)
            iter_count += 1
            if problem.status in ("infeasible", "infeasible_inaccurate"):
                print("当前状态下不存在可行原始解，CVXPY 无法为大多数约束计算 violation。")
                return float('inf'), iter_count
            elif problem.status in ("optimal", "optimal_inaccurate", "unbounded", "unbounded_inaccurate"):
                cur_pure_energy_val = compute_pure_energy_value(
                    args=args,
                    cur_uavs_sen_beams=[v.value for v in var_uavs_sen_beam],
                    cur_uavs_off_beams=[v.value for v in var_uavs_off_beam],
                    cur_bs_2_uav_freqs_norm=var_bs_2_uav_freqs_norm.value,
                    cur_auxiliary_variable_z=var_auxiliary_variable_z.value,
                    cur_cus_off_duration=var_cus_off_duration.value,
                    cus_entertaining_task_size=cus_entertaining_task_size,
                    uavs_off_duration=uavs_off_duration,
                    cus_off_power=cus_off_power,
                    uavs_pos_pre=uavs_pos_pre,
                    uavs_pos_cur=uavs_pos_cur
                )
                cur_opt_energy_val = problem.value
                print(f"第 {iter_count} 轮求解的问题的 VALUE 值:", cur_opt_energy_val, f"当前状态 : {problem.status}")
                print(f"第 {iter_count} 轮不考虑罚函数下的纯能耗值:", cur_pure_energy_val)
            for i in range(args.uavs_num):
                hat_uav_sen_beams[i].value = var_uavs_sen_beam[i].value
                hat_uav_off_beams[i].value = var_uavs_off_beam[i].value
            hat_auxiliary_variable_z.value = var_auxiliary_variable_z.value
            hat_bs_2_uav_freqs_norm.value = var_bs_2_uav_freqs_norm.value
            rank1_gap_max = 0.0
            for i in range(args.uavs_num):
                w_gap = np.real(np.trace(var_uavs_sen_beam[i].value)) - np.linalg.norm(var_uavs_sen_beam[i].value, 2)
                b_gap = np.real(np.trace(var_uavs_off_beam[i].value)) - np.linalg.norm(var_uavs_off_beam[i].value, 2)
                rank1_gap_max = max(rank1_gap_max, float(max(w_gap, b_gap)))
            print(f"第 {iter_count} 轮最大 rank1 gap:", rank1_gap_max)
            energy_val_list.append(cur_opt_energy_val)
            rank1_val_list.append(rank1_gap_max)

            obj_fun_opt = cur_opt_energy_val
            if inner_iter != 0 and abs(cur_opt_energy_val - pre_penalized_obj_val) < args.cccp_threshold:
                print(f" Convergency!!! - 第 {iter_count} 轮求解的问题的 VALUE 值:", cur_opt_energy_val)
                print(f" Convergency!!! - 第 {iter_count} 轮最大 rank1 gap:", rank1_gap_max)
                break
            pre_penalized_obj_val = cur_opt_energy_val
            update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                                    rank1_sen_proj_mats, rank1_off_proj_mats,
                                                    cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                                    rank1_const_terms)
            update_obj5_linearization_parameters(args, hat_auxiliary_variable_z, hat_bs_2_uav_freqs_norm,
                                                    obj5_coef_z, obj5_coef_f, obj5_const_terms)
            update_constraint_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                                        matched_uav_sensing_channel, uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels,
                                                        uavs_cus_matched_matrix, uavs_off_duration, cus_off_power,
                                                        c44_const_terms, c44_grad_mats,
                                                        c45_term4_const, c45_term5_const, c45_coef_w, c45_coef_b, c45_const_w, c45_const_b)
        if rank1_gap_max < args.rank1_threshold:
            break
        cur_penalty_factor.value *= args.zoom_factor
        update_rank1_linearization_parameters(args, hat_uav_sen_beams, hat_uav_off_beams,
                                              rank1_sen_proj_mats, rank1_off_proj_mats,
                                              cur_penalty_factor, rank1_sen_scaled_proj_mats, rank1_off_scaled_proj_mats,
                                              rank1_const_terms)

    return obj_fun_opt, iter_count, energy_val_list, rank1_val_list


def dbm_2_watt(dbm):
    """
    将 dBm 转换为瓦特

    :param dbm: 功率值（dBm）
    :return: 功率值（瓦特）
    """
    return 10 ** ((dbm - 30) / 10)


def db_2_watt(db):
    """
    将 dB 转换为瓦特

    :param db: 功率值（dB）
    :return: 功率值（瓦特）
    """
    return 10 ** (db / 10)


if __name__ == "__main__":
    # 获取参数
    args = get_base_args()
    np.random.seed(args.seed)  # 确保仿真结果可复现
    
    uavs_pos, cus_pos, targets_pos = generate_pos(
        uavs_num=args.uavs_num, cus_num=args.cus_num, targets_num=args.targets_num,
        center=(0, 0), radius=args.radius, uav_height=args.uav_height
    )
    
    uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels = compute_com_channel_gain(
        uavs_pos=uavs_pos, cus_pos=cus_pos, ref_path_loss=db_2_watt(args.ref_path_loss_db),
        frac_d_lambda=args.frac_d_lambda, alpha_uav_link=args.alpha_uav_link,
        alpha_cu_link=args.alpha_cu_link, rician_factor=db_2_watt(args.rician_factor_db),
        antenna_nums=args.antenna_nums
    )
    
    uavs_2_targets_channels = compute_sen_channel_gain(
        radar_rcs=args.radar_rcs, frac_d_lambda=args.frac_d_lambda,
        uavs_pos=uavs_pos, targets_pos=targets_pos,
        antenna_nums=args.antenna_nums, ref_path_loss=db_2_watt(args.ref_path_loss_db)
    )
    
    uavs_cus_matched_matrix = random_choose_matched_matrix(uavs_pos=uavs_pos, cus_pos=cus_pos)
    uavs_targets_matched_matrix = match_uav_targets_nearest(uavs_pos=uavs_pos, targets_pos=targets_pos)
    uavs_pos_cur = compute_uav_pos_cur(args=args, uavs_pos_pre=uavs_pos)
    
    # 固定 DRL 侧输入
    uavs_off_duration = np.full(args.uavs_num, (args.uav_max_delay - args.uav_sen_duration) * 0.8)
    cus_off_power = np.full(args.cus_num, dbm_2_watt(args.cu_max_power_dbm) / 5)
    cus_entertaining_task_size = np.random.uniform(4e3, 8e3, args.cus_num)
    
    rho_list = [10]

    for rho in rho_list:
        args.penalty_factor = rho
        energy_opt, _, energy_val_list, rank1_val_list = penalty_based_cccp(
            args=args,
            uavs_2_cus_channels=uavs_2_cus_channels,
            uavs_2_bs_channels=uavs_2_bs_channels,
            cus_2_bs_channels=cus_2_bs_channels,
            uavs_2_targets_channels=uavs_2_targets_channels,
            uavs_targets_matched_matrix=uavs_targets_matched_matrix,
            uavs_cus_matched_matrix=uavs_cus_matched_matrix,
            uavs_pos_pre=uavs_pos,
            uavs_pos_cur=uavs_pos_cur,
            uavs_off_duration=uavs_off_duration,
            cus_off_power=cus_off_power,
            use_penalty_rank1=True,
            cus_entertaining_task_size=cus_entertaining_task_size
        )
        with open(f"{date_str}_energy_val_list_rho{args.penalty_factor}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(energy_val_list)
        with open(f"{date_str}_rank1_val_list_rho{args.penalty_factor}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(rank1_val_list)


    # print("======================================== 分界线 ========================================")
    # print("======================================== 分界线 ========================================")
    # print("======================================== 分界线 ========================================")
    # # 采用基于高斯随机化的 CCCP 算法计算出最优的能耗和迭代次数
    # gaussian_based_energy_opt, iter_count = gaussian_randomized_based_cccp(args = args,
    #                                             uavs_2_cus_channels = uavs_2_cus_channels,
    #                                             uavs_2_bs_channels = uavs_2_bs_channels,
    #                                             cus_2_bs_channels = cus_2_bs_channels,
    #                                             uavs_2_targets_channels = uavs_2_targets_channels,
    #                                             uavs_targets_matched_matrix = uavs_targets_matched_matrix,
    #                                             uavs_cus_matched_matrix = uavs_cus_matched_matrix,
    #                                             uavs_pos_pre = uavs_pos,
    #                                             uavs_pos_cur = uavs_pos_cur,
    #                                             uavs_off_duration = uavs_off_duration,
    #                                             cus_off_power = cus_off_power,
    #                                             use_penalty_rank1 = True,
    #                                             cus_entertaining_task_size = cus_entertaining_task_size)
