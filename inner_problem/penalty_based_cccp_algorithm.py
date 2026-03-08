import argparse
from math import pi, sqrt
import os
import time
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from equivalent_channel import generate_equivalent_channel
import warnings
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import linear_sum_assignment  # 引入线性求和分配函数
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('error', category=RuntimeWarning)  # 把 RuntimeWarning 当作异常处理
GRID_COLOR = '#E0E0E0'  # 柔和的网格线颜色

def get_args():
    base_parser = argparse.Argumentbase_parser(description="场景的基本参数")

    # 仿真场景参数
    base_parser.add_argument("--num_cases", type=int, default=30, help="随机案例数量")
    base_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    base_parser.add_argument("--targets_num", type=int, default=8, help="目标数量")
    base_parser.add_argument("--cus_num", type=int, default=10, help="CU 数量")
    base_parser.add_argument("--uav_height", type=float, default=100, help="UAV 高度 (m)")
    base_parser.add_argument("--radius", type=float, default=200, help="区域半径 (m)")

    # 信道参数
    base_parser.add_argument("--ref_path_loss_db", type=float, default=-30, help="1m 参考路径损耗 (dB)")
    base_parser.add_argument("--frac_d_lambda", type=float, default=0.5, help="天线间距与波长比例")
    base_parser.add_argument("--alpha_uav_link", type=float, default=2.5, help="UAV 链路路径损耗指数")
    base_parser.add_argument("--alpha_cu_link", type=float, default=2.5, help="CU 链路路径损耗指数")
    base_parser.add_argument("--rician_factor_db", type=float, default=10, help="Rician 因子 (dB)")
    base_parser.add_argument("--antennas_nums", type=int, default=8, help="UAV 天线数量")
    base_parser.add_argument("--radar_rcs", type=float, default=10, help="雷达 RCS (m^2)")
    base_parser.add_argument("--noise_power_density_dbm", type=float, default=-174, help="噪声功率谱密度 (dBm/Hz)")
    base_parser.add_argument("--bandwidth", type=float, default=10e6, help="带宽 (Hz)")

    # 算法/物理参数
    base_parser.add_argument("--uav_c1", type=float, default=0.00614, help="UAV 飞行参数 c1")
    base_parser.add_argument("--uav_c2", type=float, default=15.976, help="UAV 飞行参数 c2")
    base_parser.add_argument("--kappa", type=float, default=1e-28, help="BS CPU 有效开关电容")
    base_parser.add_argument("--bs_max_freq", type=float, default=10e9, help="BS 最大工作频率 (Hz)")
    base_parser.add_argument("--bs_cycles_per_bit", type=float, default=1000, help="BS 处理 1bit 需要的周期数")
    base_parser.add_argument("--time_slot_duration", type=float, default=0.5, help="时隙长度 (s)")
    base_parser.add_argument("--uav_sen_duration", type=float, default=0.1, help="UAV 感知时长 (s)")
    base_parser.add_argument("--cu_max_power_dbm", type=float, default=23, help="CU 最大发射功率 (dBm)")
    base_parser.add_argument("--uav_max_power", type=float, default=1.0, help="UAV 最大功率 (W)")
    
    # 权重参数
    base_parser.add_argument("--omega_weight_1", type=float, default=0.2, help="BS 权重")
    base_parser.add_argument("--omega_weight_2", type=float, default=0.4, help="UAV 权重")
    base_parser.add_argument("--omega_weight_3", type=float, default=0.4, help="CU 权重")

    # 雷达参数
    base_parser.add_argument("--radar_duty_ratio", type=float, default=0.01, help="雷达占空比")
    base_parser.add_argument("--var_range_fluctuation", type=float, default=1e-14, help="范围波动过程方差")
    base_parser.add_argument("--radar_impulse_duration", type=float, default=2e-5, help="雷达脉冲持续时间")
    base_parser.add_argument("--radar_spectrum_shape", type=float, default=pi / sqrt(3), help="雷达频谱形状参数")

    return base_parser.parse_args()


def compute_largest_eigenvector(matrix):
    """
    计算矩阵中最大特征值对应的特征向量

    :param matrix: 需要计算最大特征值对应的特征向量的矩阵
    :return: 矩阵中最大特征值对应的特征向量
    """
    # 确保矩阵的对称性，防止数值截断误差
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # eigh 返回特征值升序排列，取最后一个
    v_max = eigenvectors[:, -1]
    # 返回 v * v^T (秩一矩阵)
    return np.outer(v_max, v_max)


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


def compute_com_channel_gain(uavs_pos, cus_pos, ref_path_loss, frac_d_lambda, alpha_uav_link, alpha_cu_link, rician_factor, antennas_nums):
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
    :param antennas_nums: 天线数量 N
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
            # h_los shape: (N_pos1, N_pos2, antennas_nums)
            # array_response: [1, exp(j*2*pi*d/lambda*sin(phi)), ..., exp(j*2*pi*d/lambda*(N-1)*sin(phi))]
            n_range = np.arange(antennas_nums)
            exponent = 1j * 2 * np.pi * frac_d_lambda * np.sin(phi)[..., np.newaxis] * n_range
            h_los = np.exp(exponent)
            
            # NLoS 分量 (Rayleigh)
            # h_nlos shape: (N_pos1, N_pos2, antennas_nums)
            h_nlos = (np.random.randn(*dist.shape, antennas_nums) + 1j * np.random.randn(*dist.shape, antennas_nums)) / np.sqrt(2)
            
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


def compute_sen_channel_gain(radar_rcs, frac_d_lambda, uavs_pos, targets_pos, antennas_nums, ref_path_loss):
    """
    根据雷达截面积、天线间距与波长的比例，UAVs 的位置和 targets 的位置计算 UAVs 与 targets 之间的信道响应矩阵

    :param radar_rcs: 雷达截面积 (xi)
    :param frac_d_lambda: 天线间距与波长的比例
    :param uavs_pos: UAVs 的位置 (I * 3)
    :param targets_pos: targets 的位置 (J * 3)
    :param antennas_nums: 天线数量 N
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
    n_range = np.arange(antennas_nums)
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

def penalty_based_cccp(args, uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, matched_uav_sensing_channel):
    B = args.bandwidth
    noise_power = dbm_2_watt(args.noise_power_density_dbm) * B
    
    # 提取参数
    c1 = args.uav_c1
    c2 = args.uav_c2
    kappa = args.kappa
    bs_max_frequency = args.bs_max_freq
    bs_1bit_cpu_cycle = args.bs_cycles_per_bit
    
    time_slot_duration = args.time_slot_duration
    uav_sen_duration = args.uav_sen_duration
    cu_max_delay = time_slot_duration
    uav_max_delay = cu_max_delay - uav_sen_duration
    
    cu_max_power = dbm_2_watt(args.cu_max_power_dbm)
    uav_max_power = args.uav_max_power
    
    cus_entertaining_task_size = np.random.uniform(4e3, 8e3, args.cus_num)
    
    omega_weight_1 = args.omega_weight_1
    omega_weight_2 = args.omega_weight_2
    omega_weight_3 = args.omega_weight_3
    
    radar_duty_ratio = args.radar_duty_ratio
    variance_of_range_fluctuation_process = args.var_range_fluctuation
    radar_impulse_duration = args.radar_impulse_duration
    radar_spectrum_shape = args.radar_spectrum_shape
    
    pass

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
    args = get_args()
    
    # 结果参数设定
    num_cases = args.num_cases  # 随机案例数量
    iteration_result = []  # 存储迭代次数
    delay_result = []  # 存储每个案例的总时延结果
    time_result = []  # 存储每个案例的CCCP执行时长
    np.random.seed(args.seed)  # 确保仿真结果可复现
    
    # 仿真参数设定
    targets_num = args.targets_num
    uavs_num = targets_num
    cus_num = args.cus_num
    
    uavs_pos, cus_pos, targets_pos \
        = generate_pos(uavs_num = uavs_num,  # UAVs 数量和 targets 数量设置相同
                       cus_num = cus_num,  # CUs 的数量多于 UAVs
                       targets_num = targets_num,
                       center = (0, 0),
                       radius = args.radius,
                       uav_height = args.uav_height
                       )  # 根据圆心和半径生成 UAVs 和 CUs 位置
    
    # uavs_2_cus_channel: UAVs -> CUs 信道 (I * J * N)
    # uavs_2_bs_channel: UAVs -> BS 信道 (I * 1 * N)
    # cus_2_bs_channel: CUs -> BS 信道 (J * 1)
    uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels \
        = compute_com_channel_gain(uavs_pos = uavs_pos,
                                   cus_pos = cus_pos,
                                   ref_path_loss = db_2_watt(args.ref_path_loss_db),  # 1m 下参考路径损耗
                                   frac_d_lambda = args.frac_d_lambda,  # 天线间距为半波长
                                   alpha_uav_link = args.alpha_uav_link,  # 与 UAV 有关链路的路径损耗系数
                                   alpha_cu_link = args.alpha_cu_link,  # 与 CU 有关的路径损耗系数
                                   rician_factor = db_2_watt(args.rician_factor_db),  # Rician 因子
                                   antennas_nums = args.antennas_nums  # UAV 天线数量
                                   )  # 计算论文通信信道增益
    
    # uavs_2_targets_channels: UAVs -> targets 信道响应矩阵
    uavs_2_targets_channels \
        = compute_sen_channel_gain(radar_rcs = args.radar_rcs,  # 目标 RCS
                                   frac_d_lambda = args.frac_d_lambda,  # 天线间距为半波长
                                   uavs_pos = uavs_pos,
                                   targets_pos = targets_pos,
                                   antennas_nums = args.antennas_nums,  # UAV 天线数量
                                   ref_path_loss = db_2_watt(args.ref_path_loss_db)  # 1m 下参考路径损耗
                                   )  # 计算论文感知信道响应矩阵
    
    # 随机选择 UAVs 复用 CUs 的信道
    uavs_cus_matched_matrix \
        = random_choose_matched_matrix(uavs_pos = uavs_pos,
                                       cus_pos = cus_pos
                                       )  # 随机匹配 UAVs 和 CUs 并确保 CU 只能选择一个 UAV 和一个 UAV 只能被一个 CU 选择
    
    # 每个 UAV 默认感知距离最近的目标 (使用匈牙利算法进行一一匹配)
    uavs_targets_matched_matrix \
        = match_uav_targets_nearest(uavs_pos = uavs_pos,
                                    targets_pos = targets_pos
                                    )
    
    # 根据匹配矩阵提取对应的信道响应矩阵
    # 结果维度: (I, N, N)，其中 I 是 UAV 数量，N 是天线数量
    matched_uav_sensing_channel \
        = extract_matched_sensing_channel(uavs_targets_matched_matrix = uavs_targets_matched_matrix,
                                          uavs_2_targets_channels = uavs_2_targets_channels
                                          )
    
    delay_opt = penalty_based_cccp(args = args, 
                                   uavs_2_cus_channels = uavs_2_cus_channels,
                                   uavs_2_bs_channels = uavs_2_bs_channels,
                                   cus_2_bs_channels = cus_2_bs_channels,
                                   matched_uav_sensing_channel = matched_uav_sensing_channel
                                   )
