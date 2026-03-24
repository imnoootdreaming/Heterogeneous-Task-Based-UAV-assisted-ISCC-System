import gym
from gym import spaces
import numpy as np
from scipy.optimize import linear_sum_assignment
from my_reward import MyReward
from math import pi, sqrt


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


class MyEnv(gym.Env):
    def __init__(self, base_args, madrl_args):
        super(MyEnv, self).__init__()
        # 获取仿真参数设定
        self.t = 0  # 初始时隙为 0 
        self.base_args = base_args
        self.madrl_args = madrl_args
        self.cus_entertaining_task_size = np.ones(self.base_args.cus_num) * 170e3  # 每个 CU 的娱乐任务大小 (bits)
        # -------- 初始化位置 --------
        self.init_uavs_pos, self.init_cus_pos, self.init_targets_pos = self.generate_pos(self.base_args.uavs_num, self.base_args.cus_num, self.base_args.targets_num, self.base_args.center, self.base_args.radius, self.base_args.uav_height)
        self.cur_uavs_pos = self.init_uavs_pos  # 初始的 UAV 位置
        self.precomputed_cus_traj = self.generate_cu_trajectory()  # 提前生成的 CU 轨迹
        self.cur_cus_pos = self.init_cus_pos  # 初始的 CU 位置
        # -------- 初始化位置 --------

        # -------- 初始化信道矩阵 --------
        # uavs_2_cus_channel: UAVs -> CUs 信道 (I * J * N)
        # uavs_2_bs_channel: UAVs -> BS 信道 (I * 1 * N)
        # cus_2_bs_channel: CUs -> BS 信道 (J * 1)
        self.uavs_2_cus_channels, self.uavs_2_bs_channels, self.cus_2_bs_channels \
            = self.compute_com_channel_gain(uavs_pos = self.init_uavs_pos,
                                    cus_pos = self.init_cus_pos,
                                    ref_path_loss = db_2_watt(base_args.ref_path_loss_db),  # 1m 下参考路径损耗
                                    frac_d_lambda = base_args.frac_d_lambda,  # 天线间距为半波长
                                    alpha_uav_link = base_args.alpha_uav_link,  # 与 UAV 有关链路的路径损耗系数
                                    alpha_cu_link = base_args.alpha_cu_link,  # 与 CU 有关的路径损耗系数
                                    rician_factor = db_2_watt(base_args.rician_factor_db),  # Rician 因子
                                    antenna_nums = base_args.antenna_nums  # UAV 天线数量
                                    )  # 计算通信信道增益
        
        # uavs_2_targets_channels: UAVs -> targets 信道响应矩阵 (I * I * N * N)
        # I 个 UAV 对 I 个 TARGET 的信道 I * I * (a * a^{H}) = I * I * N * N
        self.uavs_2_targets_channels \
            = self.compute_sen_channel_gain(radar_rcs = base_args.radar_rcs,  # 目标 RCS
                                    frac_d_lambda = base_args.frac_d_lambda,  # 天线间距为半波长
                                    uavs_pos = self.init_uavs_pos,
                                    targets_pos = self.init_targets_pos,
                                    antenna_nums = base_args.antenna_nums,  # UAV 天线数量
                                    ref_path_loss = db_2_watt(base_args.ref_path_loss_db)  # 1m 下参考路径损耗
                                    )  # 计算感知信道响应矩阵
        
        # 20260324 - 修改 UAV 感知目标对象：预生成每个时隙每个 UAV 的感知目标，
        # 并直接构造当前时隙的 UAV-target 匹配矩阵，替代动态最近匹配。
        self.precomputed_uav_target_schedule = self.generate_uav_target_schedule()
        self.uavs_targets_matched_matrix = self.build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t]
        )
        # -------- 初始化信道矩阵 --------
        
        # -------- 定义动作空间 --------
        self.action_space = {}
        self.epsilon = 1e-8
        #  每个UAV单独动作空间
        #  1. UAV 飞行角度 [0, 2π]
        #  2. UAV 飞行距离 [min_dist, max_dist]
        #  3. 感知任务卸载时长 [epsilon, max_delay - sen_duration]
        self.action_space["uav"] = {
            f"uav_{i}": spaces.Box(
                low=np.array([
                    0,
                    self.base_args.uav_min_speed * self.base_args.time_slot_duration,
                    self.epsilon
                ], dtype=np.float32),
                high=np.array([
                    2 * np.pi,
                    self.base_args.uav_max_speed * self.base_args.time_slot_duration,
                    self.base_args.uav_max_delay - self.base_args.uav_sen_duration
                ], dtype=np.float32),
                dtype=np.float32
            )
            for i in range(self.base_args.uavs_num)
        }
        # BS 动作空间
        # 1. CU 的发射功率
        # 2. UAV 频谱共享系数 [0, cus_num]
        self.action_space["bs"] = spaces.Box(
            low=np.concatenate([
                np.zeros(self.base_args.cus_num) + self.epsilon,  # CU 发射功率
                np.zeros(self.base_args.uavs_num) + self.epsilon  # UAV 频谱共享系数
            ]),
            high=np.concatenate([
                np.ones(self.base_args.cus_num) * dbm_2_watt(self.base_args.cu_max_power_dbm),  # CU 发射功率最大
                np.ones(self.base_args.uavs_num) * self.base_args.cus_num  # UAV 频谱共享系数最大
            ]),
            dtype=np.float32
        )
        # -------- 定义动作空间 --------
        
        # -------- 定义状态空间 --------
        self.observation_space = {}
        # UAV 观测空间
        # 1. UAV i 的坐标
        # 2. UAV <==> BS 的信道 (real + imag 实部 + 虚部)
        # 3. UAV <==> 感知的 TARGET 的信道 (real + imag 实部 + 虚部)
        # 20260324 - 修改 UAV 感知目标对象：为 UAV observation 显式加入目标坐标 3 维。
        obs_dim_uav = 6 + self.base_args.antenna_nums * 2 + self.base_args.antenna_nums * self.base_args.antenna_nums * 2
        obs_dim_uav = 6 + self.base_args.antenna_nums * 2 + self.base_args.antenna_nums * self.base_args.antenna_nums * 2
        self.observation_space["uav"] = {
            f"uav_{i}": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim_uav,),
                dtype=np.float32
            )
            for i in range(self.base_args.uavs_num)
        }

        # BS 观测空间
        # 1. 每个 CU <==> BS 信道 (shape = (real + imag 实部 + 虚部))
        # 2. 每个 CU <==> 每个 UAV 信道 (shape = (real + imag 实部 + 虚部))
        obs_dim_bs = (
            self.base_args.cus_num * 2 +                              # CU-BS 信道
            self.base_args.cus_num * self.base_args.uavs_num * self.base_args.antenna_nums * 2    # CU-UAV 信道
        )
        self.observation_space["bs"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim_bs,),
            dtype=np.float32
        )
        # -------- 定义状态空间 --------

        # -------- 初始化奖励函数 --------
        self.reward_calculator = MyReward(self.base_args) # 初始化奖励类
        # -------- 初始化奖励函数 --------

        self.t = 0  # 初始时隙 index 为 0 

    def step(self, actions, i_episode=None):
        """
        执行动作，返回 (obs, reward, done, info)
        
        Args:
            action (dict): 动作字典
                action["uav"][id] = np.array([angle, dist, delay, cu_idx_cont])
                action["bs"] = ...
            i_episode: 当前 episode 索引 (optional)
        """

        uav_actions_dict = actions["uav"]
        diff_thetas = []
        diff_distances = []
        off_durations = []

        # UAV 动作提取
        for i in range(self.base_args.uavs_num):
            act = np.array(uav_actions_dict[f"uav_{i}"])
            low = self.action_space["uav"][f"uav_{i}"].low
            high = self.action_space["uav"][f"uav_{i}"].high
            act = act * (high - low) + low
            diff_thetas.append(act[0])
            diff_distances.append(act[1])
            off_durations.append(act[2])
        diff_theta = np.array(diff_thetas)
        diff_distance = np.array(diff_distances)
        off_duration = np.array(off_durations)

        # BS 动作提取
        bs_actions = actions["bs"]
        bs_actions_low = self.action_space["bs"].low
        bs_actions_high = self.action_space["bs"].high
        bs_actions = bs_actions * (bs_actions_high - bs_actions_low) + bs_actions_low
        cus_off_power = bs_actions[:self.base_args.cus_num]
        uav_spectrum_sharing_coeffs = bs_actions[self.base_args.cus_num:]
        # 将每个UAV选择的CU索引转换为 one-hot 矩阵
        uavs_cus_matched_matrix = np.zeros((self.base_args.uavs_num, self.base_args.cus_num))
        for i in range(self.base_args.uavs_num):
            # 取整得到CU索引
            cu_idx = int(np.clip(round(uav_spectrum_sharing_coeffs[i]), 0, self.base_args.cus_num - 1))
            uavs_cus_matched_matrix[i, cu_idx] = 1

        # next_uavs_pos 根据动作生成
        next_uavs_pos = self.cur_uavs_pos + np.stack([diff_distance * np.cos(diff_theta),
                                                    diff_distance * np.sin(diff_theta),
                                                    np.zeros_like(diff_distance * np.cos(diff_theta))], axis=1)
        
        

        total_reward, reward, energy_opt = self.reward_calculator.reward_compute(uavs_2_cus_channels = self.uavs_2_cus_channels,
                                                                                  uavs_2_bs_channels = self.uavs_2_bs_channels,
                                                                                  cus_2_bs_channels = self.cus_2_bs_channels,
                                                                                  uavs_2_targets_channels = self.uavs_2_targets_channels,
                                                                                  uavs_targets_matched_matrix = self.uavs_targets_matched_matrix,
                                                                                  uavs_cus_matched_matrix = uavs_cus_matched_matrix,
                                                                                  uavs_pos = self.cur_uavs_pos,
                                                                                  uavs_pos_cur = next_uavs_pos,
                                                                                  uavs_off_duration = off_duration,
                                                                                  cus_off_power = cus_off_power,
                                                                                  cus_entertaining_task_size = self.cus_entertaining_task_size
                                                                                  )
        print(f"Episode {i_episode}, Time Slot {self.t}: Total Reward = {total_reward:.4f}, Energy Opt = {energy_opt:.4f}")
        # 下一状态更新
        self.t += 1  # 时隙 index 增加 1
        self.cur_uavs_pos = next_uavs_pos  # 更新UAV位置
        self.cur_cus_pos = self.precomputed_cus_traj[self.t]  # 更新 CU 位置
        # 更新信道状态
        self.uavs_2_cus_channels, self.uavs_2_bs_channels, self.cus_2_bs_channels \
            = self.compute_com_channel_gain(uavs_pos = self.cur_uavs_pos,
                                cus_pos = self.cur_cus_pos,
                                ref_path_loss = db_2_watt(self.base_args.ref_path_loss_db),  # 1m 下参考路径损耗
                                frac_d_lambda = self.base_args.frac_d_lambda,  # 天线间距为半波长
                                alpha_uav_link = self.base_args.alpha_uav_link,  # 与 UAV 有关链路的路径损耗系数
                                alpha_cu_link = self.base_args.alpha_cu_link,  # 与 CU 有关的路径损耗系数
                                rician_factor = db_2_watt(self.base_args.rician_factor_db),  # Rician 因子
                                antenna_nums = self.base_args.antenna_nums  # UAV 天线数量
                                )  # 计算通信信道增益

        # uavs_2_targets_channels: UAVs -> targets 信道响应矩阵 (I * I * N * N)
        # I 个 UAV 对 I 个 TARGET 的信道 I * I * (a * a^{H}) = I * I * N * N
        self.uavs_2_targets_channels \
            = self.compute_sen_channel_gain(radar_rcs = self.base_args.radar_rcs,  # 目标 RCS
                                    frac_d_lambda = self.base_args.frac_d_lambda,  # 天线间距为半波长
                                    uavs_pos = self.cur_uavs_pos,
                                    targets_pos = self.init_targets_pos,
                                    antenna_nums = self.base_args.antenna_nums,  # UAV 天线数量
                                    ref_path_loss = db_2_watt(self.base_args.ref_path_loss_db)  # 1m 下参考路径损耗 
                                    )  # 计算感知信道响应矩阵

        # 20260324 - 修改 UAV 感知目标对象：下一时隙的 UAV-target 匹配矩阵
        self.uavs_targets_matched_matrix = self.build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t]
        )

        next_state_dict = {"uav": {}, "bs": None}

        # 生成下一个状态空间
        for i in range(self.base_args.uavs_num):
            uav_coord = self.cur_uavs_pos[i]
            uavs_2_cus_channels_i = self.uavs_2_cus_channels[i]
            uavs_2_bs_channels_i = self.uavs_2_bs_channels[i]
            # 根据索引选择 UAV 此时感知的 TARGET
            target_idx = int(self.precomputed_uav_target_schedule[self.t, i])
            # 20260324 - 修改 UAV 感知目标对象：next_state 中显式加入当前时隙的目标坐标。
            target_coord = self.init_targets_pos[target_idx]
            target_coord = self.init_targets_pos[target_idx]
            uavs_2_targets_channels_i = self.uavs_2_targets_channels[i, target_idx]  # 选择对应的通道
            uav_obs = np.concatenate([
                                    uav_coord.astype(np.float32),
                                    target_coord.astype(np.float32),
                                    uavs_2_bs_channels_i.real.flatten(),
                                    uavs_2_bs_channels_i.imag.flatten(),
                                    uavs_2_targets_channels_i.real.flatten(),
                                    uavs_2_targets_channels_i.imag.flatten()
                                ]).astype(np.float32)
            next_state_dict["uav"][f"uav_{i}"] = uav_obs
        # BS 观测空间
        # 1. CU <==> BS 信道
        # 2. CU <==> UAV 信道
        bs_obs = np.concatenate([self.cus_2_bs_channels.real.flatten(), self.cus_2_bs_channels.imag.flatten(),
                                self.uavs_2_cus_channels.real.flatten(), self.uavs_2_cus_channels.imag.flatten()]).astype(np.float32)
        next_state_dict["bs"] = bs_obs

        # 判断是否需要进行截断
        if self.t < self.madrl_args.total_time_slots:
            done = 0
        else:
            done = 1

        return next_state_dict, float(total_reward), reward, done

    # 用于重置环境
    def reset(self):
        self.t = 0  # 时隙初始化
        # 初始化到最初状态
        # init_cur_uavs_pos: 初始 UAV 坐标
        # init_user_pos: 初始化用户坐标
        self.cur_uavs_pos = self.init_uavs_pos.copy()
        self.cur_cus_pos = self.precomputed_cus_traj[self.t].copy()  # 读取轨迹的初始点
        
        # 初始化信道
        self.uavs_2_cus_channels, self.uavs_2_bs_channels, self.cus_2_bs_channels \
            = self.compute_com_channel_gain(uavs_pos = self.cur_uavs_pos,
                                   cus_pos = self.cur_cus_pos,
                                   ref_path_loss = db_2_watt(self.base_args.ref_path_loss_db),  # 1m 下参考路径损耗
                                   frac_d_lambda = self.base_args.frac_d_lambda,  # 天线间距为半波长
                                   alpha_uav_link = self.base_args.alpha_uav_link,  # 与 UAV 有关链路的路径损耗系数
                                   alpha_cu_link = self.base_args.alpha_cu_link,  # 与 CU 有关的路径损耗系数
                                   rician_factor = db_2_watt(self.base_args.rician_factor_db),  # Rician 因子
                                   antenna_nums = self.base_args.antenna_nums  # UAV 天线数量
                                   )  # 计算通信信道增益
        self.uavs_2_targets_channels \
            = self.compute_sen_channel_gain(radar_rcs = self.base_args.radar_rcs,  # 目标 RCS
                                    frac_d_lambda = self.base_args.frac_d_lambda,  # 天线间距为半波长
                                    uavs_pos = self.cur_uavs_pos,
                                    targets_pos = self.init_targets_pos,
                                    antenna_nums = self.base_args.antenna_nums,  # UAV 天线数量
                                    ref_path_loss = db_2_watt(self.base_args.ref_path_loss_db)  # 1m 下参考路径损耗 
                                    )  # 计算感知信道响应矩阵
        # 初始化 UAV <==> TARGET 匹配矩阵
        self.uavs_targets_matched_matrix = self.build_uav_targets_matched_matrix(
            self.precomputed_uav_target_schedule[self.t]
        )

        # 初始化观测空间 init_state_dict
        # UAV 观测空间
        # 1. UAV i 的坐标
        # 2. TARGET <==> UAV 信道
        # 3. UAV <==> BS 的信道
        # TODO - 删除 TARGET <==> UAV 信道
        init_state_dict = {"uav": {}, "bs": None}
        for i in range(self.base_args.uavs_num):
            uav_coord = self.init_uavs_pos[i]
            uavs_2_bs_channels_i = self.uavs_2_bs_channels[i]
            # 根据匹配矩阵选择对应的目标通道
            target_idx = int(self.precomputed_uav_target_schedule[self.t, i])
            # 20260324 - 修改 UAV 感知目标对象：reset 初始状态中显式加入当前时隙的目标坐标。
            target_coord = self.init_targets_pos[target_idx]
            uavs_2_targets_channels_i = self.uavs_2_targets_channels[i, target_idx]  # 选择对应的通道
            target_coord = self.init_targets_pos[target_idx]
            uav_obs = np.concatenate([
                                    uav_coord.astype(np.float32),
                                    target_coord.astype(np.float32),
                                    uavs_2_bs_channels_i.real.flatten(),
                                    uavs_2_bs_channels_i.imag.flatten(),
                                    uavs_2_targets_channels_i.real.flatten(),
                                    uavs_2_targets_channels_i.imag.flatten()
                                ]).astype(np.float32)
            init_state_dict["uav"][f"uav_{i}"] = uav_obs
        # BS 观测空间
        # 1. CU <==> BS 信道
        # 2. CU <==> UAV 信道
        bs_obs = np.concatenate([self.cus_2_bs_channels.real.flatten(), self.cus_2_bs_channels.imag.flatten(),
                                self.uavs_2_cus_channels.real.flatten(), self.uavs_2_cus_channels.imag.flatten()]).astype(np.float32)
        init_state_dict["bs"] = bs_obs

        return init_state_dict  # reward, done, info can't be included


    def getPosUAV(self):
        """
        获取当前时隙 UAV 坐标
        """
        return self.cur_uavs_pos.copy()


    def getPosCU(self):
        """
        获取当前时隙 CU 坐标
        """
        # cur_cus_pos stores all CUs at the current time slot.
        return self.cur_cus_pos.copy()


    def getPosTarget(self):
        """
        获取当前时隙 Target 坐标
        """
        return self.init_targets_pos.copy()


    def generate_uav_target_schedule(self):
        total_slots = self.madrl_args.total_time_slots + 1
        schedule = np.zeros((total_slots, self.base_args.uavs_num), dtype=np.int64)
        for t in range(total_slots):
            if self.base_args.targets_num >= self.base_args.uavs_num:
                schedule[t] = np.random.permutation(self.base_args.targets_num)[:self.base_args.uavs_num]
            else:
                schedule[t] = np.random.choice(
                    self.base_args.targets_num,
                    size=self.base_args.uavs_num,
                    replace=True
                )
        return schedule


    def build_uav_targets_matched_matrix(self, target_indices):
        matched_matrix = np.zeros((self.base_args.uavs_num, self.base_args.targets_num), dtype=np.float32)
        clipped_target_indices = np.clip(
            np.asarray(target_indices, dtype=np.int64),
            0,
            self.base_args.targets_num - 1
        )
        for i, target_idx in enumerate(clipped_target_indices):
            matched_matrix[i, target_idx] = 1.0
        return matched_matrix


    def generate_pos(self, uavs_num, cus_num, targets_num, center, radius, uav_height):
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
            
        return uavs_pos, cus_pos, targets_pos


    def compute_com_channel_gain(self, uavs_pos, cus_pos, ref_path_loss, frac_d_lambda, alpha_uav_link, alpha_cu_link, rician_factor, antenna_nums):
        """
        UAVs, CUs 和 BS 之间的均建模为莱斯路径损耗模型
        根据莱斯路径损耗模型计算 UAVs、CUs 和 BS 之间的通信信道增益
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


    def compute_sen_channel_gain(self, radar_rcs, frac_d_lambda, uavs_pos, targets_pos, antenna_nums, ref_path_loss):
        """
        根据雷达截面积、天线间距与波长的比例，UAVs 的位置和 targets 的位置计算 UAVs 与 targets 之间的信道响应矩阵
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


    def generate_cu_trajectory(self):
        """
        Gauss-Markov 模型轨迹
        """
        # NOTE - 比总的时隙多生成一个时刻，因为最后一个时刻还要获取下一个时刻的动作
        traj = np.zeros((self.madrl_args.total_time_slots + 1, self.base_args.cus_num, 3), dtype=float)
        # 使用二维向量表示速度，因为用户在地面移动
        velocities = np.zeros((self.madrl_args.total_time_slots + 1, self.base_args.cus_num, 2), dtype=float)

        # 初始化起始位置和速度
        if self.init_cus_pos.shape[0] > 0:
            # Preserve each CU's own initial position at t=0.
            traj[0] = self.init_cus_pos.copy()
        
        # Fix: markov_velocity is list, need to convert or use directly if numpy handled
        # Assuming base_args.markov_velocity is list like [1, 0, 0]
        v_init = np.array(self.base_args.markov_velocity)[:2]
        velocities[0] = np.tile(v_init, (self.base_args.cus_num, 1))
        
        # Helper for random numbers
        rng = np.random.default_rng(self.base_args.seed)

        # 循环为所有时间片生成轨迹
        for t in range(self.madrl_args.total_time_slots):
            # 为所有用户生成随机分量
            # 对应于 Gauss-Markov 模型中的随机过程 w_k[n]
            random_component = rng.normal(size=(self.base_args.cus_num, 2))
            
            # Use arrays for args
            v_bar = np.array(self.base_args.markov_asymptotic_mean_of_velocity)[:2]

            # 根据高斯-马尔可夫模型方程更新速度:
            # v_k[n+1] = α*v_k[n] + (1-α)*v_bar + sqrt(1-α^2)*σ_bar*w_k[n]
            velocities[t + 1] = (self.base_args.markov_memory_level * velocities[t] +  
                                 (1 - self.base_args.markov_memory_level) * v_bar +  
                                 np.sqrt(1 - self.base_args.markov_memory_level ** 2) * self.base_args.markov_standard_deviation_of_velocity * random_component)  
            
            # 根据上一个时间片的速度更新位置: p_k[n+1] = p_k[n] + v_k[n]*Δt
            # traj[t, :, :2]: p_k[n] (当前位置)
            # velocities[t]: v_k[n] (当前速度)
            # self.tau: Δt (时间片时长)
            traj[t + 1, :, :2] = traj[t, :, :2] + velocities[t] * self.base_args.time_slot_duration

            # 保持z坐标不变，因为用户在地面上
            traj[t + 1, :, 2] = traj[t, :, 2]

        return traj


    def update_cus_pos(self):
        """
        用户移动：直接取预生成轨迹
            如果用户移动超出边界 则拉回边界
        """
        if self.t < len(self.precomputed_cus_traj):
            self.cus_pos[self.t] = self.precomputed_cus_traj[self.t]
            # 限制用户在给定圆形区域内
            for i in range(self.base_args.cus_num):
                pos_2d = self.cus_pos[self.t, i, :2]  # 只取 x, y
                vec = pos_2d - self.base_args.center
                dist = np.linalg.norm(vec)
                if dist > self.base_args.radius:
                    # 拉回边界点
                    clipped = np.array(self.base_args.center) + vec / dist * self.base_args.radius
                    self.cus_pos[self.t, i, 0] = clipped[0]
                    self.cus_pos[self.t, i, 1] = clipped[1]
