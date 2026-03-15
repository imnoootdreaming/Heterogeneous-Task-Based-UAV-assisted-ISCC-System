import gym
from gym import spaces
import numpy as np


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
    """
    通信系统优化自定义环境
    继承自 gym.Env
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, base_args, madrl_args):
        """
        系统参数初始化
        """
        super(MyEnv, self).__init__()
        # 获取仿真参数设定
        self.base_args = base_args
        self.madrl_args = madrl_args
        # 初始化位置
        self.uavs_pos = np.zeros((self.madrl_args.total_time_slots, self.base_args.uavs_num, 3), dtype=float)
        self.cus_pos = np.zeros((self.madrl_args.total_time_slots, self.base_args.cus_num, 3), dtype=float)
        self.targets_pos = np.zeros((self.madrl_args.total_time_slots, self.base_args.targets_num, 3), dtype=float)
        self.precomputed_cus_traj = generate_cu_trajectory()  # 提前生成的 CU 轨迹
        
        #  动作空间定义 
        self.action_space = {}
        self.epsilon = 1e-8
        
        #  每个UAV单独动作空间
        #  1. UAV 飞行角度 [0, 2π]
        #  2. UAV 飞行距离 [min_dist, max_dist]
        #  3. 感知任务卸载时长 [epsilon, max_delay - sen_duration]
        #  4. 频谱共享系数 (J维) [0, 1]
        self.action_space["uav"] = {
            f"uav_{i}": spaces.Dict({
                # 连续动作：方向角、移动距离、感知延迟
                "uav_action_continuous": spaces.Box(
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
                ),
                # 离散动作：选择复用频谱 CU 索引 (0 ~ cus_num-1)
                "uav_action_discrete": spaces.Discrete(self.base_args.cus_num)
            })
            for i in range(self.base_args.uavs_num)
        }

        # BS 动作空间
        # CU 的发射功率
        # BS 动作空间
        self.action_space["bs"] = spaces.Box(
            low=np.full(self.base_args.cus_num, self.epsilon, dtype=np.float32),
            high=np.full(self.base_args.cus_num, dbm_2_watt(self.base_args.cu_max_power_dbm), dtype=np.float32),
            dtype=np.float32
        )

        # 观测空间定义
        self.observation_space = {}

        # UAV 观测空间
        # 1. UAV i 的坐标
        # 3. CU <==> UAV 信道
        # 4. UAV <==> BS 的信道
        self.observation_space["uav"] = {
            f"uav_{i}": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3 +
                       self.base_args.uavs_num * self.base_args.cus_num +
                       1,),
                dtype=np.float32
            )
            for i in range(self.base_args.uavs_num)
        }

        # BS 观测空间
        # 1. CU <=> BS 信道
        self.observation_space["bs"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.base_args.cus_num),
            dtype=np.float32
        )

        self.reward = MyReward() # 初始化奖励类
        
        self.t = 0  # 初始时隙 index 为 0 


    def reset(self):
        """
        环境重置，返回初始观测
        Returns:
            observation (np.array): 初始观测
        """
        self.steps = 0
        # 初始化信道状态、用户位置等
        self.state = np.zeros(self.obs_dim) 
        
        return self._get_obs()


    def step(self, action):
        """
        执行动作，返回 (obs, reward, done, info)
        
        Args:
            action (dict or tuple): 混合动作，包含连续和离散部分
        """
        self.steps += 1
        
        # 1. 解析动作 (Parse Action)
        # action 可能是 dict 或 tuple，取决于 Agent 的输出
        if isinstance(action, dict):
            cont_action = action['continuous']
            disc_action = action['discrete']
        elif isinstance(action, (tuple, list)):
             # 假设顺序: [continuous, discrete]
             cont_action = action[0]
             disc_action = action[1]
        else:
             # 如果动作被展平 (flattened)
             cont_action = action[:self.cont_action_dim]
             disc_action = int(action[self.cont_action_dim])

        # 2. 更新系统状态 (Update System State)
        # 应用功率、波束赋形和频谱复用策略
        # 更新信道状态、干扰、SINR 等
        
        # 3. 计算奖励 (Calculate Reward)
        # 对接 MyReward
        # reward = self.reward_calculator.calculate(self.state, action)
        reward = 0.0 # 占位符
        
        # 4. 检查终止条件 (Check Termination)
        done = self.steps >= self.max_steps
        
        # 5. 获取下一时刻观测 (Get Next Observation)
        next_obs = self._get_obs()
        
        info = {}
        
        return next_obs, reward, done, info


    def _get_obs(self):
        """
        构建观测向量
        Returns:
            obs (np.array): 当前观测
        """
        # 收集信道增益 (H), 干扰 (I), 队列长度 (Q)
        # 进行必要的归一化
        return self.state


    def render(self, mode='human'):
        """
        可视化环境
        """
        print(f"Step: {self.steps}, State: {self.state}")
    

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
        # targets_pos[:, 2] = 0  # 默认为 0
        self.uavs_pos = uavs_pos
        self.cus_pos = cus_pos
        self.targets_pos = targets_pos
        return uavs_pos, cus_pos, targets_pos


    def compute_com_channel_gain(self, uavs_pos, cus_pos, ref_path_loss, frac_d_lambda, alpha_uav_link, alpha_cu_link, rician_factor, antenna_nums):
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


    def compute_sen_channel_gain(self, radar_rcs, frac_d_lambda, uavs_pos, targets_pos, antenna_nums, ref_path_loss):
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


    def match_uav_targets_nearest(self, uavs_pos, targets_pos):
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


    def generate_cu_trajectory(self):
        """
        Gauss-Markov 模型轨迹
        """
        traj = np.zeros((self.total_time_slots, self.base_args.cus_num, 3), dtype=float)
        # 使用二维向量表示速度，因为用户在地面移动
        velocities = np.zeros((self.madrl_args.total_time_slots, self.base_args.cus_num, 2), dtype=float)

        # 初始化起始位置和速度
        traj[0] = self.cus_pos.copy()
        velocities[0] = np.tile(self.args.markov_velocity, (self.base_args.cus_num, 1))

        # 循环为所有时间片生成轨迹
        for t in range(self.madrl_args.total_time_slots - 1):
            # 为所有用户生成随机分量
            # 对应于 Gauss-Markov 模型中的随机过程 w_k[n]
            random_component = self.rng.normal(size=(self.base_args.cus_num, 2))

            # 根据高斯-马尔可夫模型方程更新速度:
            # v_k[n+1] = α*v_k[n] + (1-α)*v_bar + sqrt(1-α^2)*σ_bar*w_k[n]
            velocities[t + 1] = (self.base_args.markov_memory_level * velocities[t] +  
                                 (1 - self.base_args.markov_memory_level) * self.base_args.markov_asymptotic_mean_of_velocity +  
                                 np.sqrt(1 - self.base_args.markov_memory_level ** 2) * self.base_args.markov_standard_deviation_of_velocity * random_component)  

            # 根据上一个时间片的速度更新位置: p_k[n+1] = p_k[n] + v_k[n]*Δt
            # traj[t, :, :2]: p_k[n] (当前位置)
            # velocities[t]: v_k[n] (当前速度)
            # self.tau: Δt (时间片时长)
            traj[t + 1, :, :2] = traj[t, :, :2] + velocities[t] * self.time_slot_duration

            # 保持z坐标不变，因为用户在地面上
            traj[t + 1, :, 2] = traj[t, :, 2]

        return traj


    def update_cus_pos(self):
        """
        用户移动：直接取预生成轨迹
            如果用户移动超出边界 则拉回边界
        """
        self.cus_pos = self.precomputed_cus_traj[self.t]
        # 限制用户在给定圆形区域内
        for i in range(self.cus_pos.shape[0]):
            pos_2d = self.cus_pos[i, :2]  # 只取 x, y
            vec = pos_2d - [0, 0]
            dist = np.linalg.norm(vec)
            if dist > self.base_args.radius:
                # 拉回边界点
                clipped = self.base_args.center + vec / dist * self.base_args.radius
                self.cus_pos[i, 0] = clipped[0]
                self.cus_pos[i, 1] = clipped[1]