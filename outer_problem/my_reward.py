import os
import sys
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取outer_problem的父目录（即项目根目录）
parent_dir = os.path.dirname(current_dir)
# 将父目录添加到sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from inner_problem.penalty_based_cccp_algorithm import penalty_based_cccp
import numpy as np

class MyReward:
    def __init__(self, base_args):
        self.base_args = base_args
        self.x_max = 200
        self.y_max = 200
        self.safe_distance = 5  # 安全距离设为5米
        pass

    def reward_compute(self, uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, uavs_2_targets_channels, 
                       uavs_targets_matched_matrix, uavs_cus_matched_matrix, 
                       uavs_pos, uavs_pos_cur, uavs_off_duration, cus_off_power):
        uav_collision_penalty = np.zeros(self.base_args.uavs_num)  # 每个 UAV 的碰撞惩罚
        uav_exceed_boundary_penalty = np.zeros(self.base_args.uavs_num)  # 每个 UAV 的越界惩罚
        reward = {"uav": [0.0] * self.base_args.uavs_num, "bs": 0.0}

        # UAV 边界检查
        for i in range(self.base_args.uavs_num):
            x, y, z = uavs_pos[i]
            if x < -self.x_max or x > self.x_max or y < -self.y_max or y > self.y_max:
                uav_exceed_boundary_penalty[i] = 1
                # 拉回边界
                uavs_pos[i][0] = np.clip(x, -self.x_max, self.x_max)
                uavs_pos[i][1] = np.clip(y, -self.y_max, self.y_max)    
            # UAV 碰撞检测
            for j in range(i + 1, self.base_args.uavs_num):
                dist = np.linalg.norm(uavs_pos[i] - uavs_pos[j])
                if dist < self.safe_distance:  # 小于5米判定为碰撞或过近
                    uav_collision_penalty[i] = 1
                    uav_collision_penalty[j] = 1
        
        # 对每一列求和，计算每个CU被多少个UAV选择，对于每个CU，如果被选择的UAV数量大于1，多出的部分就是惩罚
        bs_alloc_spectrum_penalty = np.sum(np.maximum(0, np.sum(uavs_cus_matched_matrix, axis=0) - 1))

        energy_opt, _ = penalty_based_cccp_algorithm.penalty_based_cccp(args = self.base_args,
                                                                        uavs_2_cus_channels = uavs_2_cus_channels,
                                                                        uavs_2_bs_channels = uavs_2_bs_channels,
                                                                        cus_2_bs_channels = cus_2_bs_channels,
                                                                        uavs_2_targets_channels = uavs_2_targets_channels,
                                                                        uavs_targets_matched_matrix = uavs_targets_matched_matrix,
                                                                        uavs_cus_matched_matrix = uavs_cus_matched_matrix,
                                                                        uavs_pos_pre = uavs_pos,
                                                                        uavs_pos_cur = uavs_pos_cur,
                                                                        uavs_off_duration = uavs_off_duration,
                                                                        cus_off_power = cus_off_power)
                                                                        
        reward_4_energy = np.exp(-energy_opt)
        for i in range(self.base_args.uavs_num):
            reward["uav"][i] = reward_4_energy - uav_collision_penalty[i] - uav_exceed_boundary_penalty[i]
        reward["bs"] = reward_4_energy - bs_alloc_spectrum_penalty
        # 计算总奖励
        total_reward = reward_4_energy - bs_alloc_spectrum_penalty - np.sum(uav_exceed_boundary_penalty) - np.sum(uav_collision_penalty)
        return total_reward, reward, energy_opt
