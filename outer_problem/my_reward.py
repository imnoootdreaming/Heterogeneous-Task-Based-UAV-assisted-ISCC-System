import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from inner_problem.penalty_based_cccp_algorithm import penalty_based_cccp
import numpy as np


class MyReward:
    def __init__(self, base_args):
        self.base_args = base_args
        self.x_max = float(getattr(base_args, "radius", 600))
        self.y_max = float(getattr(base_args, "radius", 600))

    def reward_compute(self, uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels, uavs_2_targets_channels,
                       uavs_targets_matched_matrix, uavs_cus_matched_matrix,
                       uavs_pos, uavs_pos_cur, uavs_off_duration, cus_off_power, cus_entertaining_task_size):
        uav_collision_penalty = np.zeros(self.base_args.uavs_num)
        uav_exceed_boundary_penalty = np.zeros(self.base_args.uavs_num)

        for i in range(self.base_args.uavs_num):
            x, y, _ = uavs_pos_cur[i]
            if x < -self.x_max or x > self.x_max or y < -self.y_max or y > self.y_max:
                uav_exceed_boundary_penalty[i] = 1
                uavs_pos_cur[i][0] = np.clip(x, -self.x_max, self.x_max)
                uavs_pos_cur[i][1] = np.clip(y, -self.y_max, self.y_max)

            for j in range(i + 1, self.base_args.uavs_num):
                dist = np.linalg.norm(uavs_pos_cur[i] - uavs_pos_cur[j])
                if dist < self.base_args.uav_safe_distance:
                    uav_collision_penalty[i] = 1
                    uav_collision_penalty[j] = 1

        bs_alloc_spectrum_penalty = np.sum(np.maximum(0, np.sum(uavs_cus_matched_matrix, axis=0) - 1))
        uavs_off_duration = [float(x) for x in np.asarray(uavs_off_duration).reshape(-1)]
        cus_off_power = [float(x) for x in np.asarray(cus_off_power).reshape(-1)]

        energy_opt, _, _, _, _, _, _ = penalty_based_cccp(
            args=self.base_args,
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
            cus_entertaining_task_size=cus_entertaining_task_size
        )

        no_solution_penalty = 0
        if energy_opt == float("inf"):
            no_solution_penalty = 1

        total_reward_4_energy = np.exp(-energy_opt / 1000)
        total_reward = (
            total_reward_4_energy
            - bs_alloc_spectrum_penalty
            - np.sum(uav_exceed_boundary_penalty)
            - np.sum(uav_collision_penalty)
            - no_solution_penalty
        )

        reward = {
            "bs": float(total_reward),
            "components": {
                "total_reward_4_energy": float(total_reward_4_energy),
                "bs_alloc_spectrum_penalty": float(bs_alloc_spectrum_penalty),
                "uav_exceed_boundary_penalty_sum": float(np.sum(uav_exceed_boundary_penalty)),
                "uav_collision_penalty_sum": float(np.sum(uav_collision_penalty)),
                "no_solution_penalty":float(no_solution_penalty)
            }
        }
        return float(total_reward), reward, energy_opt