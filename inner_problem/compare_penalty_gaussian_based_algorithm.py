import argparse
import csv
import os
from datetime import datetime
from math import pi, sqrt

import numpy as np


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def get_base_args():
    base_parser = argparse.ArgumentParser(description="Compare penalty-based and gaussian-based CCCP on random cases.")

    # Scenario parameters
    base_parser.add_argument("--num_cases", type=int, default=10, help="Number of random cases.")
    base_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    base_parser.add_argument("--targets_num", type=int, default=8, help="Number of targets.")
    base_parser.add_argument("--uavs_num", type=int, default=8, help="Number of UAVs.")
    base_parser.add_argument("--cus_num", type=int, default=16, help="Number of CUs.")
    base_parser.add_argument("--uav_height", type=float, default=100, help="UAV height in meters.")
    base_parser.add_argument("--radius", type=float, default=200, help="Scenario radius in meters.")

    # Channel parameters
    base_parser.add_argument("--ref_path_loss_db", type=float, default=-30, help="Reference path loss at 1m in dB.")
    base_parser.add_argument("--frac_d_lambda", type=float, default=0.5, help="Antenna spacing over wavelength.")
    base_parser.add_argument("--alpha_uav_link", type=float, default=2, help="Path loss exponent of UAV links.")
    base_parser.add_argument("--alpha_cu_link", type=float, default=2.5, help="Path loss exponent of CU links.")
    base_parser.add_argument("--rician_factor_db", type=float, default=10, help="Rician factor in dB.")
    base_parser.add_argument("--antenna_nums", type=int, default=6, help="Number of antennas per UAV.")
    base_parser.add_argument("--radar_rcs", type=float, default=10, help="Radar RCS in mm^2.")
    base_parser.add_argument("--noise_power_density_dbm", type=float, default=-174, help="Noise power spectral density in dBm/Hz.")
    base_parser.add_argument("--bandwidth", type=float, default=10e6, help="Bandwidth in Hz.")

    # Algorithm and physical parameters
    base_parser.add_argument("--uav_c1", type=float, default=0.00614, help="UAV flight parameter c1.")
    base_parser.add_argument("--uav_c2", type=float, default=15.976, help="UAV flight parameter c2.")
    base_parser.add_argument("--kappa", type=float, default=1e-28, help="BS CPU effective switched capacitance.")
    base_parser.add_argument("--bs_max_freq", type=float, default=10e9, help="Maximum BS CPU frequency in Hz.")
    base_parser.add_argument("--freq_scale", type=float, default=1e9, help="Frequency normalization scale.")
    base_parser.add_argument("--z_scale", type=float, default=1e5, help="z-variable normalization scale.")
    base_parser.add_argument("--bs_cycles_per_bit", type=float, default=1000, help="BS cycles required per bit.")
    base_parser.add_argument("--time_slot_duration", type=float, default=0.6, help="Time-slot duration in seconds.")
    base_parser.add_argument("--uav_sen_duration", type=float, default=0.1, help="UAV sensing duration in seconds.")
    base_parser.add_argument("--cu_max_power_dbm", type=float, default=23, help="Maximum CU transmit power in dBm.")
    base_parser.add_argument("--uav_max_power", type=float, default=10, help="Maximum UAV transmit power in W.")
    base_parser.add_argument("--cu_max_delay", type=float, default=0.6, help="Maximum CU delay in seconds.")
    base_parser.add_argument("--uav_max_delay", type=float, default=0.2, help="Maximum UAV delay in seconds.")
    base_parser.add_argument("--uav_max_speed", type=float, default=10.0, help="Maximum UAV speed in m/s.")
    base_parser.add_argument("--uav_min_speed", type=float, default=5.0, help="Minimum UAV speed in m/s.")
    base_parser.add_argument("--uav_safe_distance", type=float, default=5.0, help="Minimum safe UAV distance in meters.")
    base_parser.add_argument("--sen_sinr", type=float, default=20, help="Sensing SINR threshold in dB.")

    # Weight parameters
    base_parser.add_argument("--omega_weight_1", type=float, default=0.2, help="BS weight.")
    base_parser.add_argument("--omega_weight_2", type=float, default=0.4, help="UAV weight.")
    base_parser.add_argument("--omega_weight_3", type=float, default=0.4, help="CU weight.")

    # Radar parameters
    base_parser.add_argument("--radar_duty_ratio", type=float, default=0.01, help="Radar duty ratio.")
    base_parser.add_argument("--var_range_fluctuation", type=float, default=1e-14, help="Variance of range fluctuation.")
    base_parser.add_argument("--radar_impulse_duration", type=float, default=2e-5, help="Radar impulse duration.")
    base_parser.add_argument("--radar_spectrum_shape", type=float, default=pi / sqrt(3), help="Radar spectrum shape parameter.")

    # CCCP parameters
    base_parser.add_argument("--max_iterations", type=int, default=5, help="Maximum CCCP iterations.")
    base_parser.add_argument("--cccp_threshold", type=float, default=1e-6, help="CCCP convergence threshold.")
    base_parser.add_argument("--rank1_threshold", type=float, default=1e-6, help="Rank-one convergence threshold.")
    base_parser.add_argument("--penalty_factor", type=float, default=0.1, help="Penalty factor.")
    base_parser.add_argument("--zoom_factor", type=float, default=2, help="Penalty zoom factor.")
    base_parser.add_argument(
        "--enable_cccp_diagnostics",
        type=str_to_bool,
        default=False,
        help="Enable CCCP diagnostics. Default keeps the current lightweight behavior.",
    )
    base_parser.add_argument("--diagnostic_violation_tol", type=float, default=1e-7, help="Tolerance used in diagnostics.")
    base_parser.add_argument("--diagnostic_top_k", type=int, default=5, help="Maximum number of diagnostic violations to print.")
    base_parser.add_argument(
        "--constraint_include_groups",
        type=str,
        default="4.5,4.12,4.23,4.25,4.27,4.28,4.29,4.32,4.39,4.40,4.44,4.45,auxiliary_t,var",
        help="Constraint groups to include, separated by commas.",
    )
    base_parser.add_argument(
        "--constraint_exclude_groups",
        type=str,
        default="",
        help="Constraint groups to exclude, separated by commas.",
    )
    base_parser.add_argument(
        "--linearization_psi_floor",
        type=float,
        default=1e-8,
        help="Lower bound of Psi used in linearization.",
    )
    base_parser.add_argument(
        "--enable_first_iter_rank_boost",
        type=str_to_bool,
        default=True,
        help="Enable the first-iteration rank boost.",
    )
    base_parser.add_argument("--first_iter_rank_boost_eps", type=float, default=0.1, help="Rank boost strength.")
    base_parser.add_argument(
        "--solver_backend",
        type=str,
        default="fusion",
        choices=["fusion", "cvxpy"],
        help="Backend used by the relaxed inner problem solver.",
    )

    return base_parser.parse_args()


def build_output_csv_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, f"{timestamp}_compare_penalty_gaussian_based_obj_val.csv")


def save_case_results_to_csv(case_results, csv_path):
    fieldnames = [
        "case_id",
        "penalty_based_obj",
        "penalty_based_iter_count",
        "gaussian_based_obj",
        "gaussian_based_iter_count",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_results)


if __name__ == "__main__":
    args = get_base_args()
    np.random.seed(args.seed)

    from gaussian_based_cccp_algorithm import gaussian_based_cccp
    from penalty_based_cccp_algorithm import (
        compute_com_channel_gain,
        compute_sen_channel_gain,
        compute_uav_pos_cur,
        db_2_watt,
        dbm_2_watt,
        generate_pos,
        match_uav_targets_nearest,
        penalty_based_cccp,
        random_choose_matched_matrix,
    )

    penalty_based_obj_list = []
    gaussian_based_obj_list = []
    case_results = []

    for i in range(args.num_cases):
        print(f"========== Case {i + 1}/{args.num_cases} ==========")

        uavs_pos, cus_pos, targets_pos = generate_pos(
            uavs_num=args.uavs_num,
            cus_num=args.cus_num,
            targets_num=args.targets_num,
            center=(0, 0),
            radius=args.radius,
            uav_height=args.uav_height,
        )

        uavs_2_cus_channels, uavs_2_bs_channels, cus_2_bs_channels = compute_com_channel_gain(
            uavs_pos=uavs_pos,
            cus_pos=cus_pos,
            ref_path_loss=db_2_watt(args.ref_path_loss_db),
            frac_d_lambda=args.frac_d_lambda,
            alpha_uav_link=args.alpha_uav_link,
            alpha_cu_link=args.alpha_cu_link,
            rician_factor=db_2_watt(args.rician_factor_db),
            antenna_nums=args.antenna_nums,
        )

        uavs_2_targets_channels = compute_sen_channel_gain(
            radar_rcs=args.radar_rcs,
            frac_d_lambda=args.frac_d_lambda,
            uavs_pos=uavs_pos,
            targets_pos=targets_pos,
            antenna_nums=args.antenna_nums,
            ref_path_loss=db_2_watt(args.ref_path_loss_db),
        )

        uavs_cus_matched_matrix = random_choose_matched_matrix(uavs_pos=uavs_pos, cus_pos=cus_pos)
        uavs_targets_matched_matrix = match_uav_targets_nearest(uavs_pos=uavs_pos, targets_pos=targets_pos)
        uavs_pos_cur = compute_uav_pos_cur(args=args, uavs_pos_pre=uavs_pos)

        uavs_off_duration = np.full(args.uavs_num, (args.uav_max_delay - args.uav_sen_duration) * 0.8)
        cus_off_power = np.full(args.cus_num, dbm_2_watt(args.cu_max_power_dbm) / 5)
        cus_entertaining_task_size = np.random.uniform(140e3, 200e3, args.cus_num)

        penalty_obj_opt, penalty_iter_count, penalty_energy_history, penalty_rank1_history = penalty_based_cccp(
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
            cus_entertaining_task_size=cus_entertaining_task_size,
        )

        gaussian_obj_opt, gaussian_iter_count, gaussian_energy_history, gaussian_rank1_history = gaussian_based_cccp(
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
            cus_entertaining_task_size=cus_entertaining_task_size,
        )

        penalty_based_obj_list.append(float(penalty_obj_opt))
        gaussian_based_obj_list.append(float(gaussian_obj_opt))
        case_results.append(
            {
                "case_id": i + 1,
                "penalty_based_obj": float(penalty_obj_opt),
                "penalty_based_iter_count": int(penalty_iter_count),
                "gaussian_based_obj": float(gaussian_obj_opt),
                "gaussian_based_iter_count": int(gaussian_iter_count),
            }
        )

        print(f"Penalty-based objective: {float(penalty_obj_opt)}")
        print(f"Gaussian-based objective: {float(gaussian_obj_opt)}")
        print(f"Penalty history length: {len(penalty_energy_history)}, rank1 length: {len(penalty_rank1_history)}")
        print(f"Gaussian history length: {len(gaussian_energy_history)}, rank1 length: {len(gaussian_rank1_history)}")

    output_csv_path = build_output_csv_path()
    save_case_results_to_csv(case_results=case_results, csv_path=output_csv_path)

    print("========================================================")
    print("Comparison finished.")
    print(f"Penalty-based results list: {penalty_based_obj_list}")
    print(f"Gaussian-based results list: {gaussian_based_obj_list}")
    print(f"Case result CSV saved to: {output_csv_path}")
