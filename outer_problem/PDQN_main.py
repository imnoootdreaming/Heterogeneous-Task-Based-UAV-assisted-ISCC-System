import argparse
from contextlib import contextmanager
import importlib.util
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from PDQN_agent import PDQN, ReplayBuffer
from my_env import MyEnv
from utils.normalization import Normalization


def _load_beta_main_helpers():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    beta_main_path = os.path.join(current_dir, "beta-HPPO_main.py")
    spec = importlib.util.spec_from_file_location("beta_hppo_main_helpers", beta_main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_beta_main_helpers = _load_beta_main_helpers()
setSeed = _beta_main_helpers.setSeed
get_tensorboard_writer = _beta_main_helpers.get_tensorboard_writer
get_madrl_args = _beta_main_helpers.get_madrl_args
get_base_args = _beta_main_helpers.get_base_args


@contextmanager
def _ignore_unknown_cli_args():
    original_parse_args = argparse.ArgumentParser.parse_args

    def _parse_args_ignore_unknown(self, args=None, namespace=None):
        parsed_args, _ = self.parse_known_args(args=args, namespace=namespace)
        return parsed_args

    argparse.ArgumentParser.parse_args = _parse_args_ignore_unknown
    try:
        yield
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args


def get_pdqn_args():
    pdqn_parser = argparse.ArgumentParser(description="PDQN unique arguments")
    pdqn_parser.add_argument("--tau", type=float, default=5e-3, help="Target network soft update factor")
    pdqn_parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer capacity")
    pdqn_parser.add_argument("--minimal_size", type=int, default=1024, help="Minimum samples before updates")
    pdqn_parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    pdqn_parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon")
    pdqn_parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final epsilon")
    pdqn_parser.add_argument("--epsilon_decay_steps", type=int, default=50000, help="Epsilon decay steps")
    pdqn_parser.add_argument("--updates_per_step", type=int, default=1, help="Number of updates per environment step")
    return pdqn_parser.parse_args()


def get_all_args():
    with _ignore_unknown_cli_args():
        base_args = get_base_args()
        madrl_args = get_madrl_args()
        pdqn_args = get_pdqn_args()

    for key, value in vars(pdqn_args).items():
        setattr(madrl_args, key, value)
    return base_args, madrl_args


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = get_tensorboard_writer(log_dir=f"runs/pdqn/{current_time_str}_pdqn_result")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args, madrl_args = get_all_args()
    setSeed(seed=base_args.seed)

    env = MyEnv(base_args=base_args, madrl_args=madrl_args)

    state_dim_bs = env.observation_space["bs"].shape[0]
    bs_continuous_action_space = env.action_space["bs"]["continuous"]
    bs_discrete_action_space = env.action_space["bs"]["discrete"]
    continuous_action_dim = int(np.prod(bs_continuous_action_space.shape))

    agent_bs = PDQN(
        state_dim=state_dim_bs,
        hidden_dim=madrl_args.hidden_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_action_dims=bs_discrete_action_space.nvec,
        actor_lr=madrl_args.actor_lr,
        critic_lr=madrl_args.critic_lr,
        gamma=madrl_args.gamma,
        tau=madrl_args.tau,
        epsilon_start=madrl_args.epsilon_start,
        epsilon_end=madrl_args.epsilon_end,
        epsilon_decay_steps=madrl_args.epsilon_decay_steps,
        device=device,
    )
    replay_buffer = ReplayBuffer(madrl_args.buffer_size)
    running_norm_bs = Normalization(state_dim_bs)

    reward_res = []
    all_agents_rewards = []
    max_avg_reward = -np.inf
    best_uav_trajectory = None
    cu_trajectory = None
    target_trajectory = None
    global_step = 0

    with tqdm(total=int(madrl_args.episodes), desc="Training Progress") as pbar:
        for i_episode in range(int(madrl_args.episodes)):
            episode_rewards_total = []
            episode_reward_bs = 0.0

            s = env.reset()
            terminal = False
            uav_positions_episode = []
            cu_positions_episode = []
            target_positions_episode = []

            while not terminal:
                uav_positions_episode.append(env.getPosUAV())
                cu_positions_episode.append(env.getPosCU())
                target_positions_episode.append(env.getPosTarget())

                state_bs_norm = running_norm_bs(np.array(s["bs"], dtype=np.float32))
                action_bs = agent_bs.choose_action(state_bs_norm)

                next_s, total_reward, r_dict, done = env.step(
                    {"bs": {"continuous": action_bs["continuous"], "discrete": action_bs["discrete"]}},
                    i_episode
                )

                r_bs = float(r_dict["bs"])
                next_state_bs_norm = running_norm_bs(np.array(next_s["bs"], dtype=np.float32))

                replay_buffer.add(
                    state=state_bs_norm,
                    continuous_action=action_bs["continuous"],
                    discrete_index=action_bs["discrete_index"],
                    reward=r_bs,
                    next_state=next_state_bs_norm,
                    done=bool(done),
                )

                if replay_buffer.size() >= madrl_args.minimal_size:
                    for _ in range(madrl_args.updates_per_step):
                        transition_dict_bs = replay_buffer.sample(madrl_args.batch_size)
                        agent_bs.update(transition_dict_bs, writer=writer, step=global_step, agent_name="BS")

                episode_rewards_total.append(float(total_reward))
                episode_reward_bs += r_bs

                writer.add_scalar("BS/Epsilon", action_bs["epsilon"], global_step)
                s = next_s
                terminal = done
                global_step += 1

            if np.mean(episode_rewards_total) > max_avg_reward:
                max_avg_reward = np.mean(episode_rewards_total)
                if len(uav_positions_episode) > 0:
                    best_uav_trajectory = np.array(uav_positions_episode, copy=True)
                if len(cu_positions_episode) > 0:
                    cu_trajectory = np.array(cu_positions_episode, copy=True)
                if len(target_positions_episode) > 0:
                    target_trajectory = np.array(target_positions_episode, copy=True)

            avg_total_reward = np.mean(episode_rewards_total)
            avg_bs_reward = episode_reward_bs / max(len(episode_rewards_total), 1)

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
    plt.title("Single-Agent PDQN training performance")
    plt.show()

    filename = f"{current_time_str}_PDQN_training_rewards_seed_{base_args.seed}.csv"
    df = pd.DataFrame({"episode": episodes_list, "reward": reward_array})
    df.to_csv(filename, index=False)
    print(f"PDQN training results saved to {filename}")

    all_agents_rewards = np.array(all_agents_rewards)
    df_agents = pd.DataFrame(all_agents_rewards, columns=["BS_reward"])
    df_agents.insert(0, "episode", np.arange(madrl_args.episodes))
    filename_agents = f"{current_time_str}_PDQN_all_agents_rewards_seed_{base_args.seed}.csv"
    df_agents.to_csv(filename_agents, index=False)
    print(f"PDQN BS agent rewards saved to {filename_agents}")

    if best_uav_trajectory is not None:
        uav_traj_list = []
        for t in range(best_uav_trajectory.shape[0]):
            for uav_i in range(base_args.uavs_num):
                x, y, z = best_uav_trajectory[t, uav_i]
                uav_traj_list.append([t, uav_i, x, y, z])
        df_uav = pd.DataFrame(uav_traj_list, columns=["time_slot", "uav_id", "x", "y", "z"])
        df_uav.to_csv(f"{current_time_str}_SEED{base_args.seed}_pdqn_best_uav_trajectory.csv", index=False)

    if cu_trajectory is not None:
        cu_traj_list = []
        for t in range(cu_trajectory.shape[0]):
            for cu_i in range(base_args.cus_num):
                x, y, z = cu_trajectory[t, cu_i]
                cu_traj_list.append([t, cu_i, x, y, z])
        df_cu = pd.DataFrame(cu_traj_list, columns=["time_slot", "cu_id", "x", "y", "z"])
        df_cu.to_csv(f"{current_time_str}_SEED{base_args.seed}_pdqn_cu_trajectory.csv", index=False)

    if target_trajectory is not None:
        target_traj_list = []
        for t in range(target_trajectory.shape[0]):
            for target_i in range(base_args.targets_num):
                x, y, z = target_trajectory[t, target_i]
                target_traj_list.append([t, target_i, x, y, z])
        df_target = pd.DataFrame(target_traj_list, columns=["time_slot", "target_id", "x", "y", "z"])
        df_target.to_csv(f"{current_time_str}_SEED{base_args.seed}_pdqn_target_trajectory.csv", index=False)
