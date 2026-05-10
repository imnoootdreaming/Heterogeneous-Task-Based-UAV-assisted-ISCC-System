import os
import importlib.util
import argparse
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from HPPO_agent import HPPO
from my_env import MyEnv
from utils.normalization import Normalization, RewardScaling


def _load_beta_main_helpers():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    beta_main_path = os.path.join(current_dir, "beta-HPPO_main.py")
    spec = importlib.util.spec_from_file_location("beta_hppo_main_helpers", beta_main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_beta_main_helpers = _load_beta_main_helpers()
setSeed = _beta_main_helpers.setSeed
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


def get_all_args():
    with _ignore_unknown_cli_args():
        base_args = get_base_args()
        madrl_args = get_madrl_args()
    return base_args, madrl_args


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_args, madrl_args = get_all_args()

    test_seeds = list(range(10))

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint", "checkpoint_mhgppo")
    actor_path = os.path.join(checkpoint_dir, "mhgppo_actor.pth")
    critic_path = os.path.join(checkpoint_dir, "mhgppo_critic.pth")

    all_results = []

    for test_seed in tqdm(test_seeds, desc="Testing Seeds"):
        setSeed(seed=test_seed)
        base_args.seed = test_seed

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
            entropy_coef=0.01,
            continuous_dist_type="gaussian",
        )

        agent_bs.actor.load_state_dict(torch.load(actor_path, map_location=device))
        agent_bs.critic.load_state_dict(torch.load(critic_path, map_location=device))
        agent_bs.actor.eval()
        agent_bs.critic.eval()

        running_norm_bs = Normalization(state_dim_bs)
        reward_scaler_bs = RewardScaling(shape=1, gamma=madrl_args.gamma)

        # warm-up episodes with diverse action paths to initialize Normalization statistics
        for warm_idx in range(3):
            torch.manual_seed(test_seed * 1000 + warm_idx)
            s_warm = env.reset()
            terminal_warm = False
            while not terminal_warm:
                state_bs_norm_warm = running_norm_bs(np.array(s_warm["bs"], dtype=np.float32))
                action_bs_warm, _, _, _ = agent_bs.choose_action(state_bs_norm_warm)
                next_s_warm, _, _, done_warm = env.step({"bs": action_bs_warm}, i_episode=test_seed)
                s_warm = next_s_warm
                terminal_warm = done_warm

        episode_rewards_total = []
        episode_reward_bs = 0.0

        s = env.reset()
        terminal = False
        reward_scaler_bs.reset()

        while not terminal:
            state_bs_norm = running_norm_bs(np.array(s["bs"], dtype=np.float32), update=False)
            action_bs, _, _, _ = agent_bs.choose_action(state_bs_norm)

            next_s, total_reward, r_dict, done = env.step({"bs": action_bs}, i_episode=test_seed)

            r_bs = float(r_dict["bs"])
            r_bs_norm = float(np.asarray(reward_scaler_bs(r_bs)).item())

            episode_rewards_total.append(float(total_reward))
            episode_reward_bs += r_bs

            s = next_s
            terminal = done

        avg_total_reward = np.mean(episode_rewards_total)
        avg_bs_reward = episode_reward_bs / len(episode_rewards_total)

        all_results.append({
            "seed": test_seed,
            "avg_total_reward": avg_total_reward,
            "avg_bs_reward": avg_bs_reward,
        })

    df = pd.DataFrame(all_results)
    filename = f"{current_time_str}_Gaussian_HPPO_test_results.csv"
    df.to_csv(filename, index=False)
    print(f"Test results saved to {filename}")
    print(df.to_string(index=False))
