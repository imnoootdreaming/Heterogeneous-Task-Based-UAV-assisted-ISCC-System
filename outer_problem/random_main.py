import argparse
import csv
import importlib.util
import os
import sys
import types
import pandas as pd
from contextlib import contextmanager
from datetime import datetime

import numpy as np
from tqdm import tqdm


def _install_gym_fallback():
    if "gym" in sys.modules or "gymnasium" in sys.modules:
        return

    class _Env:
        pass

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.dtype = np.dtype(dtype)
            if shape is not None:
                if np.isscalar(low):
                    self.low = np.full(shape, low, dtype=self.dtype)
                else:
                    self.low = np.asarray(low, dtype=self.dtype)
                if np.isscalar(high):
                    self.high = np.full(shape, high, dtype=self.dtype)
                else:
                    self.high = np.asarray(high, dtype=self.dtype)
                self.shape = tuple(shape)
            else:
                self.low = np.asarray(low, dtype=self.dtype)
                self.high = np.asarray(high, dtype=self.dtype)
                self.shape = self.low.shape

    class _MultiDiscrete:
        def __init__(self, nvec):
            self.nvec = np.asarray(nvec, dtype=np.int64)
            self.shape = self.nvec.shape

    spaces_module = types.SimpleNamespace(Box=_Box, MultiDiscrete=_MultiDiscrete)
    gym_module = types.ModuleType("gym")
    gym_module.Env = _Env
    gym_module.spaces = spaces_module
    gymnasium_module = types.ModuleType("gymnasium")
    gymnasium_module.Env = _Env
    gymnasium_module.spaces = spaces_module

    sys.modules["gym"] = gym_module
    sys.modules["gymnasium"] = gymnasium_module


_install_gym_fallback()

try:
    from my_env import MyEnv
except Exception as exc:
    raise RuntimeError(
        "Failed to import MyEnv. This baseline relies on the same environment and reward stack as the "
        "training scripts, so the local Python environment must also provide compatible dependencies for "
        "gym/gymnasium and the my_reward.py solver stack (for example cvxpy/scipy/matplotlib built against "
        "the installed NumPy version)."
    ) from exc


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


def sample_random_action(env, rng):
    continuous_shape = env.action_space["bs"]["continuous"].shape
    # `env.step()` expects normalized continuous actions in [0, 1].
    continuous_action = rng.uniform(0.0, 1.0, size=continuous_shape).astype(np.float32)

    discrete_dims = env.action_space["bs"]["discrete"].nvec
    discrete_action = np.array(
        [rng.integers(low=0, high=int(action_dim)) for action_dim in discrete_dims],
        dtype=np.int64,
    )

    return {
        "continuous": continuous_action,
        "discrete": discrete_action,
    }


if __name__ == "__main__":
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_args, madrl_args = get_all_args()
    setSeed(seed=base_args.seed)
    rng = np.random.default_rng(base_args.seed)

    env = MyEnv(base_args=base_args, madrl_args=madrl_args)

    reward_res = []

    with tqdm(total=int(madrl_args.episodes), desc="Random Baseline Progress") as pbar:
        for i_episode in range(int(madrl_args.episodes)):
            episode_rewards_total = []
            env.reset()
            terminal = False

            while not terminal:
                random_action_bs = sample_random_action(env, rng)
                _, total_reward, _, done, _, _ = env.step({"bs": random_action_bs}, i_episode)
                episode_rewards_total.append(float(total_reward))
                terminal = done

            avg_total_reward = (
                float(np.mean(episode_rewards_total))
                if len(episode_rewards_total) > 0
                else 0.0
            )
            reward_res.append(avg_total_reward)

            pbar.set_postfix({"avg_reward": f"{avg_total_reward:.3f}"})
            pbar.update(1)

    reward_array = np.asarray(reward_res, dtype=np.float32)
    episodes_list = np.arange(reward_array.shape[0], dtype=np.int64)

    filename = f"{current_time_str}_random_rewards_seed_{base_args.seed}.csv"
    df = pd.DataFrame({"episode": episodes_list, "reward": reward_array})
    df.to_csv(filename, index=False)
    print(f"HSAC training results saved to {filename}")


