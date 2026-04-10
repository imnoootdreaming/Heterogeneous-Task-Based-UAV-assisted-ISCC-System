import argparse
import csv
import importlib.util
import os
import sys
import types
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


def save_reward_csv(csv_path, episodes, rewards):
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "reward"])
        for episode, reward in zip(episodes, rewards):
            writer.writerow([int(episode), float(reward)])


def _format_tick_label(value):
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.2f}"


def save_reward_curve_svg(svg_path, episodes, rewards):
    width = 900
    height = 540
    margin_left = 90
    margin_right = 30
    margin_top = 40
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = np.asarray(episodes, dtype=np.float64)
    y_values = np.asarray(rewards, dtype=np.float64)

    x_min = float(np.min(x_values)) if x_values.size > 0 else 0.0
    x_max = float(np.max(x_values)) if x_values.size > 0 else 1.0
    y_min = float(np.min(y_values)) if y_values.size > 0 else 0.0
    y_max = float(np.max(y_values)) if y_values.size > 0 else 1.0

    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-12:
        y_padding = max(abs(y_min) * 0.1, 1.0)
        y_min -= y_padding
        y_max += y_padding

    def map_x(value):
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def map_y(value):
        return margin_top + (1.0 - (value - y_min) / (y_max - y_min)) * plot_height

    x_tick_count = int(min(6, max(2, x_values.size if x_values.size > 0 else 2)))
    y_tick_count = 5
    x_ticks = np.linspace(x_min, x_max, x_tick_count)
    y_ticks = np.linspace(y_min, y_max, y_tick_count)

    grid_lines = []
    tick_labels = []

    for tick in x_ticks:
        x_pos = map_x(tick)
        grid_lines.append(
            f'<line x1="{x_pos:.2f}" y1="{margin_top}" x2="{x_pos:.2f}" y2="{height - margin_bottom}" '
            'stroke="#e0e0e0" stroke-dasharray="4 4" stroke-width="1"/>'
        )
        tick_labels.append(
            f'<text x="{x_pos:.2f}" y="{height - margin_bottom + 28}" text-anchor="middle" '
            f'font-size="14" fill="#333333">{_format_tick_label(tick)}</text>'
        )

    for tick in y_ticks:
        y_pos = map_y(tick)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y_pos:.2f}" x2="{width - margin_right}" y2="{y_pos:.2f}" '
            'stroke="#e0e0e0" stroke-dasharray="4 4" stroke-width="1"/>'
        )
        tick_labels.append(
            f'<text x="{margin_left - 12}" y="{y_pos + 5:.2f}" text-anchor="end" '
            f'font-size="14" fill="#333333">{_format_tick_label(tick)}</text>'
        )

    polyline_points = " ".join(
        f"{map_x(x_value):.2f},{map_y(y_value):.2f}"
        for x_value, y_value in zip(x_values, y_values)
    )

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.2f}" y="24" text-anchor="middle" font-size="22" fill="#222222">Random policy performance</text>
{''.join(grid_lines)}
<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#444444" stroke-width="2"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#444444" stroke-width="2"/>
{''.join(tick_labels)}
<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{polyline_points}"/>
<text x="{width / 2:.2f}" y="{height - 18}" text-anchor="middle" font-size="18" fill="#222222">Episodes</text>
<text x="24" y="{height / 2:.2f}" text-anchor="middle" font-size="18" fill="#222222" transform="rotate(-90 24 {height / 2:.2f})">Reward</text>
</svg>
"""

    with open(svg_path, "w", encoding="utf-8") as svg_file:
        svg_file.write(svg_content)


def try_save_reward_curve_with_matplotlib(plot_path, episodes, rewards):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return False, str(exc)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episodes, rewards, color="#1f77b4", linewidth=1.8)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.set_title("Random policy performance")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return True, None


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
                _, total_reward, _, done = env.step({"bs": random_action_bs}, i_episode)
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

    rewards_csv_path = f"{current_time_str}_random_rewards_seed_{base_args.seed}.csv"
    reward_curve_png_path = f"{current_time_str}_random_reward_curve_seed_{base_args.seed}.png"
    reward_curve_svg_path = f"{current_time_str}_random_reward_curve_seed_{base_args.seed}.svg"

    save_reward_csv(
        csv_path=rewards_csv_path,
        episodes=episodes_list,
        rewards=reward_array,
    )

    plot_saved_with_matplotlib, plot_error = try_save_reward_curve_with_matplotlib(
        plot_path=reward_curve_png_path,
        episodes=episodes_list,
        rewards=reward_array,
    )

    if plot_saved_with_matplotlib:
        plot_path = reward_curve_png_path
    else:
        save_reward_curve_svg(
            svg_path=reward_curve_svg_path,
            episodes=episodes_list,
            rewards=reward_array,
        )
        plot_path = reward_curve_svg_path

    print(f"Random baseline rewards saved to {rewards_csv_path}")
    print(f"Random baseline reward curve saved to {plot_path}")
    if plot_error is not None:
        print("Matplotlib is unavailable, so the script exported an SVG curve instead.")
        print(f"Matplotlib error: {plot_error}")
