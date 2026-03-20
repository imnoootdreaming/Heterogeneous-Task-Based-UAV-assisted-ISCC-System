import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

RANK1_COLOR = "#4169E1"
ENERGY_COLOR = "#FF6B6B"
GRID_COLOR = "#E0E0E0"

base_dir = os.path.dirname(os.path.abspath(__file__))
energy_path = os.path.join(base_dir, "energy_val_list_rho5.csv")
rank1_path = os.path.join(base_dir, "rank1_val_list_rho5.csv")

energy_values = np.atleast_1d(np.loadtxt(energy_path, delimiter=","))
rank1_values = np.atleast_1d(np.loadtxt(rank1_path, delimiter=","))

iters_energy = np.arange(1, energy_values.size + 1)
iters_rank1 = np.arange(1, rank1_values.size + 1)

fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()

line1 = ax1.plot(
    iters_energy,
    energy_values,
    marker="s",
    linestyle="-",
    markersize=8,
    label="Energy value",
    color=ENERGY_COLOR,
    linewidth=2,
    markerfacecolor="white",
    markeredgewidth=1.5,
)
line2 = ax2.plot(
    iters_rank1,
    rank1_values,
    marker="o",
    linestyle="--",
    markersize=8,
    label="Rank-1 gap",
    color=RANK1_COLOR,
    linewidth=2,
    markerfacecolor="white",
    markeredgewidth=1.5,
)

ax1.set_xlabel("The number of iteration", fontsize=24)
ax1.set_ylabel("The minimum weighted total energy consumption (J)", fontsize=24, color=ENERGY_COLOR)
ax2.set_ylabel("Rank-1 gap", fontsize=24, color=RANK1_COLOR)
ax1.spines["left"].set_color(ENERGY_COLOR)
ax2.spines["right"].set_color(RANK1_COLOR)

ax1.tick_params(axis="y", which="major", labelsize=18, colors=ENERGY_COLOR)
ax2.tick_params(axis="y", which="major", labelsize=18, colors=RANK1_COLOR)

ax1.spines['left'].set_color(ENERGY_COLOR)
ax1.spines['left'].set_linewidth(1.6)
ax2.spines['right'].set_color(RANK1_COLOR)
ax2.spines['right'].set_linewidth(1.6)
ax2.spines['left'].set_visible(False)
ax1.spines['top'].set_color('#E0E0E0')
ax1.spines['bottom'].set_color('#E0E0E0')

# x轴保持黑色（论文规范）
ax1.tick_params(axis="x", which="major", labelsize=18)
ax1.grid(
    True,
    linestyle=(0, (3, 5)),  # 短虚线
    color=GRID_COLOR,
    linewidth=1.0,
    alpha=1.0,
    zorder=1
)
lines = line1 + line2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, fontsize=24)

plt.tight_layout()
plt.show()
