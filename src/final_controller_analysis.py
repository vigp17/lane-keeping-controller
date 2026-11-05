from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. RESULTS DIRECTORY
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. CONTROLLER NAMES
# ==========================================

controllers = [
    "Baseline",
    "Stanley",
    "Pure Pursuit",
    "Augmented LQR",
    "MPC"
]


# ==========================================
# 3. MEASURED CARLA RESULTS
# ==========================================

# Values transcribed from completed
# CARLA experiment reports.

rear_rmse = np.array([
    0.2847,
    0.0488,
    0.1460,
    0.1873,
    0.0664
])

front_rmse = np.array([
    0.3276,
    0.0915,
    0.1871,
    0.2315,
    0.1116
])

steering_rms = np.array([
    3.4049,
    3.3415,
    3.3156,
    3.3987,
    3.4207
])

steering_rate_rms = np.array([
    2.8547,
    4.4192,
    1.5476,
    2.4933,
    2.6930
])


# ==========================================
# 4. CREATE COMPARISON FIGURE
# ==========================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


# Rear-axle RMSE
axes[0, 0].bar(
    controllers,
    rear_rmse
)

axes[0, 0].set_title(
    "Rear-Axle Tracking RMSE"
)

axes[0, 0].set_ylabel(
    "RMSE (m)"
)


# Front-axle RMSE
axes[0, 1].bar(
    controllers,
    front_rmse
)

axes[0, 1].set_title(
    "Front-Axle Cross-Track RMSE"
)

axes[0, 1].set_ylabel(
    "RMSE (m)"
)


# Steering RMS
axes[1, 0].bar(
    controllers,
    steering_rms
)

axes[1, 0].set_title(
    "Steering Effort"
)

axes[1, 0].set_ylabel(
    "Steering RMS (degrees)"
)


# Steering-rate RMS
axes[1, 1].bar(
    controllers,
    steering_rate_rms
)

axes[1, 1].set_title(
    "Steering Smoothness"
)

axes[1, 1].set_ylabel(
    "Steering Rate RMS (degrees/s)"
)


# ==========================================
# 5. FORMAT GRAPHS
# ==========================================

for ax in axes.flat:

    ax.tick_params(
        axis="x",
        rotation=25
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )


fig.suptitle(
    "Five-Controller CARLA Performance Comparison",
    fontsize=16
)

plt.tight_layout()


# ==========================================
# 6. SAVE FIGURE
# ==========================================

output_path = (
    RESULTS_DIR
    / "final_controller_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"Comparison saved to: {output_path}"
)

plt.show()