from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

OUTPUT_DIR = RESULTS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. CONTROLLER TELEMETRY FILES
# ==================================================

FILES = {
    "Baseline": "baseline_carla_8ms_telemetry.csv",
    "Stanley": "stanley_carla_extended_8ms_telemetry.csv",
    "Pure Pursuit": "pure_pursuit_carla_8ms_telemetry.csv",
    "Augmented LQR": "augmented_lqr_carla_8ms_telemetry.csv",
    "MPC": "mpc_carla_8ms_telemetry.csv",
}


# ==================================================
# 3. LOAD CSV DATA
# ==================================================

def load_telemetry(filename: str) -> dict:

    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Telemetry file not found: {path}"
        )

    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(
            f"Telemetry file is empty: {filename}"
        )

    data = {}

    for column in reader.fieldnames or []:

        try:
            data[column] = np.array(
                [float(row[column]) for row in rows],
                dtype=float
            )

        except (ValueError, TypeError):
            # Skip non-numeric columns such as solver_status
            continue

    return data


telemetry = {
    name: load_telemetry(filename)
    for name, filename in FILES.items()
}


# ==================================================
# 4. COMMON COLUMN NAMES
# ==================================================

ROUTE_PROGRESS = "route_progress_m"

REAR_ERROR = "rear_axle_tracking_error_m"

FRONT_ERROR = "front_cross_track_error_m"

STEERING = "applied_steering_deg"

SPEED = "speed_mps"

TIME = "sim_time_s"


# ==================================================
# 5. STEERING-RATE COLUMN MAPPING
# ==================================================

STEERING_RATE_COLUMNS = {
    "Baseline": "applied_steering_rate_deg_s",
    "Stanley": "steering_rate_deg_s",
    "Pure Pursuit": "steering_rate_deg_s",
    "Augmented LQR": "applied_steering_rate_deg_s",
    "MPC": "applied_steering_rate_deg_s",
}


# ==================================================
# 6. ROAD SECTION MARKERS
# ==================================================

CURVE_START = 57.50
CURVE_END = 117.72
ROUTE_END = 135.24


def add_road_sections(ax):

    ax.axvspan(
        CURVE_START,
        CURVE_END,
        color="lightgray",
        alpha=0.25,
        label="Sustained Curve"
    )

    ax.axvline(
        CURVE_START,
        color="gray",
        linestyle="--",
        linewidth=1
    )

    ax.axvline(
        CURVE_END,
        color="gray",
        linestyle="--",
        linewidth=1
    )

    ax.set_xlim(0, ROUTE_END)


# ==================================================
# 7. GENERIC ROUTE-DISTANCE PLOT
# ==================================================

def plot_route_metric(
    column: str,
    title: str,
    ylabel: str,
    filename: str,
    column_mapping: dict | None = None
) -> None:

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, data in telemetry.items():

        selected_column = (
            column_mapping[name]
            if column_mapping is not None
            else column
        )

        if selected_column not in data:
            raise KeyError(
                f"Missing column '{selected_column}' "
                f"for {name}"
            )

        ax.plot(
            data[ROUTE_PROGRESS],
            data[selected_column],
            linewidth=1.8,
            label=name
        )

    add_road_sections(ax)

    ax.set_xlabel("Route Distance (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")


# ==================================================
# 8. REAR-AXLE TRACKING ERROR
# ==================================================

plot_route_metric(
    column=REAR_ERROR,
    title="Five-Controller Rear-Axle Tracking Error",
    ylabel="Signed Rear-Axle Error (m)",
    filename="final_rear_axle_error_comparison.png"
)


# ==================================================
# 9. FRONT-AXLE CROSS-TRACK ERROR
# ==================================================

plot_route_metric(
    column=FRONT_ERROR,
    title="Five-Controller Front-Axle Cross-Track Error",
    ylabel="Signed Front-Axle CTE (m)",
    filename="final_front_axle_error_comparison.png"
)


# ==================================================
# 10. APPLIED STEERING ANGLE
# ==================================================

plot_route_metric(
    column=STEERING,
    title="Five-Controller Steering Angle Comparison",
    ylabel="Applied Steering Angle (degrees)",
    filename="final_steering_angle_comparison.png"
)


# ==================================================
# 11. APPLIED STEERING RATE
# ==================================================

plot_route_metric(
    column="",
    title="Five-Controller Steering Rate Comparison",
    ylabel="Applied Steering Rate (degrees/s)",
    filename="final_steering_rate_comparison.png",
    column_mapping=STEERING_RATE_COLUMNS
)


# ==================================================
# 12. VEHICLE SPEED VS TIME
# ==================================================

fig, ax = plt.subplots(figsize=(12, 6))

for name, data in telemetry.items():

    ax.plot(
        data[TIME],
        data[SPEED],
        linewidth=1.8,
        label=name
    )

ax.axhline(
    y=8.0,
    color="black",
    linestyle="--",
    label="Target Speed"
)

ax.set_xlabel("Simulation Time (s)")
ax.set_ylabel("Vehicle Speed (m/s)")
ax.set_title("Five-Controller Vehicle Speed Comparison")

ax.grid(True)
ax.legend()

plt.tight_layout()

output_path = (
    OUTPUT_DIR / "final_speed_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Saved: {output_path}")


# ==================================================
# 13. DISPLAY ALL PLOTS
# ==================================================

plt.show()

print("\nTelemetry comparison completed successfully.")