from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from controllers.stanley_controller import StanleyController
from road import Road
from vehicle import Vehicle


# ============================================================
# 1. SIMULATION CONFIGURATION
# ============================================================

WHEELBASE: float = 2.8
DT: float = 0.1
SIMULATION_TIME: float = 10.0
STEPS: int = int(SIMULATION_TIME / DT)

INITIAL_X: float = 0.0
INITIAL_LATERAL_OFFSET: float = 2.0
INITIAL_YAW: float = 0.0

STANLEY_GAIN: float = 1.5
STANLEY_EPSILON: float = 1e-3

MAX_STEERING: float = float(np.radians(30.0))
MAX_STEERING_RATE: float = float(np.radians(20.0))

TEST_SPEEDS: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)
DEFAULT_PLOT_SPEED: float = 20.0

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. RUN STANLEY SIMULATION
# ============================================================

def run_stanley_simulation(
    road_type: str = "curved",
    velocity: float = DEFAULT_PLOT_SPEED,
    stanley_gain: float = STANLEY_GAIN,
    epsilon: float = STANLEY_EPSILON,
    initial_lateral_offset: float = INITIAL_LATERAL_OFFSET,
    initial_yaw: float = INITIAL_YAW,
    dt: float = DT,
    simulation_time: float = SIMULATION_TIME,
    wheelbase: float = WHEELBASE
) -> dict[str, NDArray[np.float64]]:
    """
    Execute a closed-loop Stanley lane-keeping simulation.

    Args:
        road_type: Either "straight" or "curved".
        velocity: Nonnegative forward vehicle speed (m/s).
        stanley_gain: Stanley cross-track proportional gain k.
        epsilon: Low-speed regularization constant.
        initial_lateral_offset: Initial lateral offset relative to the
            road reference at x = 0 (m).
        initial_yaw: Initial vehicle heading angle (rad).
        dt: Simulation timestep (s).
        simulation_time: Total simulation duration (s).
        wheelbase: Vehicle wheelbase L (m).

    Returns:
        Dictionary of simulation time-series arrays (float64).
    """
    if road_type not in ("straight", "curved"):
        raise ValueError(
            f"Unknown road_type '{road_type}'. Expected 'straight' or 'curved'."
        )

    steps = int(round(simulation_time / dt))

    # Ensure the reference path extends well beyond the travel distance
    road_length = max(120.0, velocity * simulation_time + 60.0)
    num_waypoints = int(road_length * 10) + 1

    road = Road(length=road_length)

    if road_type == "straight":
        road_x, road_y = road.generate_straight_road(
            num_points=num_waypoints
        )
        road_yaw = np.zeros_like(road_x)
        initial_y_ref = 0.0
    else:
        road_x, road_y = road.generate_curved_road(
            num_points=num_waypoints
        )
        road_yaw = road.reference_heading(road_x)
        initial_y_ref = road.reference_lateral_position(INITIAL_X)

    vehicle = Vehicle(
        x=INITIAL_X,
        y=initial_y_ref + initial_lateral_offset,
        yaw=initial_yaw,
        wheelbase=wheelbase
    )

    controller = StanleyController(
        stanley_gain=stanley_gain,
        epsilon=epsilon
    )

    applied_steering: float = 0.0

    # State histories (length = steps + 1)
    x_history: list[float] = [vehicle.x]
    y_history: list[float] = [vehicle.y]
    yaw_history: list[float] = [vehicle.yaw]

    front_x_0, front_y_0 = vehicle.get_front_axle_position()
    front_x_history: list[float] = [front_x_0]
    front_y_history: list[float] = [front_y_0]

    # Rear-axle reference and error at initial state
    initial_yaw_ref = (
        0.0
        if road_type == "straight"
        else road.reference_heading(vehicle.x)
    )
    y_ref_rear_history: list[float] = [initial_y_ref]
    yaw_ref_rear_history: list[float] = [initial_yaw_ref]
    lateral_error_history: list[float] = [
        vehicle.y - initial_y_ref
    ]
    rear_heading_error_history: list[float] = [
        StanleyController.normalize_angle(
            initial_yaw_ref - vehicle.yaw
        )
    ]

    # Front-axle cross-track error at each control step (length = steps + 1)
    cross_track_error_history: list[float] = []
    front_heading_error_history: list[float] = []

    # Control histories (length = steps)
    requested_steering_history: list[float] = []
    angle_limited_steering_history: list[float] = []
    steering_history: list[float] = []
    steering_rate_history: list[float] = []

    for _ in range(steps):
        details = controller.compute_with_details(
            vehicle=vehicle,
            road_x=road_x,
            road_y=road_y,
            velocity=velocity,
            road_yaw=road_yaw
        )

        requested_steering = float(details["steering"])
        cross_track_error = float(details["cross_track_error"])
        front_heading_error = float(details["heading_error"])

        cross_track_error_history.append(cross_track_error)
        front_heading_error_history.append(front_heading_error)

        # 1. Apply steering-angle constraint to requested command
        angle_limited_steering = float(
            np.clip(
                requested_steering,
                -MAX_STEERING,
                MAX_STEERING
            )
        )

        # 2. Apply steering-rate constraint relative to previous applied steering
        max_delta = MAX_STEERING_RATE * dt
        new_steering = float(
            np.clip(
                angle_limited_steering,
                applied_steering - max_delta,
                applied_steering + max_delta
            )
        )

        # 3. Ensure final applied steering stays within angle limits
        new_steering = float(
            np.clip(
                new_steering,
                -MAX_STEERING,
                MAX_STEERING
            )
        )

        # 4. Calculate actual applied steering rate (including step 0 from 0.0)
        actual_steering_rate = (
            new_steering - applied_steering
        ) / dt

        applied_steering = new_steering

        # 5. Update vehicle kinematics
        vehicle.update(
            velocity=velocity,
            steering_angle=applied_steering,
            dt=dt
        )

        # 6. Record post-step state and reference errors
        x_history.append(vehicle.x)
        y_history.append(vehicle.y)
        yaw_history.append(vehicle.yaw)

        fx, fy = vehicle.get_front_axle_position()
        front_x_history.append(fx)
        front_y_history.append(fy)

        if road_type == "straight":
            y_ref_rear = 0.0
            yaw_ref_rear = 0.0
        else:
            y_ref_rear = road.reference_lateral_position(vehicle.x)
            yaw_ref_rear = road.reference_heading(vehicle.x)

        y_ref_rear_history.append(y_ref_rear)
        yaw_ref_rear_history.append(yaw_ref_rear)
        lateral_error_history.append(vehicle.y - y_ref_rear)
        rear_heading_error_history.append(
            StanleyController.normalize_angle(
                yaw_ref_rear - vehicle.yaw
            )
        )

        requested_steering_history.append(requested_steering)
        angle_limited_steering_history.append(angle_limited_steering)
        steering_history.append(applied_steering)
        steering_rate_history.append(actual_steering_rate)

    # Record final front-axle cross-track and heading error at step N
    final_details = controller.compute_with_details(
        vehicle=vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=velocity,
        road_yaw=road_yaw
    )
    cross_track_error_history.append(
        float(final_details["cross_track_error"])
    )
    front_heading_error_history.append(
        float(final_details["heading_error"])
    )

    return {
        "time_states": np.arange(steps + 1, dtype=np.float64) * dt,
        "time_controls": np.arange(steps, dtype=np.float64) * dt,
        "road_x": road_x,
        "road_y": road_y,
        "road_yaw": road_yaw,
        "x": np.asarray(x_history, dtype=np.float64),
        "y": np.asarray(y_history, dtype=np.float64),
        "yaw": np.asarray(yaw_history, dtype=np.float64),
        "front_x": np.asarray(front_x_history, dtype=np.float64),
        "front_y": np.asarray(front_y_history, dtype=np.float64),
        "y_ref": np.asarray(y_ref_rear_history, dtype=np.float64),
        "yaw_ref": np.asarray(yaw_ref_rear_history, dtype=np.float64),
        "lateral_error": np.asarray(
            lateral_error_history,
            dtype=np.float64
        ),
        "heading_error": np.asarray(
            rear_heading_error_history,
            dtype=np.float64
        ),
        "cross_track_error": np.asarray(
            cross_track_error_history,
            dtype=np.float64
        ),
        "front_heading_error": np.asarray(
            front_heading_error_history,
            dtype=np.float64
        ),
        "requested_steering": np.asarray(
            requested_steering_history,
            dtype=np.float64
        ),
        "angle_limited_steering": np.asarray(
            angle_limited_steering_history,
            dtype=np.float64
        ),
        "steering": np.asarray(
            steering_history,
            dtype=np.float64
        ),
        "steering_rate": np.asarray(
            steering_rate_history,
            dtype=np.float64
        )
    }


# ============================================================
# 3. CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_stanley_metrics(
    results: dict[str, NDArray[np.float64]]
) -> dict[str, float | int]:
    """
    Calculate the 10 required Stanley tracking and constraint metrics.

    Note:
        - Rear-axle lateral error (y - y_ref(x)) is reported for consistent
          comparison with rear-axle controllers (Baseline PID and LQR).
        - Front-axle signed cross-track error is reported separately to
          reflect Stanley's geometric control law.
    """
    lateral_error = results["lateral_error"]
    cross_track_error = results["cross_track_error"]
    heading_error = results["heading_error"]
    steering = results["steering"]
    steering_rate = results["steering_rate"]

    # 1. Rear-axle lateral tracking RMSE (m)
    lateral_rmse = float(
        np.sqrt(np.mean(lateral_error ** 2))
    )

    # 2. Maximum absolute rear-axle lateral error (m)
    max_lateral_error = float(
        np.max(np.abs(lateral_error))
    )

    # 3. Final absolute rear-axle lateral error (m)
    final_lateral_error = float(
        abs(lateral_error[-1])
    )

    # 4. Maximum steering angle (deg)
    max_steering_deg = float(
        np.degrees(np.max(np.abs(steering)))
    )

    # 5. Steering RMS (deg)
    steering_rms_deg = float(
        np.degrees(np.sqrt(np.mean(steering ** 2)))
    )

    # 6. Maximum steering rate (deg/s)
    max_steering_rate_deg = float(
        np.degrees(np.max(np.abs(steering_rate)))
    )

    # 7. Final heading error (deg)
    final_heading_error_deg = float(
        np.degrees(abs(heading_error[-1]))
    )

    # 8. Maximum front-axle cross-track error (m) and front-axle CTE RMSE (m)
    max_cross_track_error = float(
        np.max(np.abs(cross_track_error))
    )
    cross_track_rmse = float(
        np.sqrt(np.mean(cross_track_error ** 2))
    )

    # 9. Number of steering constraint violations
    angle_violations = int(
        np.sum(np.abs(steering) > MAX_STEERING + 1e-9)
    )
    rate_violations = int(
        np.sum(np.abs(steering_rate) > MAX_STEERING_RATE + 1e-9)
    )
    total_constraint_violations = (
        angle_violations + rate_violations
    )

    # 10. Number of non-finite numerical values across all arrays
    non_finite_count = int(
        sum(
            int(np.sum(~np.isfinite(arr)))
            for arr in results.values()
        )
    )

    return {
        "Lateral RMSE (m)": lateral_rmse,
        "Max Lateral Error (m)": max_lateral_error,
        "Final Lateral Error (m)": final_lateral_error,
        "Front-Axle CTE RMSE (m)": cross_track_rmse,
        "Max Cross-Track Error (m)": max_cross_track_error,
        "Max Steering (deg)": max_steering_deg,
        "Steering RMS (deg)": steering_rms_deg,
        "Max Steering Rate (deg/s)": max_steering_rate_deg,
        "Final Heading Error (deg)": final_heading_error_deg,
        "Constraint Violations": total_constraint_violations,
        "Non-Finite Values": non_finite_count
    }


# ============================================================
# 4. PRINT PERFORMANCE TABLES
# ============================================================

def print_speed_sensitivity_table(
    road_type: str,
    speed_metrics: dict[float, dict[str, float | int]]
) -> None:
    """
    Print a formatted summary table across all tested vehicle speeds.
    """
    title = (
        "STRAIGHT-ROAD"
        if road_type == "straight"
        else "SINUSOIDAL CURVED-ROAD"
    )
    print(f"\nSTANLEY CONTROLLER - {title} PERFORMANCE BY SPEED")
    print("=" * 95)

    header = f"{'Metric':<30}" + "".join(
        f"{f'{v:.0f} m/s':>15}" for v in speed_metrics
    )
    print(header)
    print("-" * 95)

    first_metrics = next(iter(speed_metrics.values()))
    for metric_name in first_metrics:
        row = f"{metric_name:<30}"
        for v in speed_metrics:
            val = speed_metrics[v][metric_name]
            if isinstance(val, int):
                row += f"{val:>15d}"
            else:
                row += f"{val:>15.4f}"
        print(row)

    print("=" * 95)


# ============================================================
# 5. GENERATE AND SAVE VISUALIZATIONS
# ============================================================

def generate_stanley_plots(
    straight_results_by_speed: dict[float, dict[str, NDArray[np.float64]]],
    curved_results_by_speed: dict[float, dict[str, NDArray[np.float64]]],
    straight_metrics_by_speed: dict[float, dict[str, float | int]],
    curved_metrics_by_speed: dict[float, dict[str, float | int]],
    nominal_speed: float = DEFAULT_PLOT_SPEED,
    show_plots: bool = True
) -> None:
    """
    Generate and save all 7 Phase 8 Stanley controller visualizations.
    """
    straight_nominal = straight_results_by_speed[nominal_speed]
    curved_nominal = curved_results_by_speed[nominal_speed]

    max_x_nominal = float(curved_nominal["x"][-1]) + 10.0
    road_mask_straight = straight_nominal["road_x"] <= max_x_nominal
    road_mask_curved = curved_nominal["road_x"] <= max_x_nominal

    # ========================================================
    # PLOT 1: STRAIGHT-ROAD TRAJECTORY
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        straight_nominal["road_x"][road_mask_straight],
        straight_nominal["road_y"][road_mask_straight],
        "k--",
        linewidth=2,
        label="Reference Road (y = 0)"
    )

    plt.plot(
        straight_nominal["x"],
        straight_nominal["y"],
        color="blue",
        linewidth=2,
        label=f"Stanley Rear Axle (v = {nominal_speed:.0f} m/s)"
    )

    plt.plot(
        straight_nominal["front_x"],
        straight_nominal["front_y"],
        color="cyan",
        linestyle=":",
        linewidth=1.8,
        label="Stanley Front Axle"
    )

    plt.scatter(
        straight_nominal["x"][0],
        straight_nominal["y"][0],
        color="green",
        s=80,
        label="Start"
    )

    plt.scatter(
        straight_nominal["x"][-1],
        straight_nominal["y"][-1],
        color="red",
        s=80,
        label="End"
    )

    plt.xlabel("Longitudinal Position X (m)")
    plt.ylabel("Lateral Position Y (m)")
    plt.title(
        f"Stanley Controller — Straight-Road Trajectory (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "stanley_straight_trajectory.png",
        dpi=300
    )

    # ========================================================
    # PLOT 2: CURVED-ROAD TRAJECTORY
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        curved_nominal["road_x"][road_mask_curved],
        curved_nominal["road_y"][road_mask_curved],
        "k--",
        linewidth=2,
        label="Curved Reference Road"
    )

    plt.plot(
        curved_nominal["x"],
        curved_nominal["y"],
        color="blue",
        linewidth=2,
        label=f"Stanley Rear Axle (v = {nominal_speed:.0f} m/s)"
    )

    plt.plot(
        curved_nominal["front_x"],
        curved_nominal["front_y"],
        color="darkorange",
        linestyle=":",
        linewidth=1.8,
        label="Stanley Front Axle"
    )

    plt.scatter(
        curved_nominal["x"][0],
        curved_nominal["y"][0],
        color="green",
        s=80,
        label="Start"
    )

    plt.scatter(
        curved_nominal["x"][-1],
        curved_nominal["y"][-1],
        color="red",
        s=80,
        label="End"
    )

    plt.xlabel("Longitudinal Position X (m)")
    plt.ylabel("Lateral Position Y (m)")
    plt.title(
        f"Stanley Controller — Sinusoidal Curved-Road Trajectory (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "stanley_curved_trajectory.png",
        dpi=300
    )

    # ========================================================
    # PLOT 3: CROSS-TRACK & LATERAL ERROR VS TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        curved_nominal["time_states"],
        curved_nominal["cross_track_error"],
        color="crimson",
        linewidth=2,
        label="Front-Axle Signed Cross-Track Error e_fa (m)"
    )

    plt.plot(
        curved_nominal["time_states"],
        curved_nominal["lateral_error"],
        color="royalblue",
        linestyle="--",
        linewidth=2,
        label="Rear-Axle Lateral Error y - y_ref (m)"
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        label="Zero Error"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Tracking Error (m)")
    plt.title(
        f"Stanley Controller — Curved-Road Tracking Error vs Time (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "stanley_curved_error.png",
        dpi=300
    )

    # ========================================================
    # PLOT 4: STEERING ANGLE VS TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        curved_nominal["time_controls"],
        np.degrees(curved_nominal["requested_steering"]),
        color="gray",
        linestyle=":",
        linewidth=1.8,
        label="Requested Steering"
    )

    plt.plot(
        curved_nominal["time_controls"],
        np.degrees(curved_nominal["steering"]),
        color="purple",
        linewidth=2,
        label="Applied Steering (Angle & Rate Limited)"
    )

    plt.axhline(
        y=30.0,
        color="red",
        linestyle="--",
        label="Steering Limit (+/-30 deg)"
    )

    plt.axhline(
        y=-30.0,
        color="red",
        linestyle="--"
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle (degrees)")
    plt.title(
        f"Stanley Controller — Curved-Road Steering Angle vs Time (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "stanley_curved_steering.png",
        dpi=300
    )

    # ========================================================
    # PLOT 5: STEERING RATE VS TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        curved_nominal["time_controls"],
        np.degrees(curved_nominal["steering_rate"]),
        color="darkorange",
        linewidth=2,
        label="Applied Steering Rate"
    )

    plt.axhline(
        y=20.0,
        color="red",
        linestyle="--",
        label="Steering Rate Limit (+/-20 deg/s)"
    )

    plt.axhline(
        y=-20.0,
        color="red",
        linestyle="--"
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Rate (degrees/s)")
    plt.title(
        f"Stanley Controller — Curved-Road Steering Rate vs Time (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "stanley_curved_steering_rate.png",
        dpi=300
    )

    # ========================================================
    # PLOT 6: REFERENCE HEADING VS VEHICLE HEADING
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        curved_nominal["time_states"],
        np.degrees(curved_nominal["yaw_ref"]),
        "k--",
        linewidth=2,
        label="Reference Road Heading"
    )

    plt.plot(
        curved_nominal["time_states"],
        np.degrees(curved_nominal["yaw"]),
        color="seagreen",
        linewidth=2,
        label="Vehicle Heading"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Heading Angle (degrees)")
    plt.title(
        f"Stanley Controller — Reference vs Vehicle Heading (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "stanley_heading_comparison.png",
        dpi=300
    )

    # ========================================================
    # PLOT 7: SPEED COMPARISON ACROSS VEHICLE SPEEDS
    # ========================================================

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel (0, 0): Straight-road lateral error vs time across speeds
    ax0 = axes[0, 0]
    for speed, res in straight_results_by_speed.items():
        ax0.plot(
            res["time_states"],
            res["lateral_error"],
            linewidth=2,
            label=f"{speed:.0f} m/s"
        )
    ax0.axhline(y=0.0, color="black", linestyle="--")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Rear-Axle Lateral Error (m)")
    ax0.set_title("Straight Road — Lateral Error vs Time")
    ax0.grid(True)
    ax0.legend()

    # Panel (0, 1): Curved-road front-axle cross-track error vs time across speeds
    ax1 = axes[0, 1]
    for speed, res in curved_results_by_speed.items():
        ax1.plot(
            res["time_states"],
            res["cross_track_error"],
            linewidth=2,
            label=f"{speed:.0f} m/s"
        )
    ax1.axhline(y=0.0, color="black", linestyle="--")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Front-Axle Cross-Track Error (m)")
    ax1.set_title("Curved Road — Cross-Track Error vs Time")
    ax1.grid(True)
    ax1.legend()

    # Panel (1, 0): Curved-road trajectories across speeds
    ax2 = axes[1, 0]
    longest_road = curved_results_by_speed[max(curved_results_by_speed)]
    ax2.plot(
        longest_road["road_x"],
        longest_road["road_y"],
        "k--",
        linewidth=1.8,
        label="Reference Road"
    )
    for speed, res in curved_results_by_speed.items():
        ax2.plot(
            res["x"],
            res["y"],
            linewidth=1.8,
            label=f"{speed:.0f} m/s"
        )
    ax2.set_xlabel("Longitudinal Position X (m)")
    ax2.set_ylabel("Lateral Position Y (m)")
    ax2.set_title("Curved Road — Trajectories by Speed")
    ax2.grid(True)
    ax2.legend()

    # Panel (1, 1): RMSE vs Speed bar/line summary
    ax3 = axes[1, 1]
    speeds_list = list(straight_metrics_by_speed.keys())
    straight_rmse = [
        float(straight_metrics_by_speed[s]["Lateral RMSE (m)"])
        for s in speeds_list
    ]
    curved_rmse = [
        float(curved_metrics_by_speed[s]["Lateral RMSE (m)"])
        for s in speeds_list
    ]
    curved_cte_rmse = [
        float(curved_metrics_by_speed[s]["Front-Axle CTE RMSE (m)"])
        for s in speeds_list
    ]

    ax3.plot(
        speeds_list,
        straight_rmse,
        marker="o",
        linewidth=2,
        label="Straight Rear-Axle RMSE"
    )
    ax3.plot(
        speeds_list,
        curved_rmse,
        marker="s",
        linewidth=2,
        label="Curved Rear-Axle RMSE"
    )
    ax3.plot(
        speeds_list,
        curved_cte_rmse,
        marker="^",
        linestyle="--",
        linewidth=2,
        label="Curved Front-Axle CTE RMSE"
    )
    ax3.set_xlabel("Vehicle Speed (m/s)")
    ax3.set_ylabel("RMSE (m)")
    ax3.set_title("Tracking Error RMSE vs Vehicle Speed")
    ax3.grid(True)
    ax3.legend()

    fig.suptitle(
        "Stanley Controller — Multi-Speed Performance Comparison",
        fontsize=13
    )
    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "stanley_speed_comparison.png",
        dpi=300
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")


# ============================================================
# 6. MAIN ENTRY POINT
# ============================================================

def main(show_plots: bool = True) -> None:
    """
    Run Stanley simulations across straight and curved roads for all speeds.
    """
    straight_results_by_speed: dict[
        float, dict[str, NDArray[np.float64]]
    ] = {}
    curved_results_by_speed: dict[
        float, dict[str, NDArray[np.float64]]
    ] = {}

    straight_metrics_by_speed: dict[
        float, dict[str, float | int]
    ] = {}
    curved_metrics_by_speed: dict[
        float, dict[str, float | int]
    ] = {}

    for speed in TEST_SPEEDS:
        straight_res = run_stanley_simulation(
            road_type="straight",
            velocity=speed
        )
        curved_res = run_stanley_simulation(
            road_type="curved",
            velocity=speed
        )

        straight_results_by_speed[speed] = straight_res
        curved_results_by_speed[speed] = curved_res

        straight_metrics_by_speed[speed] = (
            calculate_stanley_metrics(straight_res)
        )
        curved_metrics_by_speed[speed] = (
            calculate_stanley_metrics(curved_res)
        )

    print_speed_sensitivity_table(
        "straight",
        straight_metrics_by_speed
    )
    print_speed_sensitivity_table(
        "curved",
        curved_metrics_by_speed
    )

    generate_stanley_plots(
        straight_results_by_speed=straight_results_by_speed,
        curved_results_by_speed=curved_results_by_speed,
        straight_metrics_by_speed=straight_metrics_by_speed,
        curved_metrics_by_speed=curved_metrics_by_speed,
        nominal_speed=DEFAULT_PLOT_SPEED,
        show_plots=show_plots
    )


if __name__ == "__main__":
    main()
