from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from controllers.pure_pursuit_controller import PurePursuitController
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

FIXED_LOOKAHEAD: float = 10.0
MIN_LOOKAHEAD: float = 3.0
LOOKAHEAD_GAIN: float = 0.5

MAX_STEERING: float = float(np.radians(30.0))
MAX_STEERING_RATE: float = float(np.radians(20.0))

TEST_SPEEDS: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)
DEFAULT_PLOT_SPEED: float = 20.0

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. RUN PURE PURSUIT SIMULATION
# ============================================================

def run_pure_pursuit_simulation(
    road_type: str = "curved",
    velocity: float = DEFAULT_PLOT_SPEED,
    use_dynamic_lookahead: bool = True,
    fixed_lookahead: float = FIXED_LOOKAHEAD,
    min_lookahead: float = MIN_LOOKAHEAD,
    lookahead_gain: float = LOOKAHEAD_GAIN,
    initial_lateral_offset: float = INITIAL_LATERAL_OFFSET,
    initial_yaw: float = INITIAL_YAW,
    dt: float = DT,
    simulation_time: float = SIMULATION_TIME,
    wheelbase: float = WHEELBASE
) -> dict[str, NDArray[np.float64]]:
    """
    Execute a closed-loop Pure Pursuit lane-keeping simulation.

    Args:
        road_type: Either "straight" or "curved".
        velocity: Nonnegative forward vehicle speed (m/s).
        use_dynamic_lookahead: True for Ld = L_min + K_v * v,
            False for fixed lookahead.
        fixed_lookahead: Fixed lookahead distance (m) when
            use_dynamic_lookahead is False.
        min_lookahead: Minimum lookahead L_min (m).
        lookahead_gain: Velocity gain K_v (s).
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

    # Ensure the reference path extends well beyond max travel + lookahead
    max_lookahead = max(
        fixed_lookahead,
        min_lookahead + lookahead_gain * velocity
    )
    road_length = max(
        150.0,
        velocity * simulation_time + max_lookahead + 60.0
    )
    num_waypoints = int(road_length * 10) + 1

    road = Road(length=road_length)

    if road_type == "straight":
        road_x, road_y = road.generate_straight_road(
            num_points=num_waypoints
        )
        road_yaw = np.zeros_like(road_x)
        initial_y_ref = 0.0
        initial_yaw_ref = 0.0
    else:
        road_x, road_y = road.generate_curved_road(
            num_points=num_waypoints
        )
        road_yaw = road.reference_heading(road_x)
        initial_y_ref = road.reference_lateral_position(INITIAL_X)
        initial_yaw_ref = road.reference_heading(INITIAL_X)

    vehicle = Vehicle(
        x=INITIAL_X,
        y=initial_y_ref + initial_lateral_offset,
        yaw=initial_yaw,
        wheelbase=wheelbase
    )

    controller = PurePursuitController(
        wheelbase=wheelbase,
        lookahead_distance=fixed_lookahead,
        min_lookahead=min_lookahead,
        lookahead_gain=lookahead_gain,
        use_dynamic_lookahead=use_dynamic_lookahead
    )

    applied_steering: float = 0.0
    current_min_index: int = 0

    # State histories (length = steps + 1)
    x_history: list[float] = [vehicle.x]
    y_history: list[float] = [vehicle.y]
    yaw_history: list[float] = [vehicle.yaw]

    y_ref_history: list[float] = [initial_y_ref]
    yaw_ref_history: list[float] = [initial_yaw_ref]
    lateral_error_history: list[float] = [
        vehicle.y - initial_y_ref
    ]
    heading_error_history: list[float] = [
        PurePursuitController.normalize_angle(
            initial_yaw_ref - vehicle.yaw
        )
    ]

    # Control and lookahead histories (length = steps)
    requested_steering_history: list[float] = []
    angle_limited_steering_history: list[float] = []
    steering_history: list[float] = []
    steering_rate_history: list[float] = []

    desired_lookahead_history: list[float] = []
    actual_lookahead_history: list[float] = []
    alpha_history: list[float] = []
    target_index_history: list[float] = []
    target_x_history: list[float] = []
    target_y_history: list[float] = []

    for _ in range(steps):
        details = controller.compute_with_details(
            vehicle=vehicle,
            road_x=road_x,
            road_y=road_y,
            velocity=velocity,
            use_dynamic_lookahead=use_dynamic_lookahead,
            min_index=current_min_index
        )

        # Enforce forward progression along the path
        current_min_index = int(details["nearest_index"])

        requested_steering = float(details["requested_steering"])
        desired_lookahead = float(
            details["desired_lookahead_distance"]
        )
        actual_lookahead = float(
            details["actual_lookahead_distance"]
        )
        alpha = float(details["heading_to_target_error"])
        target_index = float(details["target_index"])
        target_x = float(details["target_x"])
        target_y = float(details["target_y"])

        # 1. Steering angle limit on requested command
        angle_limited_steering = float(
            np.clip(
                requested_steering,
                -MAX_STEERING,
                MAX_STEERING
            )
        )

        # 2. Steering rate limit relative to previous applied steering
        max_delta = MAX_STEERING_RATE * dt
        new_steering = float(
            np.clip(
                angle_limited_steering,
                applied_steering - max_delta,
                applied_steering + max_delta
            )
        )

        # 3. Final steering angle clip
        new_steering = float(
            np.clip(
                new_steering,
                -MAX_STEERING,
                MAX_STEERING
            )
        )

        # 4. Actual applied steering rate (including step 0 from 0.0)
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

        # 6. Record post-step states and reference path errors
        x_history.append(vehicle.x)
        y_history.append(vehicle.y)
        yaw_history.append(vehicle.yaw)

        if road_type == "straight":
            y_ref = 0.0
            yaw_ref = 0.0
        else:
            y_ref = road.reference_lateral_position(vehicle.x)
            yaw_ref = road.reference_heading(vehicle.x)

        y_ref_history.append(y_ref)
        yaw_ref_history.append(yaw_ref)
        lateral_error_history.append(vehicle.y - y_ref)
        heading_error_history.append(
            PurePursuitController.normalize_angle(
                yaw_ref - vehicle.yaw
            )
        )

        requested_steering_history.append(requested_steering)
        angle_limited_steering_history.append(angle_limited_steering)
        steering_history.append(applied_steering)
        steering_rate_history.append(actual_steering_rate)

        desired_lookahead_history.append(desired_lookahead)
        actual_lookahead_history.append(actual_lookahead)
        alpha_history.append(alpha)
        target_index_history.append(target_index)
        target_x_history.append(target_x)
        target_y_history.append(target_y)

    return {
        "time_states": np.arange(steps + 1, dtype=np.float64) * dt,
        "time_controls": np.arange(steps, dtype=np.float64) * dt,
        "road_x": road_x,
        "road_y": road_y,
        "road_yaw": road_yaw,
        "x": np.asarray(x_history, dtype=np.float64),
        "y": np.asarray(y_history, dtype=np.float64),
        "yaw": np.asarray(yaw_history, dtype=np.float64),
        "y_ref": np.asarray(y_ref_history, dtype=np.float64),
        "yaw_ref": np.asarray(yaw_ref_history, dtype=np.float64),
        "lateral_error": np.asarray(
            lateral_error_history,
            dtype=np.float64
        ),
        "heading_error": np.asarray(
            heading_error_history,
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
        ),
        "desired_lookahead": np.asarray(
            desired_lookahead_history,
            dtype=np.float64
        ),
        "actual_lookahead": np.asarray(
            actual_lookahead_history,
            dtype=np.float64
        ),
        "alpha": np.asarray(alpha_history, dtype=np.float64),
        "target_index": np.asarray(
            target_index_history,
            dtype=np.float64
        ),
        "target_x": np.asarray(target_x_history, dtype=np.float64),
        "target_y": np.asarray(target_y_history, dtype=np.float64)
    }


# ============================================================
# 3. CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_pure_pursuit_metrics(
    results: dict[str, NDArray[np.float64]]
) -> dict[str, float | int]:
    """
    Calculate the 10 required Pure Pursuit performance and lookahead metrics.
    """
    lateral_error = results["lateral_error"]
    heading_error = results["heading_error"]
    steering = results["steering"]
    steering_rate = results["steering_rate"]
    actual_lookahead = results["actual_lookahead"]
    desired_lookahead = results["desired_lookahead"]

    # 1. Rear-axle lateral tracking RMSE (m)
    lateral_rmse = float(
        np.sqrt(np.mean(lateral_error ** 2))
    )

    # 2. Maximum absolute lateral error (m)
    max_lateral_error = float(
        np.max(np.abs(lateral_error))
    )

    # 3. Final absolute lateral error (m)
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

    # 8. Number of steering constraint violations
    angle_violations = int(
        np.sum(np.abs(steering) > MAX_STEERING + 1e-9)
    )
    rate_violations = int(
        np.sum(np.abs(steering_rate) > MAX_STEERING_RATE + 1e-9)
    )
    total_constraint_violations = (
        angle_violations + rate_violations
    )

    # 9. Number of non-finite numerical values
    non_finite_count = int(
        sum(
            int(np.sum(~np.isfinite(arr)))
            for arr in results.values()
        )
    )

    # 10. Lookahead target distance statistics (m)
    desired_ld = float(np.mean(desired_lookahead))
    mean_actual_ld = float(np.mean(actual_lookahead))
    min_actual_ld = float(np.min(actual_lookahead))
    max_actual_ld = float(np.max(actual_lookahead))

    return {
        "Lateral RMSE (m)": lateral_rmse,
        "Max Lateral Error (m)": max_lateral_error,
        "Final Lateral Error (m)": final_lateral_error,
        "Max Steering (deg)": max_steering_deg,
        "Steering RMS (deg)": steering_rms_deg,
        "Max Steering Rate (deg/s)": max_steering_rate_deg,
        "Final Heading Error (deg)": final_heading_error_deg,
        "Desired Lookahead (m)": desired_ld,
        "Mean Lookahead (m)": mean_actual_ld,
        "Min Lookahead (m)": min_actual_ld,
        "Max Lookahead (m)": max_actual_ld,
        "Constraint Violations": total_constraint_violations,
        "Non-Finite Values": non_finite_count
    }


# ============================================================
# 4. PRINT PERFORMANCE TABLES
# ============================================================

def print_pure_pursuit_table(
    road_type: str,
    mode_label: str,
    speed_metrics: dict[float, dict[str, float | int]]
) -> None:
    """
    Print a formatted performance table across all tested vehicle speeds.
    """
    road_label = (
        "STRAIGHT ROAD"
        if road_type == "straight"
        else "SINUSOIDAL CURVED ROAD"
    )
    print(f"\nPURE PURSUIT ({mode_label}) - {road_label}")
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

def generate_pure_pursuit_plots(
    dynamic_straight: dict[float, dict[str, NDArray[np.float64]]],
    dynamic_curved: dict[float, dict[str, NDArray[np.float64]]],
    fixed_straight: dict[float, dict[str, NDArray[np.float64]]],
    fixed_curved: dict[float, dict[str, NDArray[np.float64]]],
    dynamic_straight_metrics: dict[float, dict[str, float | int]],
    dynamic_curved_metrics: dict[float, dict[str, float | int]],
    fixed_straight_metrics: dict[float, dict[str, float | int]],
    fixed_curved_metrics: dict[float, dict[str, float | int]],
    nominal_speed: float = DEFAULT_PLOT_SPEED,
    show_plots: bool = True
) -> None:
    """
    Generate and save all 7 Phase 9 Pure Pursuit visualizations.
    """
    dyn_s_nom = dynamic_straight[nominal_speed]
    fix_s_nom = fixed_straight[nominal_speed]
    dyn_c_nom = dynamic_curved[nominal_speed]
    fix_c_nom = fixed_curved[nominal_speed]

    max_x_nom = float(dyn_c_nom["x"][-1]) + 10.0
    mask_straight = dyn_s_nom["road_x"] <= max_x_nom
    mask_curved = dyn_c_nom["road_x"] <= max_x_nom

    # ========================================================
    # PLOT 1: STRAIGHT-ROAD TRAJECTORY
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        dyn_s_nom["road_x"][mask_straight],
        dyn_s_nom["road_y"][mask_straight],
        "k--",
        linewidth=2,
        label="Reference Road (y = 0)"
    )

    plt.plot(
        dyn_s_nom["x"],
        dyn_s_nom["y"],
        color="blue",
        linewidth=2,
        label=f"Dynamic Lookahead (v = {nominal_speed:.0f} m/s)"
    )

    plt.plot(
        fix_s_nom["x"],
        fix_s_nom["y"],
        color="darkorange",
        linestyle="-.",
        linewidth=2,
        label=f"Fixed Lookahead ({FIXED_LOOKAHEAD:.0f} m, v = {nominal_speed:.0f} m/s)"
    )

    plt.scatter(
        dyn_s_nom["x"][0],
        dyn_s_nom["y"][0],
        color="green",
        s=80,
        label="Start"
    )

    plt.scatter(
        dyn_s_nom["x"][-1],
        dyn_s_nom["y"][-1],
        color="red",
        s=80,
        label="End"
    )

    plt.xlabel("Longitudinal Position X (m)")
    plt.ylabel("Lateral Position Y (m)")
    plt.title(
        f"Pure Pursuit - Straight-Road Trajectory (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "pure_pursuit_straight_trajectory.png",
        dpi=300
    )

    # ========================================================
    # PLOT 2: CURVED-ROAD TRAJECTORY
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        dyn_c_nom["road_x"][mask_curved],
        dyn_c_nom["road_y"][mask_curved],
        "k--",
        linewidth=2,
        label="Curved Reference Road"
    )

    plt.plot(
        dyn_c_nom["x"],
        dyn_c_nom["y"],
        color="blue",
        linewidth=2,
        label=f"Dynamic Lookahead (v = {nominal_speed:.0f} m/s)"
    )

    plt.plot(
        fix_c_nom["x"],
        fix_c_nom["y"],
        color="darkorange",
        linestyle="-.",
        linewidth=2,
        label=f"Fixed Lookahead ({FIXED_LOOKAHEAD:.0f} m, v = {nominal_speed:.0f} m/s)"
    )

    plt.scatter(
        dyn_c_nom["x"][0],
        dyn_c_nom["y"][0],
        color="green",
        s=80,
        label="Start"
    )

    plt.scatter(
        dyn_c_nom["x"][-1],
        dyn_c_nom["y"][-1],
        color="red",
        s=80,
        label="End"
    )

    plt.xlabel("Longitudinal Position X (m)")
    plt.ylabel("Lateral Position Y (m)")
    plt.title(
        f"Pure Pursuit - Sinusoidal Curved-Road Trajectory (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "pure_pursuit_curved_trajectory.png",
        dpi=300
    )

    # ========================================================
    # PLOT 3: LATERAL TRACKING ERROR VS TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        dyn_c_nom["time_states"],
        dyn_c_nom["lateral_error"],
        color="crimson",
        linewidth=2,
        label="Dynamic Lookahead Lateral Error (m)"
    )

    plt.plot(
        fix_c_nom["time_states"],
        fix_c_nom["lateral_error"],
        color="royalblue",
        linestyle="--",
        linewidth=2,
        label=f"Fixed Lookahead ({FIXED_LOOKAHEAD:.0f} m) Lateral Error (m)"
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        label="Zero Error"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Rear-Axle Lateral Error y - y_ref (m)")
    plt.title(
        f"Pure Pursuit - Curved-Road Lateral Tracking Error vs Time (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "pure_pursuit_curved_error.png",
        dpi=300
    )

    # ========================================================
    # PLOT 4: STEERING ANGLE VS TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        dyn_c_nom["time_controls"],
        np.degrees(dyn_c_nom["requested_steering"]),
        color="gray",
        linestyle=":",
        linewidth=1.8,
        label="Requested Steering (Dynamic)"
    )

    plt.plot(
        dyn_c_nom["time_controls"],
        np.degrees(dyn_c_nom["steering"]),
        color="purple",
        linewidth=2,
        label="Applied Steering (Dynamic)"
    )

    plt.plot(
        fix_c_nom["time_controls"],
        np.degrees(fix_c_nom["steering"]),
        color="teal",
        linestyle="--",
        linewidth=1.8,
        label=f"Applied Steering (Fixed {FIXED_LOOKAHEAD:.0f} m)"
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
        f"Pure Pursuit - Curved-Road Steering Angle vs Time (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "pure_pursuit_steering.png",
        dpi=300
    )

    # ========================================================
    # PLOT 5: STEERING RATE VS TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        dyn_c_nom["time_controls"],
        np.degrees(dyn_c_nom["steering_rate"]),
        color="darkorange",
        linewidth=2,
        label="Applied Steering Rate (Dynamic)"
    )

    plt.plot(
        fix_c_nom["time_controls"],
        np.degrees(fix_c_nom["steering_rate"]),
        color="steelblue",
        linestyle="--",
        linewidth=1.8,
        label=f"Applied Steering Rate (Fixed {FIXED_LOOKAHEAD:.0f} m)"
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
        f"Pure Pursuit - Curved-Road Steering Rate vs Time (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "pure_pursuit_steering_rate.png",
        dpi=300
    )

    # ========================================================
    # PLOT 6: LOOKAHEAD DISTANCE VS TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    for speed, res in dynamic_curved.items():
        plt.plot(
            res["time_controls"],
            res["actual_lookahead"],
            linewidth=2,
            label=f"Dynamic Actual Ld (v = {speed:.0f} m/s)"
        )

    plt.plot(
        fix_c_nom["time_controls"],
        fix_c_nom["actual_lookahead"],
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Fixed Lookahead ({FIXED_LOOKAHEAD:.0f} m, v = {nominal_speed:.0f} m/s)"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Actual Lookahead Distance Ld (m)")
    plt.title("Pure Pursuit - Actual Lookahead Distance vs Time Across Speeds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "pure_pursuit_lookahead.png",
        dpi=300
    )

    # ========================================================
    # PLOT 7: SPEED SENSITIVITY COMPARISON
    # ========================================================

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel (0, 0): Straight-road lateral error vs time (Dynamic Lookahead)
    ax0 = axes[0, 0]
    for speed, res in dynamic_straight.items():
        ax0.plot(
            res["time_states"],
            res["lateral_error"],
            linewidth=2,
            label=f"{speed:.0f} m/s"
        )
    ax0.axhline(y=0.0, color="black", linestyle="--")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Rear-Axle Lateral Error (m)")
    ax0.set_title("Straight Road - Lateral Error vs Time (Dynamic Ld)")
    ax0.grid(True)
    ax0.legend()

    # Panel (0, 1): Curved-road lateral error vs time (Dynamic Lookahead)
    ax1 = axes[0, 1]
    for speed, res in dynamic_curved.items():
        ax1.plot(
            res["time_states"],
            res["lateral_error"],
            linewidth=2,
            label=f"{speed:.0f} m/s"
        )
    ax1.axhline(y=0.0, color="black", linestyle="--")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Rear-Axle Lateral Error (m)")
    ax1.set_title("Curved Road - Lateral Error vs Time (Dynamic Ld)")
    ax1.grid(True)
    ax1.legend()

    # Panel (1, 0): Curved-road trajectories across speeds (Dynamic Lookahead)
    ax2 = axes[1, 0]
    longest_road = dynamic_curved[max(dynamic_curved)]
    ax2.plot(
        longest_road["road_x"],
        longest_road["road_y"],
        "k--",
        linewidth=1.8,
        label="Reference Road"
    )
    for speed, res in dynamic_curved.items():
        ax2.plot(
            res["x"],
            res["y"],
            linewidth=1.8,
            label=f"{speed:.0f} m/s"
        )
    ax2.set_xlabel("Longitudinal Position X (m)")
    ax2.set_ylabel("Lateral Position Y (m)")
    ax2.set_title("Curved Road - Trajectories by Speed (Dynamic Ld)")
    ax2.grid(True)
    ax2.legend()

    # Panel (1, 1): Fixed vs Dynamic RMSE across speeds
    ax3 = axes[1, 1]
    speeds_list = list(dynamic_straight_metrics.keys())

    dyn_s_rmse = [
        float(dynamic_straight_metrics[s]["Lateral RMSE (m)"])
        for s in speeds_list
    ]
    fix_s_rmse = [
        float(fixed_straight_metrics[s]["Lateral RMSE (m)"])
        for s in speeds_list
    ]
    dyn_c_rmse = [
        float(dynamic_curved_metrics[s]["Lateral RMSE (m)"])
        for s in speeds_list
    ]
    fix_c_rmse = [
        float(fixed_curved_metrics[s]["Lateral RMSE (m)"])
        for s in speeds_list
    ]

    ax3.plot(
        speeds_list,
        dyn_s_rmse,
        marker="o",
        linewidth=2,
        label="Straight (Dynamic Ld)"
    )
    ax3.plot(
        speeds_list,
        fix_s_rmse,
        marker="o",
        linestyle="--",
        linewidth=2,
        label=f"Straight (Fixed {FIXED_LOOKAHEAD:.0f} m)"
    )
    ax3.plot(
        speeds_list,
        dyn_c_rmse,
        marker="s",
        linewidth=2,
        label="Curved (Dynamic Ld)"
    )
    ax3.plot(
        speeds_list,
        fix_c_rmse,
        marker="s",
        linestyle="--",
        linewidth=2,
        label=f"Curved (Fixed {FIXED_LOOKAHEAD:.0f} m)"
    )
    ax3.set_xlabel("Vehicle Speed (m/s)")
    ax3.set_ylabel("Lateral RMSE (m)")
    ax3.set_title("Fixed vs Speed-Dependent Lookahead RMSE")
    ax3.grid(True)
    ax3.legend()

    fig.suptitle(
        "Pure Pursuit Controller - Multi-Speed & Lookahead Comparison",
        fontsize=13
    )
    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "pure_pursuit_speed_comparison.png",
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
    Run Pure Pursuit simulations across straight and curved roads for
    both fixed and speed-dependent lookahead across all test speeds.
    """
    dynamic_straight: dict[float, dict[str, NDArray[np.float64]]] = {}
    dynamic_curved: dict[float, dict[str, NDArray[np.float64]]] = {}
    fixed_straight: dict[float, dict[str, NDArray[np.float64]]] = {}
    fixed_curved: dict[float, dict[str, NDArray[np.float64]]] = {}

    dynamic_straight_metrics: dict[float, dict[str, float | int]] = {}
    dynamic_curved_metrics: dict[float, dict[str, float | int]] = {}
    fixed_straight_metrics: dict[float, dict[str, float | int]] = {}
    fixed_curved_metrics: dict[float, dict[str, float | int]] = {}

    for speed in TEST_SPEEDS:
        # Speed-dependent lookahead
        dyn_s = run_pure_pursuit_simulation(
            road_type="straight",
            velocity=speed,
            use_dynamic_lookahead=True
        )
        dyn_c = run_pure_pursuit_simulation(
            road_type="curved",
            velocity=speed,
            use_dynamic_lookahead=True
        )

        # Fixed lookahead
        fix_s = run_pure_pursuit_simulation(
            road_type="straight",
            velocity=speed,
            use_dynamic_lookahead=False,
            fixed_lookahead=FIXED_LOOKAHEAD
        )
        fix_c = run_pure_pursuit_simulation(
            road_type="curved",
            velocity=speed,
            use_dynamic_lookahead=False,
            fixed_lookahead=FIXED_LOOKAHEAD
        )

        dynamic_straight[speed] = dyn_s
        dynamic_curved[speed] = dyn_c
        fixed_straight[speed] = fix_s
        fixed_curved[speed] = fix_c

        dynamic_straight_metrics[speed] = (
            calculate_pure_pursuit_metrics(dyn_s)
        )
        dynamic_curved_metrics[speed] = (
            calculate_pure_pursuit_metrics(dyn_c)
        )
        fixed_straight_metrics[speed] = (
            calculate_pure_pursuit_metrics(fix_s)
        )
        fixed_curved_metrics[speed] = (
            calculate_pure_pursuit_metrics(fix_c)
        )

    print_pure_pursuit_table(
        "straight",
        "SPEED-DEPENDENT LOOKAHEAD: Ld = 3.0 + 0.5*v",
        dynamic_straight_metrics
    )
    print_pure_pursuit_table(
        "curved",
        "SPEED-DEPENDENT LOOKAHEAD: Ld = 3.0 + 0.5*v",
        dynamic_curved_metrics
    )
    print_pure_pursuit_table(
        "straight",
        f"FIXED LOOKAHEAD: Ld = {FIXED_LOOKAHEAD:.1f} m",
        fixed_straight_metrics
    )
    print_pure_pursuit_table(
        "curved",
        f"FIXED LOOKAHEAD: Ld = {FIXED_LOOKAHEAD:.1f} m",
        fixed_curved_metrics
    )

    generate_pure_pursuit_plots(
        dynamic_straight=dynamic_straight,
        dynamic_curved=dynamic_curved,
        fixed_straight=fixed_straight,
        fixed_curved=fixed_curved,
        dynamic_straight_metrics=dynamic_straight_metrics,
        dynamic_curved_metrics=dynamic_curved_metrics,
        fixed_straight_metrics=fixed_straight_metrics,
        fixed_curved_metrics=fixed_curved_metrics,
        nominal_speed=DEFAULT_PLOT_SPEED,
        show_plots=show_plots
    )


if __name__ == "__main__":
    main()
