from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from controllers.mpc_controller import MPCController
from road import Road
from vehicle import Vehicle


# ============================================================
# 1. SIMULATION CONFIGURATION
# ============================================================

WHEELBASE: float = 2.8
DT: float = 0.1
SIMULATION_TIME: float = 10.0
STEPS: int = int(SIMULATION_TIME / DT)

HORIZON: int = 15

INITIAL_X: float = 0.0
INITIAL_LATERAL_OFFSET: float = 2.0
INITIAL_YAW: float = 0.0
INITIAL_STEERING: float = 0.0

MAX_STEERING: float = float(np.radians(30.0))
MAX_STEERING_RATE: float = float(np.radians(20.0))

SETTLING_TOLERANCE: float = 0.05

TEST_SPEEDS: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)
DEFAULT_PLOT_SPEED: float = 20.0

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. RUN MPC SIMULATION
# ============================================================

def run_mpc_simulation(
    road_type: str = "curved",
    velocity: float = DEFAULT_PLOT_SPEED,
    horizon: int = HORIZON,
    initial_lateral_offset: float = INITIAL_LATERAL_OFFSET,
    initial_yaw: float = INITIAL_YAW,
    initial_steering: float = INITIAL_STEERING,
    dt: float = DT,
    simulation_time: float = SIMULATION_TIME,
    wheelbase: float = WHEELBASE,
    Q: NDArray[np.float64] | None = None,
    R: NDArray[np.float64] | float | None = None,
    Qf: NDArray[np.float64] | None = None,
    warm_start: bool = True
) -> dict[str, NDArray[np.float64]]:
    """
    Execute a closed-loop receding-horizon Constrained MPC lane-keeping
    simulation.

    Args:
        road_type: Either "straight" or "curved".
        velocity: Constant forward vehicle speed (m/s).
        horizon: MPC prediction horizon N (steps).
        initial_lateral_offset: Initial lateral offset relative to road
            centerline at x = 0 (m).
        initial_yaw: Initial vehicle heading angle (rad).
        initial_steering: Initial applied front-wheel steering angle (rad).
        dt: Simulation timestep (s).
        simulation_time: Total simulation duration (s).
        wheelbase: Vehicle wheelbase L (m).
        Q: Optional (3, 3) state weight matrix.
        R: Optional scalar or (1, 1) steering-rate weight.
        Qf: Optional (3, 3) terminal state weight matrix.
        warm_start: Whether to warm-start OSQP across timesteps.

    Returns:
        Dictionary of simulation time-series arrays (float64).
    """
    if road_type not in ("straight", "curved"):
        raise ValueError(
            f"Unknown road_type '{road_type}'. Expected 'straight' or 'curved'."
        )

    is_straight = road_type == "straight"
    steps = int(round(simulation_time / dt))

    road_length = max(
        150.0,
        velocity * simulation_time + velocity * horizon * dt + 60.0,
    )
    num_waypoints = int(road_length * 10) + 1

    road = Road(
        length=road_length,
        amplitude=0.0 if is_straight else 2.0,
        frequency=0.02,
    )

    if is_straight:
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
        wheelbase=wheelbase,
    )

    controller = MPCController(
        velocity=velocity,
        wheelbase=wheelbase,
        dt=dt,
        horizon=horizon,
        max_steering=MAX_STEERING,
        max_steering_rate=MAX_STEERING_RATE,
        Q=Q,
        R=R,
        Qf=Qf,
        warm_start=warm_start,
    )

    applied_steering: float = float(
        np.clip(initial_steering, -MAX_STEERING, MAX_STEERING)
    )

    # State histories (length = steps + 1)
    x_history: list[float] = [vehicle.x]
    y_history: list[float] = [vehicle.y]
    yaw_history: list[float] = [vehicle.yaw]

    e_y_0, e_psi_0, x_ref_0, y_ref_0, psi_ref_0 = (
        controller.compute_tracking_error(
            vehicle=vehicle,
            road=road,
            straight=is_straight,
        )
    )
    x_ref_history: list[float] = [x_ref_0]
    y_ref_history: list[float] = [y_ref_0]
    yaw_ref_history: list[float] = [psi_ref_0]
    lateral_error_history: list[float] = [e_y_0]
    heading_error_history: list[float] = [e_psi_0]
    applied_steering_state_history: list[float] = [applied_steering]

    # Control and solver histories (length = steps)
    steering_history: list[float] = []
    steering_rate_history: list[float] = []
    objective_history: list[float] = []
    solver_time_history: list[float] = []
    total_time_history: list[float] = []
    fallback_flag_history: list[float] = []

    # Predicted horizons at each step: shape (steps, horizon + 1)
    predicted_x_horizons: list[NDArray[np.float64]] = []
    predicted_y_horizons: list[NDArray[np.float64]] = []
    predicted_ey_horizons: list[NDArray[np.float64]] = []
    predicted_delta_horizons: list[NDArray[np.float64]] = []

    for _ in range(steps):
        details = controller.compute_with_details(
            vehicle=vehicle,
            road=road,
            applied_steering=applied_steering,
            straight=is_straight,
        )

        # Apply only the first optimized steering-rate command
        optimized_rate = float(details["steering_rate"])
        limited_rate = float(
            np.clip(
                optimized_rate,
                -MAX_STEERING_RATE,
                MAX_STEERING_RATE,
            )
        )

        # Integrate rate to obtain next applied steering angle
        new_steering = float(
            np.clip(
                applied_steering + limited_rate * dt,
                -MAX_STEERING,
                MAX_STEERING,
            )
        )
        actual_rate = float((new_steering - applied_steering) / dt)
        applied_steering = new_steering

        # Update vehicle kinematics
        vehicle.update(
            velocity=velocity,
            steering_angle=applied_steering,
            dt=dt,
        )

        # Compute post-step tracking errors relative to path normal
        e_y, e_psi, x_ref, y_ref, psi_ref = (
            controller.compute_tracking_error(
                vehicle=vehicle,
                road=road,
                straight=is_straight,
            )
        )

        x_history.append(vehicle.x)
        y_history.append(vehicle.y)
        yaw_history.append(vehicle.yaw)

        x_ref_history.append(x_ref)
        y_ref_history.append(y_ref)
        yaw_ref_history.append(psi_ref)
        lateral_error_history.append(e_y)
        heading_error_history.append(e_psi)
        applied_steering_state_history.append(applied_steering)

        steering_history.append(applied_steering)
        steering_rate_history.append(actual_rate)
        objective_history.append(float(details["objective_value"]))
        solver_time_history.append(float(details["solver_time"]))
        total_time_history.append(float(details["total_time"]))
        fallback_flag_history.append(
            1.0 if bool(details["used_fallback"]) else 0.0
        )

        predicted_x_horizons.append(
            np.asarray(details["predicted_x"], dtype=np.float64)
        )
        predicted_y_horizons.append(
            np.asarray(details["predicted_y"], dtype=np.float64)
        )
        predicted_states = np.asarray(
            details["predicted_states"], dtype=np.float64
        )
        predicted_ey_horizons.append(predicted_states[0, :])
        predicted_delta_horizons.append(predicted_states[2, :])

    return {
        "time_states": np.arange(steps + 1, dtype=np.float64) * dt,
        "time_controls": np.arange(steps, dtype=np.float64) * dt,
        "road_x": road_x,
        "road_y": road_y,
        "road_yaw": road_yaw,
        "x": np.asarray(x_history, dtype=np.float64),
        "y": np.asarray(y_history, dtype=np.float64),
        "yaw": np.asarray(yaw_history, dtype=np.float64),
        "x_ref": np.asarray(x_ref_history, dtype=np.float64),
        "y_ref": np.asarray(y_ref_history, dtype=np.float64),
        "yaw_ref": np.asarray(yaw_ref_history, dtype=np.float64),
        "lateral_error": np.asarray(
            lateral_error_history, dtype=np.float64
        ),
        "heading_error": np.asarray(
            heading_error_history, dtype=np.float64
        ),
        "steering_state": np.asarray(
            applied_steering_state_history, dtype=np.float64
        ),
        "steering": np.asarray(steering_history, dtype=np.float64),
        "steering_rate": np.asarray(
            steering_rate_history, dtype=np.float64
        ),
        "objective": np.asarray(objective_history, dtype=np.float64),
        "solver_time": np.asarray(solver_time_history, dtype=np.float64),
        "total_time": np.asarray(total_time_history, dtype=np.float64),
        "fallback_used": np.asarray(
            fallback_flag_history, dtype=np.float64
        ),
        "predicted_x_horizons": np.asarray(
            predicted_x_horizons, dtype=np.float64
        ),
        "predicted_y_horizons": np.asarray(
            predicted_y_horizons, dtype=np.float64
        ),
        "predicted_ey_horizons": np.asarray(
            predicted_ey_horizons, dtype=np.float64
        ),
        "predicted_delta_horizons": np.asarray(
            predicted_delta_horizons, dtype=np.float64
        ),
    }


# ============================================================
# 3. CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_mpc_metrics(
    results: dict[str, NDArray[np.float64]],
    dt: float = DT,
    settling_tolerance: float = SETTLING_TOLERANCE
) -> dict[str, float | int | None]:
    """
    Calculate the 16 required MPC performance, constraint, and timing metrics.
    """
    lateral_error = results["lateral_error"]
    heading_error = results["heading_error"]
    steering = results["steering"]
    steering_rate = results["steering_rate"]
    solver_time = results["solver_time"]
    total_time = results["total_time"]
    fallback_used = results["fallback_used"]

    # 1. Lateral tracking RMSE (m)
    lateral_rmse = float(np.sqrt(np.mean(lateral_error ** 2)))

    # 2. Maximum absolute lateral error (m)
    max_lateral_error = float(np.max(np.abs(lateral_error)))

    # 3. Final absolute lateral error (m)
    final_lateral_error = float(abs(lateral_error[-1]))

    # 4. Maximum steering angle (deg)
    max_steering_deg = float(np.degrees(np.max(np.abs(steering))))

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

    # 8. Settling time (s) within +/- settling_tolerance (0.05 m)
    outside_tolerance = np.where(
        np.abs(lateral_error) > settling_tolerance
    )[0]
    settling_time: float | None
    if len(outside_tolerance) == 0:
        settling_time = 0.0
    elif outside_tolerance[-1] == len(lateral_error) - 1:
        settling_time = None
    else:
        settling_time = float((outside_tolerance[-1] + 1) * dt)

    # 9. Maximum overshoot (m and %) relative to initial positive offset
    min_error = float(np.min(lateral_error))
    overshoot_m = max(0.0, -min_error)
    initial_abs_error = abs(float(lateral_error[0]))
    overshoot_pct = (
        overshoot_m / initial_abs_error * 100.0
        if initial_abs_error > 0.0
        else 0.0
    )

    # 10. Steering constraint violations
    angle_violations = int(
        np.sum(np.abs(steering) > MAX_STEERING + 1e-9)
    )
    rate_violations = int(
        np.sum(np.abs(steering_rate) > MAX_STEERING_RATE + 1e-9)
    )
    constraint_violations = angle_violations + rate_violations

    # 11. Non-finite numerical values
    non_finite_count = int(
        sum(
            int(np.sum(~np.isfinite(arr)))
            for arr in results.values()
        )
    )

    # 12–15. Solver time and total controller computation time (ms)
    mean_solver_time_ms = float(np.mean(solver_time) * 1000.0)
    max_solver_time_ms = float(np.max(solver_time) * 1000.0)
    mean_total_time_ms = float(np.mean(total_time) * 1000.0)
    max_total_time_ms = float(np.max(total_time) * 1000.0)

    # 16. Solver failure count
    solver_failures = int(np.sum(fallback_used > 0.5))

    return {
        "Lateral RMSE (m)": lateral_rmse,
        "Max Lateral Error (m)": max_lateral_error,
        "Final Lateral Error (m)": final_lateral_error,
        "Settling Time (s)": settling_time,
        "Overshoot (m)": overshoot_m,
        "Overshoot (%)": overshoot_pct,
        "Max Steering (deg)": max_steering_deg,
        "Steering RMS (deg)": steering_rms_deg,
        "Max Steering Rate (deg/s)": max_steering_rate_deg,
        "Final Heading Error (deg)": final_heading_error_deg,
        "Mean Solver Time (ms)": mean_solver_time_ms,
        "Max Solver Time (ms)": max_solver_time_ms,
        "Mean Total Time (ms)": mean_total_time_ms,
        "Max Total Time (ms)": max_total_time_ms,
        "Solver Failures": solver_failures,
        "Constraint Violations": constraint_violations,
        "Non-Finite Values": non_finite_count,
    }


# ============================================================
# 4. PRINT PERFORMANCE TABLES
# ============================================================

def print_mpc_table(
    road_type: str,
    speed_metrics: dict[float, dict[str, float | int | None]]
) -> None:
    """
    Print a formatted summary table of all 16 MPC metrics across speeds.
    """
    road_label = (
        "STRAIGHT ROAD"
        if road_type == "straight"
        else "SINUSOIDAL CURVED ROAD"
    )
    print(f"\nCONSTRAINED MPC CONTROLLER - {road_label} PERFORMANCE BY SPEED")
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
            if val is None:
                row += f"{'Not settled':>15}"
            elif isinstance(val, int):
                row += f"{val:>15d}"
            else:
                row += f"{val:>15.4f}"
        print(row)

    print("=" * 95)


# ============================================================
# 5. GENERATE AND SAVE VISUALIZATIONS
# ============================================================

def generate_mpc_plots(
    straight_results_by_speed: dict[float, dict[str, NDArray[np.float64]]],
    curved_results_by_speed: dict[float, dict[str, NDArray[np.float64]]],
    straight_metrics_by_speed: dict[float, dict[str, float | int | None]],
    curved_metrics_by_speed: dict[float, dict[str, float | int | None]],
    nominal_speed: float = DEFAULT_PLOT_SPEED,
    show_plots: bool = True
) -> None:
    """
    Generate and save all 8 Phase 10 Constrained MPC visualizations.
    """
    straight_nom = straight_results_by_speed[nominal_speed]
    curved_nom = curved_results_by_speed[nominal_speed]

    max_x_nom = float(curved_nom["x"][-1]) + 15.0
    mask_straight = straight_nom["road_x"] <= max_x_nom
    mask_curved = curved_nom["road_x"] <= max_x_nom

    # ========================================================
    # PLOT 1: STRAIGHT-ROAD VEHICLE TRAJECTORY
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        straight_nom["road_x"][mask_straight],
        straight_nom["road_y"][mask_straight],
        "k--",
        linewidth=2,
        label="Reference Road (y = 0)",
    )

    plt.plot(
        straight_nom["x"],
        straight_nom["y"],
        color="blue",
        linewidth=2,
        label=f"Constrained MPC (v = {nominal_speed:.0f} m/s)",
    )

    plt.scatter(
        straight_nom["x"][0],
        straight_nom["y"][0],
        color="green",
        s=80,
        label="Start",
    )

    plt.scatter(
        straight_nom["x"][-1],
        straight_nom["y"][-1],
        color="red",
        s=80,
        label="End",
    )

    plt.xlabel("Longitudinal Position X (m)")
    plt.ylabel("Lateral Position Y (m)")
    plt.title(
        f"Constrained MPC - Straight-Road Vehicle Trajectory (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "mpc_straight_trajectory.png",
        dpi=300,
    )

    # ========================================================
    # PLOT 2: CURVED-ROAD VEHICLE TRAJECTORY
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        curved_nom["road_x"][mask_curved],
        curved_nom["road_y"][mask_curved],
        "k--",
        linewidth=2,
        label="Curved Reference Road",
    )

    plt.plot(
        curved_nom["x"],
        curved_nom["y"],
        color="blue",
        linewidth=2,
        label=f"Constrained MPC (v = {nominal_speed:.0f} m/s)",
    )

    plt.scatter(
        curved_nom["x"][0],
        curved_nom["y"][0],
        color="green",
        s=80,
        label="Start",
    )

    plt.scatter(
        curved_nom["x"][-1],
        curved_nom["y"][-1],
        color="red",
        s=80,
        label="End",
    )

    plt.xlabel("Longitudinal Position X (m)")
    plt.ylabel("Lateral Position Y (m)")
    plt.title(
        f"Constrained MPC - Sinusoidal Curved-Road Trajectory (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "mpc_curved_trajectory.png",
        dpi=300,
    )

    # ========================================================
    # PLOT 3: LATERAL TRACKING ERROR VS TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        straight_nom["time_states"],
        straight_nom["lateral_error"],
        color="royalblue",
        linewidth=2,
        label="Straight Road Lateral Error e_y (m)",
    )

    plt.plot(
        curved_nom["time_states"],
        curved_nom["lateral_error"],
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="Curved Road Path-Normal Error e_y (m)",
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        label="Zero Error",
    )

    plt.axhline(
        y=SETTLING_TOLERANCE,
        color="gray",
        linestyle=":",
        label="Settling Band (+/-0.05 m)",
    )

    plt.axhline(
        y=-SETTLING_TOLERANCE,
        color="gray",
        linestyle=":",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Signed Lateral Error e_y (m)")
    plt.title(
        f"Constrained MPC - Lateral Tracking Error vs Time (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "mpc_lateral_error.png",
        dpi=300,
    )

    # ========================================================
    # PLOT 4: STEERING ANGLE VS TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        straight_nom["time_controls"],
        np.degrees(straight_nom["steering"]),
        color="royalblue",
        linewidth=2,
        label="Straight Road Applied Steering",
    )

    plt.plot(
        curved_nom["time_controls"],
        np.degrees(curved_nom["steering"]),
        color="purple",
        linestyle="--",
        linewidth=2,
        label="Curved Road Applied Steering",
    )

    plt.axhline(
        y=30.0,
        color="red",
        linestyle="--",
        label="Hard Steering Limit (+/-30 deg)",
    )

    plt.axhline(
        y=-30.0,
        color="red",
        linestyle="--",
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle (degrees)")
    plt.title(
        f"Constrained MPC - Applied Steering Angle vs Time (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "mpc_steering_angle.png",
        dpi=300,
    )

    # ========================================================
    # PLOT 5: STEERING RATE VS TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        straight_nom["time_controls"],
        np.degrees(straight_nom["steering_rate"]),
        color="steelblue",
        linewidth=2,
        label="Straight Road Steering Rate",
    )

    plt.plot(
        curved_nom["time_controls"],
        np.degrees(curved_nom["steering_rate"]),
        color="darkorange",
        linestyle="--",
        linewidth=2,
        label="Curved Road Steering Rate",
    )

    plt.axhline(
        y=20.0,
        color="red",
        linestyle="--",
        label="Hard Steering Rate Limit (+/-20 deg/s)",
    )

    plt.axhline(
        y=-20.0,
        color="red",
        linestyle="--",
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Rate (degrees/s)")
    plt.title(
        f"Constrained MPC - Steering Rate vs Time (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "mpc_steering_rate.png",
        dpi=300,
    )

    # ========================================================
    # PLOT 6: PREDICTED MPC TRAJECTORY AT SELECTED TIMESTEPS
    # ========================================================

    plt.figure(figsize=(12, 5.5))

    plt.plot(
        curved_nom["road_x"][mask_curved],
        curved_nom["road_y"][mask_curved],
        "k--",
        linewidth=2,
        label="Curved Reference Road",
    )

    plt.plot(
        curved_nom["x"],
        curved_nom["y"],
        color="royalblue",
        linewidth=2,
         alpha=0.85,
        label="Actual Vehicle Trajectory",
    )

    sample_steps = [0, 10, 25, 45, 70]
    horizon_colors = ["crimson", "darkorange", "forestgreen", "purple", "teal"]

    pred_x_all = curved_nom["predicted_x_horizons"]
    pred_y_all = curved_nom["predicted_y_horizons"]

    for step_idx, color in zip(sample_steps, horizon_colors):
        if step_idx < len(pred_x_all):
            t_sec = step_idx * DT
            px = pred_x_all[step_idx]
            py = pred_y_all[step_idx]
            plt.plot(
                px,
                py,
                color=color,
                linewidth=2.2,
                marker=".",
                markersize=5,
                label=f"MPC Horizon at t = {t_sec:.1f} s (N = {HORIZON})",
            )
            plt.scatter(
                px[0],
                py[0],
                color=color,
                s=55,
                zorder=5,
            )

    plt.xlabel("Longitudinal Position X (m)")
    plt.ylabel("Lateral Position Y (m)")
    plt.title(
        f"Constrained MPC - Receding-Horizon Predictions on Curved Road (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "mpc_predicted_horizons.png",
        dpi=300,
    )

    # ========================================================
    # PLOT 7: SOLVER COMPUTATION TIME VS SIMULATION TIME
    # ========================================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        curved_nom["time_controls"],
        curved_nom["solver_time"] * 1000.0,
        color="forestgreen",
        linewidth=2,
        label="OSQP Solver Time (ms)",
    )

    plt.plot(
        curved_nom["time_controls"],
        curved_nom["total_time"] * 1000.0,
        color="darkorange",
        linestyle="--",
        linewidth=2,
        label="Total Controller Computation Time (ms)",
    )

    plt.axhline(
        y=DT * 1000.0,
        color="red",
        linestyle=":",
        linewidth=1.8,
        label=f"Control Period Budget ({DT * 1000.0:.0f} ms)",
    )

    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Computation Time (ms)")
    plt.title(
        f"Constrained MPC - Solver & Controller Computation Time (v = {nominal_speed:.0f} m/s)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "mpc_computation_time.png",
        dpi=300,
    )

    # ========================================================
    # PLOT 8: SPEED SENSITIVITY COMPARISON
    # ========================================================

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel (0, 0): Straight-road lateral error vs time across speeds
    ax0 = axes[0, 0]
    for speed, res in straight_results_by_speed.items():
        ax0.plot(
            res["time_states"],
            res["lateral_error"],
            linewidth=2,
            label=f"{speed:.0f} m/s",
        )
    ax0.axhline(y=0.0, color="black", linestyle="--")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Lateral Error e_y (m)")
    ax0.set_title("Straight Road - Lateral Error vs Time")
    ax0.grid(True)
    ax0.legend()

    # Panel (0, 1): Curved-road lateral error vs time across speeds
    ax1 = axes[0, 1]
    for speed, res in curved_results_by_speed.items():
        ax1.plot(
            res["time_states"],
            res["lateral_error"],
            linewidth=2,
            label=f"{speed:.0f} m/s",
        )
    ax1.axhline(y=0.0, color="black", linestyle="--")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Path-Normal Error e_y (m)")
    ax1.set_title("Curved Road - Lateral Error vs Time")
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
        label="Reference Road",
    )
    for speed, res in curved_results_by_speed.items():
        ax2.plot(
            res["x"],
            res["y"],
            linewidth=1.8,
            label=f"{speed:.0f} m/s",
        )
    ax2.set_xlabel("Longitudinal Position X (m)")
    ax2.set_ylabel("Lateral Position Y (m)")
    ax2.set_title("Curved Road - Trajectories by Speed")
    ax2.grid(True)
    ax2.legend()

    # Panel (1, 1): RMSE vs Speed summary
    ax3 = axes[1, 1]
    speeds_list = list(straight_metrics_by_speed.keys())
    straight_rmse = [
        float(straight_metrics_by_speed[s]["Lateral RMSE (m)"] or 0.0)
        for s in speeds_list
    ]
    curved_rmse = [
        float(curved_metrics_by_speed[s]["Lateral RMSE (m)"] or 0.0)
        for s in speeds_list
    ]

    ax3.plot(
        speeds_list,
        straight_rmse,
        marker="o",
        linewidth=2,
        label="Straight Road Lateral RMSE",
    )
    ax3.plot(
        speeds_list,
        curved_rmse,
        marker="s",
        linewidth=2,
        label="Curved Road Lateral RMSE",
    )
    ax3.set_xlabel("Vehicle Speed (m/s)")
    ax3.set_ylabel("Lateral RMSE (m)")
    ax3.set_title("Lateral Tracking RMSE vs Vehicle Speed")
    ax3.grid(True)
    ax3.legend()

    fig.suptitle(
        "Constrained MPC Controller - Multi-Speed Performance Comparison",
        fontsize=13,
    )
    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "mpc_speed_comparison.png",
        dpi=300,
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
    Run Constrained MPC simulations across straight and curved roads
    for all test speeds (5, 10, 20, 30 m/s).
    """
    straight_results_by_speed: dict[
        float, dict[str, NDArray[np.float64]]
    ] = {}
    curved_results_by_speed: dict[
        float, dict[str, NDArray[np.float64]]
    ] = {}

    straight_metrics_by_speed: dict[
        float, dict[str, float | int | None]
    ] = {}
    curved_metrics_by_speed: dict[
        float, dict[str, float | int | None]
    ] = {}

    for speed in TEST_SPEEDS:
        s_res = run_mpc_simulation(
            road_type="straight",
            velocity=speed,
        )
        c_res = run_mpc_simulation(
            road_type="curved",
            velocity=speed,
        )

        straight_results_by_speed[speed] = s_res
        curved_results_by_speed[speed] = c_res

        straight_metrics_by_speed[speed] = calculate_mpc_metrics(s_res)
        curved_metrics_by_speed[speed] = calculate_mpc_metrics(c_res)

    print_mpc_table("straight", straight_metrics_by_speed)
    print_mpc_table("curved", curved_metrics_by_speed)

    generate_mpc_plots(
        straight_results_by_speed=straight_results_by_speed,
        curved_results_by_speed=curved_results_by_speed,
        straight_metrics_by_speed=straight_metrics_by_speed,
        curved_metrics_by_speed=curved_metrics_by_speed,
        nominal_speed=DEFAULT_PLOT_SPEED,
        show_plots=show_plots,
    )


if __name__ == "__main__":
    main()
