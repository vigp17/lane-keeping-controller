"""
Phase 11F: Closed-Loop Constrained MPC Lane-Keeping Integration in CARLA 0.9.16.

Connects the receding-horizon Constrained MPCController (CVXPY + OSQP) and
ConservativeLongitudinalController to a live Tesla Model 3 in CARLA 0.9.16
(Town10HD_Opt) using CarlaControllerAdapter for all physics-derived vehicle
geometry, rear-axle coordinate transformations, and steering conversions on
the validated 135.24-meter extended curved route.

Prediction Model & Curvature Preview:
    - State Vector: z[k] = [e_y[k], e_psi[k], delta_applied[k]]^T (3x1)
    - Control Input: u[k] = steering_rate[k] (1x1, rad/s)
    - Discretization: Exact Zero-Order Hold (ZOH) recalculated at the actual
      CARLA synchronous timestep dt = 0.05 s and live physics wheelbase
      L ~= 3.0046 m over horizon N = 15 (0.75 s lookahead preview):
          z[k+1] = Ad * z[k] + Bd * u[k] + Ed * kappa_preview[k]
    - Curvature Preview: At each control step, the rear-axle orthogonal
      projection s_0 along the route's cumulative arc length s is advanced
      across k = 0, ..., N-1 via:
          s[k] = clip(s_0 + k * v_prog * dt, 0.0, s_max)
      and the Gaussian-smoothed reference curvature profile kappa_ref(s) is
      interpolated at each future arc-length coordinate s[k].
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Ensure src directory is on sys.path when executed directly as a script
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import carla  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    carla = None  # type: ignore[assignment]

from carla_integration.augmented_lqr_carla import (
    STRAIGHT_APPROACH_MAX_S_M,
    ReferenceRouteProfile,
    build_reference_route_profile,
    query_reference_route_at_position,
)
from carla_integration.carla_adapter import CarlaControllerAdapter, normalize_angle
from carla_integration.stanley_carla import (
    CARLA_HOST,
    CARLA_PORT,
    CLIENT_TIMEOUT_S,
    COLLISION_SENSOR_BP_ID,
    CURVATURE_THRESHOLD_1PM,
    ConservativeLongitudinalController,
    ENDPOINT_TOLERANCE_M,
    FIXED_DELTA_SECONDS,
    MAX_ALLOWED_CTE_M,
    MAX_SIMULATION_DURATION_S,
    MAX_STEERING_DEG,
    MAX_STEERING_RAD,
    MAX_STEERING_RATE_DEG_S,
    MAX_STEERING_RATE_RAD_S,
    MAX_STEP_YAW_CHANGE_DEG,
    MAX_TOTAL_ROUTE_YAW_DEG,
    PREFERRED_SPAWN_POINT_INDEX,
    RESULTS_DIR,
    ROUTE_DISTANCE_M,
    SETTLE_TICKS,
    STOP_BRAKE_TICKS,
    TARGET_SPEED_MPS,
    VEHICLE_BLUEPRINT_ID,
    WAYPOINT_SPACING_M,
    RouteDetails,
    SegmentPerformanceMetrics,
    draw_route_markers,
    select_spawn_point_and_route,
    update_spectator_camera,
    verify_route_continuity,
)
from controllers.mpc_controller import MPCController
from vehicle import Vehicle


# ==================================================
# 1. MPC CARLA CONFIGURATION CONSTANTS
# ==================================================

MPC_HORIZON_STEPS: int = 15
MPC_HORIZON_DURATION_S: float = MPC_HORIZON_STEPS * FIXED_DELTA_SECONDS
CONTROL_PERIOD_MS: float = FIXED_DELTA_SECONDS * 1000.0

MIN_PREVIEW_SPEED_MPS: float = 1.0
MAX_CONSECUTIVE_SOLVER_FAILURES: int = 5
MAX_TOTAL_SOLVER_FAILURES: int = 10

DEFAULT_MPC_FILE_PREFIX: str = "mpc_carla_8ms"
DEFAULT_MPC_CSV_PATH: Path = RESULTS_DIR / f"{DEFAULT_MPC_FILE_PREFIX}_telemetry.csv"


# ==================================================
# 2. DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class MPCTelemetrySample:
    """Single closed-loop Constrained MPC CARLA telemetry sample."""

    step: int
    sim_time_s: float
    route_progress_m: float
    route_progress_pct: float
    speed_mps: float
    rear_x_m: float
    rear_y_m: float
    front_x_m: float
    front_y_m: float
    yaw_rad: float
    ref_x_m: float
    ref_y_m: float
    ref_yaw_rad: float
    nearest_index: int
    front_nearest_index: int
    rear_axle_tracking_error_m: float
    front_cross_track_error_m: float
    heading_error_rad: float
    heading_error_deg: float
    reference_curvature_1pm: float
    curvature_preview_mean_1pm: float
    curvature_preview_max_abs_1pm: float
    curvature_preview_end_1pm: float
    curvature_preview_str: str
    requested_steering_rate_rad_s: float
    requested_steering_rate_deg_s: float
    applied_steering_rate_rad_s: float
    applied_steering_rate_deg_s: float
    applied_steering_rad: float
    applied_steering_deg: float
    normalized_carla_steer: float
    throttle: float
    brake: float
    solver_status: str
    solver_time_ms: float
    total_controller_time_ms: float
    objective_value: float
    used_fallback: bool
    consecutive_solver_failures: int


@dataclass(frozen=True)
class MPCCarlaTestSummary:
    """Summary metrics and verification results from the Constrained MPC CARLA run."""

    termination_reason: str
    route_completed: bool
    collision_count: int
    total_steps: int
    simulation_duration_s: float
    route_details: RouteDetails
    wheelbase_m: float
    rear_axle_offset_x_m: float
    front_axle_offset_x_m: float
    max_front_wheel_steer_deg: float
    max_front_wheel_steer_rad: float
    horizon_steps: int
    horizon_duration_s: float
    max_speed_mps: float
    mean_speed_mps: float
    final_speed_mps: float
    rear_axle_rmse_m: float
    max_rear_axle_error_m: float
    final_rear_axle_error_m: float
    front_cte_rmse_m: float
    max_front_cte_m: float
    final_front_cte_m: float
    max_heading_error_deg: float
    final_heading_error_deg: float
    max_requested_rate_deg_s: float
    max_applied_steering_deg: float
    steering_rms_deg: float
    max_steering_rate_deg_s: float
    steering_rate_rms_deg_s: float
    max_normalized_carla_steer: float
    mean_solver_time_ms: float
    max_solver_time_ms: float
    mean_total_controller_time_ms: float
    max_total_controller_time_ms: float
    solver_failure_count: int
    fallback_step_count: int
    deadline_misses_50ms: int
    angle_constraint_violations: int
    rate_constraint_violations: int
    non_finite_values: int
    csv_path: Path
    plot_paths: dict[str, Path]
    segment_metrics: dict[str, SegmentPerformanceMetrics]


# ==================================================
# 3. CURVATURE PREVIEW & MPC CONTROL STEP
# ==================================================

def compute_curvature_preview_horizon(
    profile: ReferenceRouteProfile,
    current_s_m: float,
    speed_mps: float,
    heading_error_rad: float = 0.0,
    dt: float = FIXED_DELTA_SECONDS,
    horizon: int = MPC_HORIZON_STEPS,
    min_preview_speed_mps: float = MIN_PREVIEW_SPEED_MPS
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate the N-step future road curvature sequence kappa_preview[0:N]
    along the route's cumulative arc length s using the forward reference
    progress model:
        v_prog = max(speed_mps * cos(heading_error_rad), min_preview_speed_mps)
        s_preview[k] = clip(current_s_m + k * v_prog * dt, 0.0, s_max),  k = 0..N-1
        kappa_preview[k] = interp(s_preview[k], profile.road_s, profile.curvature_1pm)

    Returns:
        Tuple of (kappa_preview_1pm, s_preview_m) each of shape (horizon,).
    """
    if not (
        math.isfinite(current_s_m)
        and math.isfinite(speed_mps)
        and math.isfinite(heading_error_rad)
        and math.isfinite(dt)
    ):
        raise ValueError("All curvature preview inputs must be finite numbers.")
    if dt <= 0.0:
        raise ValueError("Timestep dt must be strictly positive.")
    if int(horizon) < 1:
        raise ValueError("Horizon N must be at least 1.")

    n_steps = int(horizon)
    s_max = float(profile.road_s[-1])
    s_0 = float(np.clip(current_s_m, 0.0, s_max))

    forward_speed_mps = max(
        float(speed_mps) * math.cos(float(heading_error_rad)),
        float(min_preview_speed_mps)
    )
    step_indices = np.arange(n_steps, dtype=np.float64)
    s_preview = np.clip(
        s_0 + step_indices * forward_speed_mps * float(dt),
        0.0,
        s_max
    ).astype(np.float64)

    kappa_preview = np.asarray(
        np.interp(s_preview, profile.road_s, profile.curvature_1pm),
        dtype=np.float64
    )

    return kappa_preview, s_preview


def format_curvature_preview_str(kappa_preview: NDArray[np.float64]) -> str:
    """Format the N-step curvature preview vector compactly for CSV logging."""
    return ";".join(f"{float(val):+.5f}" for val in kappa_preview)


def compute_mpc_control_step(
    adapter: CarlaControllerAdapter,
    controller: MPCController,
    speed_controller: ConservativeLongitudinalController,
    profile: ReferenceRouteProfile,
    previous_applied_steering_rad: float,
    min_index: int,
    consecutive_solver_failures: int,
    dt: float,
    step: int,
    sim_time_s: float,
    force_solver_failure: bool = False
) -> MPCTelemetrySample:
    """
    Execute one closed-loop receding-horizon Constrained MPC + longitudinal
    control step using live vehicle state and physics-derived geometry.
    """
    t_step_start = time.perf_counter()

    rear_x, rear_y, yaw_rad = adapter.get_rear_axle_pose()
    speed_mps = adapter.get_speed()

    # Construct kinematic Vehicle at the rear axle with live physics wheelbase
    kinematic_vehicle = Vehicle(
        x=rear_x,
        y=rear_y,
        yaw=yaw_rad,
        wheelbase=adapter.wheelbase
    )
    front_x, front_y = kinematic_vehicle.get_front_axle_position()

    # Project rear axle onto reference route
    (
        nearest_index,
        rear_s_m,
        ref_x_m,
        ref_y_m,
        ref_yaw_rad,
        kappa_ref_1pm,
        _
    ) = query_reference_route_at_position(
        point_x=rear_x,
        point_y=rear_y,
        profile=profile,
        min_index=min_index
    )

    # Compute path-normal rear-axle lateral error e_y and heading error e_psi
    # matching MPCController.compute_tracking_error sign conventions:
    #   e_y = -(x - x_ref)*sin(psi_ref) + (y - y_ref)*cos(psi_ref)
    #   e_psi = normalize_angle(yaw - psi_ref)
    e_y_m = float(
        -(rear_x - ref_x_m) * math.sin(ref_yaw_rad)
        + (rear_y - ref_y_m) * math.cos(ref_yaw_rad)
    )
    e_psi_rad = normalize_angle(yaw_rad - ref_yaw_rad)

    # Generate N-step road curvature preview along cumulative arc length
    kappa_preview, _ = compute_curvature_preview_horizon(
        profile=profile,
        current_s_m=rear_s_m,
        speed_mps=speed_mps,
        heading_error_rad=e_psi_rad,
        dt=dt,
        horizon=controller.N
    )

    # Solve Constrained MPC QP from z0 = [e_y, e_psi, previous_applied_steering_rad]
    solve_result = controller.solve_from_state(
        lateral_error=e_y_m,
        heading_error=e_psi_rad,
        steering_angle=previous_applied_steering_rad,
        kappa_preview=kappa_preview,
        force_failure=force_solver_failure
    )

    predicted_controls = np.asarray(
        solve_result["predicted_controls"],
        dtype=np.float64
    )
    requested_rate_rad_s = float(predicted_controls[0])
    applied_rate_rad_s = float(solve_result["steering_rate"])
    applied_steering_rad = float(solve_result["next_steering"])
    solver_status = str(solve_result["solver_status"])
    solver_time_ms = float(solve_result["solver_time"]) * 1000.0
    objective_value = float(solve_result["objective_value"])
    used_fallback = bool(solve_result["used_fallback"])

    new_consecutive_failures = (
        consecutive_solver_failures + 1 if used_fallback else 0
    )

    # Convert applied steering (+left, rad) to normalized CARLA steer (+right)
    normalized_carla_steer = adapter.to_carla_steer(applied_steering_rad)

    # Front-axle cross-track error & route progress (consistent with Stanley/PP/LQR)
    front_dx = profile.road_x - front_x
    front_dy = profile.road_y - front_y
    front_nearest_index = int(np.argmin(front_dx ** 2 + front_dy ** 2))
    front_ref_x = float(profile.road_x[front_nearest_index])
    front_ref_y = float(profile.road_y[front_nearest_index])
    front_ref_yaw = float(profile.road_yaw[front_nearest_index])
    front_cte_m = float(
        -(front_x - front_ref_x) * math.sin(front_ref_yaw)
        + (front_y - front_ref_y) * math.cos(front_ref_yaw)
    )

    total_route_len_m = max(float(profile.road_s[-1]), 1e-6)
    route_progress_m = max(rear_s_m, float(profile.road_s[front_nearest_index]))
    route_progress_pct = float(
        np.clip(100.0 * route_progress_m / total_route_len_m, 0.0, 100.0)
    )

    # Longitudinal speed control targeting 8.0 m/s
    throttle, brake = speed_controller.compute(
        current_speed_mps=speed_mps,
        dt=dt
    )

    total_controller_time_ms = float((time.perf_counter() - t_step_start) * 1000.0)

    return MPCTelemetrySample(
        step=int(step),
        sim_time_s=float(sim_time_s),
        route_progress_m=float(route_progress_m),
        route_progress_pct=route_progress_pct,
        speed_mps=float(speed_mps),
        rear_x_m=float(rear_x),
        rear_y_m=float(rear_y),
        front_x_m=float(front_x),
        front_y_m=float(front_y),
        yaw_rad=float(yaw_rad),
        ref_x_m=ref_x_m,
        ref_y_m=ref_y_m,
        ref_yaw_rad=ref_yaw_rad,
        nearest_index=nearest_index,
        front_nearest_index=front_nearest_index,
        rear_axle_tracking_error_m=e_y_m,
        front_cross_track_error_m=front_cte_m,
        heading_error_rad=e_psi_rad,
        heading_error_deg=float(math.degrees(e_psi_rad)),
        reference_curvature_1pm=float(kappa_ref_1pm),
        curvature_preview_mean_1pm=float(np.mean(kappa_preview)),
        curvature_preview_max_abs_1pm=float(np.max(np.abs(kappa_preview))),
        curvature_preview_end_1pm=float(kappa_preview[-1]),
        curvature_preview_str=format_curvature_preview_str(kappa_preview),
        requested_steering_rate_rad_s=requested_rate_rad_s,
        requested_steering_rate_deg_s=float(math.degrees(requested_rate_rad_s)),
        applied_steering_rate_rad_s=applied_rate_rad_s,
        applied_steering_rate_deg_s=float(math.degrees(applied_rate_rad_s)),
        applied_steering_rad=applied_steering_rad,
        applied_steering_deg=float(math.degrees(applied_steering_rad)),
        normalized_carla_steer=float(normalized_carla_steer),
        throttle=float(throttle),
        brake=float(brake),
        solver_status=solver_status,
        solver_time_ms=solver_time_ms,
        total_controller_time_ms=total_controller_time_ms,
        objective_value=objective_value,
        used_fallback=used_fallback,
        consecutive_solver_failures=new_consecutive_failures
    )


def is_mpc_sample_finite(sample: MPCTelemetrySample) -> bool:
    """Return True if all numerical fields in the telemetry sample are finite."""
    for value in asdict(sample).values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def check_mpc_stopping_conditions(
    sample: MPCTelemetrySample,
    road_x: NDArray[np.float64],
    road_y: NDArray[np.float64],
    collision_count: int,
    total_solver_failures: int = 0,
    max_consecutive_failures: int = MAX_CONSECUTIVE_SOLVER_FAILURES,
    max_total_failures: int = MAX_TOTAL_SOLVER_FAILURES,
    max_duration_s: float = MAX_SIMULATION_DURATION_S,
    max_cte_m: float = MAX_ALLOWED_CTE_M,
    endpoint_tolerance_m: float = ENDPOINT_TOLERANCE_M
) -> tuple[bool, str, bool]:
    """
    Evaluate all safety, solver reliability, and mission stopping conditions.

    Returns:
        Tuple of (should_stop, termination_reason, route_completed).
    """
    if not is_mpc_sample_finite(sample):
        return True, "invalid_numerical_values", False

    if collision_count > 0:
        return True, "collision_detected", False

    if (
        sample.consecutive_solver_failures >= max_consecutive_failures
        or total_solver_failures >= max_total_failures
    ):
        return True, "repeated_solver_failures", False

    if (
        abs(sample.front_cross_track_error_m) > max_cte_m
        or abs(sample.rear_axle_tracking_error_m) > max_cte_m
    ):
        return True, "left_intended_route", False

    dist_front_to_endpoint = math.hypot(
        sample.front_x_m - float(road_x[-1]),
        sample.front_y_m - float(road_y[-1])
    )
    dist_rear_to_endpoint = math.hypot(
        sample.rear_x_m - float(road_x[-1]),
        sample.rear_y_m - float(road_y[-1])
    )
    if (
        min(dist_front_to_endpoint, dist_rear_to_endpoint) <= endpoint_tolerance_m
        or sample.front_nearest_index >= len(road_x) - 2
        or sample.nearest_index >= len(road_x) - 2
    ):
        return True, "route_endpoint_reached", True

    if sample.sim_time_s >= max_duration_s - 1e-9:
        return True, "max_duration_reached", False

    return False, "running", False


# ==================================================
# 4. SEGMENT METRICS, CSV EXPORT & PLOTTING
# ==================================================

def _compute_mpc_subset_metrics(
    segment_name: str,
    subset: Sequence[MPCTelemetrySample],
    segment_distance_m: float
) -> SegmentPerformanceMetrics:
    """Compute tracking and steering metrics over a sample subset."""
    if not subset:
        return SegmentPerformanceMetrics(
            segment_name=segment_name,
            sample_count=0,
            distance_m=0.0,
            mean_speed_mps=0.0,
            max_speed_mps=0.0,
            front_cte_rmse_m=0.0,
            max_front_cte_m=0.0,
            rear_axle_rmse_m=0.0,
            max_rear_axle_error_m=0.0,
            max_heading_error_deg=0.0,
            steering_rms_deg=0.0,
            max_applied_steering_deg=0.0,
            max_steering_rate_deg_s=0.0,
            steering_rate_rms_deg_s=0.0
        )

    speeds = np.array([s.speed_mps for s in subset], dtype=np.float64)
    front_ctes = np.array([s.front_cross_track_error_m for s in subset], dtype=np.float64)
    rear_errs = np.array([s.rear_axle_tracking_error_m for s in subset], dtype=np.float64)
    head_deg = np.array([s.heading_error_deg for s in subset], dtype=np.float64)
    app_steer_deg = np.array([s.applied_steering_deg for s in subset], dtype=np.float64)
    steer_rate_deg_s = np.array([s.applied_steering_rate_deg_s for s in subset], dtype=np.float64)

    return SegmentPerformanceMetrics(
        segment_name=segment_name,
        sample_count=len(subset),
        distance_m=float(segment_distance_m),
        mean_speed_mps=float(np.mean(speeds)),
        max_speed_mps=float(np.max(speeds)),
        front_cte_rmse_m=float(np.sqrt(np.mean(front_ctes ** 2))),
        max_front_cte_m=float(np.max(np.abs(front_ctes))),
        rear_axle_rmse_m=float(np.sqrt(np.mean(rear_errs ** 2))),
        max_rear_axle_error_m=float(np.max(np.abs(rear_errs))),
        max_heading_error_deg=float(np.max(np.abs(head_deg))),
        steering_rms_deg=float(np.sqrt(np.mean(app_steer_deg ** 2))),
        max_applied_steering_deg=float(np.max(np.abs(app_steer_deg))),
        max_steering_rate_deg_s=float(np.max(np.abs(steer_rate_deg_s))),
        steering_rate_rms_deg_s=float(np.sqrt(np.mean(steer_rate_deg_s ** 2)))
    )


def calculate_mpc_segment_metrics(
    samples: Sequence[MPCTelemetrySample],
    route_details: RouteDetails | None = None,
    curvature_threshold_1pm: float = CURVATURE_THRESHOLD_1PM,
    approach_max_s_m: float = STRAIGHT_APPROACH_MAX_S_M
) -> dict[str, SegmentPerformanceMetrics]:
    """
    Separate Constrained MPC performance metrics into:
        - "straight_approach": Straight approach segment (0 to 57.50 m)
        - "sustained_curve": Sustained curved segment (57.50 to 117.72 m)
        - "straight_exit": Straight exit segment (117.72 to 135.24 m)
        - "straight_combined": All straight samples combined
        - "overall": Overall route (all samples)
    """
    if not samples:
        raise ValueError("Cannot calculate segment metrics from empty samples.")

    approach_samples = [
        s for s in samples
        if abs(s.reference_curvature_1pm) < curvature_threshold_1pm
        and s.route_progress_m <= approach_max_s_m + 0.5
    ]
    curve_samples = [
        s for s in samples
        if abs(s.reference_curvature_1pm) >= curvature_threshold_1pm
    ]
    exit_samples = [
        s for s in samples
        if abs(s.reference_curvature_1pm) < curvature_threshold_1pm
        and s.route_progress_m > approach_max_s_m + 0.5
    ]
    straight_all_samples = [
        s for s in samples
        if abs(s.reference_curvature_1pm) < curvature_threshold_1pm
    ]

    if route_details is not None and route_details.total_length_m > 0.0:
        overall_dist = route_details.total_length_m
        curved_dist = route_details.curved_length_m
        straight_total_dist = route_details.straight_length_m
        approach_dist = min(approach_max_s_m, straight_total_dist)
        exit_dist = max(0.0, straight_total_dist - approach_dist)
    else:
        overall_dist = float(samples[-1].route_progress_m - samples[0].route_progress_m)
        curved_dist = 0.0
        straight_total_dist = overall_dist
        approach_dist = overall_dist
        exit_dist = 0.0

    return {
        "straight_approach": _compute_mpc_subset_metrics(
            "Straight Approach", approach_samples, approach_dist
        ),
        "sustained_curve": _compute_mpc_subset_metrics(
            "Sustained Curve", curve_samples, curved_dist
        ),
        "straight_exit": _compute_mpc_subset_metrics(
            "Straight Exit", exit_samples, exit_dist
        ),
        "straight_combined": _compute_mpc_subset_metrics(
            "Straight Combined", straight_all_samples, straight_total_dist
        ),
        "overall": _compute_mpc_subset_metrics(
            "Overall Route", samples, overall_dist
        ),
    }


def save_mpc_telemetry_to_csv(
    samples: Sequence[MPCTelemetrySample],
    csv_path: Path = DEFAULT_MPC_CSV_PATH
) -> Path:
    """Save recorded Constrained MPC telemetry samples to a CSV file."""
    if not samples:
        raise ValueError("Cannot save empty telemetry sample sequence.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(samples[0]).keys())

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))

    return csv_path


def generate_mpc_carla_plots(
    samples: Sequence[MPCTelemetrySample],
    profile: ReferenceRouteProfile,
    output_dir: Path = RESULTS_DIR,
    target_speed_mps: float = TARGET_SPEED_MPS,
    file_prefix: str = DEFAULT_MPC_FILE_PREFIX
) -> dict[str, Path]:
    """
    Generate and save the 6 required closed-loop Constrained MPC CARLA plots:
        1. results/mpc_carla_8ms_trajectory.png
        2. results/mpc_carla_8ms_error.png
        3. results/mpc_carla_8ms_steering.png
        4. results/mpc_carla_8ms_speed.png
        5. results/mpc_carla_8ms_curvature.png
        6. results/mpc_carla_8ms_solver_time.png
    """
    if not samples:
        raise ValueError("Cannot generate plots from empty telemetry samples.")

    output_dir.mkdir(parents=True, exist_ok=True)

    time_s = np.array([s.sim_time_s for s in samples], dtype=np.float64)
    progress_m = np.array([s.route_progress_m for s in samples], dtype=np.float64)
    curvature_1pm = np.array([s.reference_curvature_1pm for s in samples], dtype=np.float64)
    preview_mean_1pm = np.array([s.curvature_preview_mean_1pm for s in samples], dtype=np.float64)
    preview_end_1pm = np.array([s.curvature_preview_end_1pm for s in samples], dtype=np.float64)
    rear_x = np.array([s.rear_x_m for s in samples], dtype=np.float64)
    rear_y = np.array([s.rear_y_m for s in samples], dtype=np.float64)
    front_x = np.array([s.front_x_m for s in samples], dtype=np.float64)
    front_y = np.array([s.front_y_m for s in samples], dtype=np.float64)
    rear_err = np.array([s.rear_axle_tracking_error_m for s in samples], dtype=np.float64)
    front_cte = np.array([s.front_cross_track_error_m for s in samples], dtype=np.float64)
    app_steer_deg = np.array([s.applied_steering_deg for s in samples], dtype=np.float64)
    req_rate_deg_s = np.array([s.requested_steering_rate_deg_s for s in samples], dtype=np.float64)
    app_rate_deg_s = np.array([s.applied_steering_rate_deg_s for s in samples], dtype=np.float64)
    carla_steer = np.array([s.normalized_carla_steer for s in samples], dtype=np.float64)
    speed_mps = np.array([s.speed_mps for s in samples], dtype=np.float64)
    solver_ms = np.array([s.solver_time_ms for s in samples], dtype=np.float64)
    total_ms = np.array([s.total_controller_time_ms for s in samples], dtype=np.float64)

    trajectory_path = output_dir / f"{file_prefix}_trajectory.png"
    error_path = output_dir / f"{file_prefix}_error.png"
    steering_path = output_dir / f"{file_prefix}_steering.png"
    speed_path = output_dir / f"{file_prefix}_speed.png"
    curvature_path = output_dir / f"{file_prefix}_curvature.png"
    solver_time_path = output_dir / f"{file_prefix}_solver_time.png"

    curved_mask = np.abs(curvature_1pm) >= CURVATURE_THRESHOLD_1PM
    has_curved = bool(np.any(curved_mask))
    if has_curved:
        curve_start_m = float(progress_m[curved_mask][0])
        curve_end_m = float(progress_m[curved_mask][-1])
    else:
        curve_start_m = 0.0
        curve_end_m = 0.0

    # 1. Vehicle Trajectory Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        profile.road_x,
        profile.road_y,
        "k--",
        linewidth=2.0,
        label="Reference Driving Lane (Controller Frame)"
    )
    ax.plot(
        rear_x,
        rear_y,
        color="tab:blue",
        linewidth=2.0,
        label="Constrained MPC Rear-Axle Trajectory"
    )
    ax.plot(
        front_x,
        front_y,
        color="tab:cyan",
        linestyle=":",
        linewidth=1.8,
        label="Front-Axle Trajectory"
    )
    ax.scatter(rear_x[0], rear_y[0], color="green", s=70, zorder=5, label="Start (Rear Axle)")
    ax.scatter(rear_x[-1], rear_y[-1], color="red", s=70, zorder=5, label="End (Rear Axle)")
    ax.set_title("CARLA 0.9.16 Closed-Loop Constrained MPC Trajectory (Tesla Model 3, Town10HD_Opt)")
    ax.set_xlabel("Controller X (m)")
    ax.set_ylabel("Controller Y (m, +Left)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(trajectory_path, dpi=150)
    plt.close(fig)

    # 2. Tracking Error Plot (vs. Time and vs. Route Distance)
    fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(10, 7.5))
    ax_t.plot(
        time_s,
        rear_err,
        color="tab:orange",
        linewidth=2.0,
        label="Rear-Axle Tracking Error e_y (m)"
    )
    ax_t.plot(
        time_s,
        front_cte,
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label="Front-Axle Cross-Track Error (m)"
    )
    ax_t.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_t.set_title("CARLA 0.9.16 Constrained MPC Tracking Error vs. Time and Route Distance")
    ax_t.set_xlabel("Simulation Time (s)")
    ax_t.set_ylabel("Tracking Error (m)")
    ax_t.grid(True, linestyle="--", alpha=0.5)
    ax_t.legend(loc="best")

    if has_curved:
        ax_s.axvspan(
            curve_start_m,
            curve_end_m,
            color="gold",
            alpha=0.22,
            label=f"Sustained Curved Segment ({curve_start_m:.1f}-{curve_end_m:.1f} m)"
        )
    ax_s.plot(
        progress_m,
        rear_err,
        color="tab:orange",
        linewidth=2.0,
        label="Rear-Axle Tracking Error e_y (m)"
    )
    ax_s.plot(
        progress_m,
        front_cte,
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label="Front-Axle Cross-Track Error (m)"
    )
    ax_s.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_s.set_xlabel("Route Distance / Progress s (m)")
    ax_s.set_ylabel("Tracking Error (m)")
    ax_s.grid(True, linestyle="--", alpha=0.5)
    ax_s.legend(loc="best")
    fig.tight_layout()
    fig.savefig(error_path, dpi=150)
    plt.close(fig)

    # 3. Steering Angle & Steering Rate Plot
    fig, (ax_ang, ax_rate) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    ax_ang.plot(
        time_s,
        app_steer_deg,
        color="tab:blue",
        linewidth=2.0,
        label="Applied Steering Angle delta_applied (deg, +Left)"
    )
    ax_ang.plot(
        time_s,
        carla_steer * MAX_STEERING_DEG,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label="Normalized CARLA Steer (scaled x30 deg, +Right)"
    )
    ax_ang.axhline(MAX_STEERING_DEG, color="red", linestyle="--", linewidth=1.0, alpha=0.5, label="+/-30 deg Limit")
    ax_ang.axhline(-MAX_STEERING_DEG, color="red", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_ang.set_title("CARLA 0.9.16 Constrained MPC Steering Angle and Steering Rate vs. Time")
    ax_ang.set_ylabel("Steering Angle (deg)")
    ax_ang.grid(True, linestyle="--", alpha=0.5)
    ax_ang.legend(loc="best")

    ax_rate.plot(
        time_s,
        req_rate_deg_s,
        color="tab:gray",
        linestyle="--",
        linewidth=1.5,
        label="Requested Steering Rate u*[0] (deg/s)"
    )
    ax_rate.plot(
        time_s,
        app_rate_deg_s,
        color="tab:blue",
        linewidth=2.0,
        label="Applied Steering Rate (deg/s)"
    )
    ax_rate.axhline(MAX_STEERING_RATE_DEG_S, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="+/-20 deg/s Limit")
    ax_rate.axhline(-MAX_STEERING_RATE_DEG_S, color="red", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_rate.set_xlabel("Simulation Time (s)")
    ax_rate.set_ylabel("Steering Rate (deg/s)")
    ax_rate.grid(True, linestyle="--", alpha=0.5)
    ax_rate.legend(loc="best")
    fig.tight_layout()
    fig.savefig(steering_path, dpi=150)
    plt.close(fig)

    # 4. Speed vs. Time Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        time_s,
        speed_mps,
        color="tab:green",
        linewidth=2.0,
        label="Measured Vehicle Speed (m/s)"
    )
    ax.axhline(
        target_speed_mps,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Target Speed ({target_speed_mps:.1f} m/s)"
    )
    ax.set_title("CARLA 0.9.16 Constrained MPC Longitudinal Speed Tracking vs. Time")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(speed_path, dpi=150)
    plt.close(fig)

    # 5. Reference Curvature & Preview Horizon Plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        progress_m,
        curvature_1pm,
        color="tab:red",
        linewidth=2.0,
        label="Current Reference Curvature kappa_preview[0] (m^-1)"
    )
    ax.plot(
        progress_m,
        preview_mean_1pm,
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label="Horizon Mean Preview Curvature mean(kappa_preview[0:N]) (m^-1)"
    )
    ax.plot(
        progress_m,
        preview_end_1pm,
        color="tab:purple",
        linestyle=":",
        linewidth=1.8,
        label="Horizon Terminal Preview Curvature kappa_preview[N-1] (m^-1)"
    )
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_title("CARLA 0.9.16 Constrained MPC Road Curvature Preview Over N=15 Horizon")
    ax.set_xlabel("Route Progress s (m)")
    ax.set_ylabel("Curvature (m^-1)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(curvature_path, dpi=150)
    plt.close(fig)

    # 6. Solver & Total Controller Computation Time Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        time_s,
        solver_ms,
        color="tab:blue",
        linewidth=1.8,
        label=f"OSQP Solver Time (mean = {float(np.mean(solver_ms)):.3f} ms)"
    )
    ax.plot(
        time_s,
        total_ms,
        color="tab:orange",
        linewidth=1.8,
        label=f"Total Controller Time (mean = {float(np.mean(total_ms)):.3f} ms)"
    )
    ax.axhline(
        CONTROL_PERIOD_MS,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Control Period Deadline ({CONTROL_PERIOD_MS:.1f} ms)"
    )
    ax.set_title("CARLA 0.9.16 Constrained MPC OSQP Solver & Total Step Time vs. Time")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Computation Time (ms)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(solver_time_path, dpi=150)
    plt.close(fig)

    return {
        "trajectory": trajectory_path,
        "error": error_path,
        "steering": steering_path,
        "speed": speed_path,
        "curvature": curvature_path,
        "solver_time": solver_time_path
    }


def summarize_mpc_test_run(
    samples: Sequence[MPCTelemetrySample],
    termination_reason: str,
    route_completed: bool,
    collision_count: int,
    route_details: RouteDetails,
    adapter: CarlaControllerAdapter,
    controller: MPCController,
    csv_path: Path,
    plot_paths: dict[str, Path]
) -> MPCCarlaTestSummary:
    """Compute final Constrained MPC tracking, timing, constraint, and safety metrics."""
    if not samples:
        raise ValueError("Cannot summarize an empty sample list.")

    speeds = np.array([s.speed_mps for s in samples], dtype=np.float64)
    rear_errs = np.array([s.rear_axle_tracking_error_m for s in samples], dtype=np.float64)
    front_ctes = np.array([s.front_cross_track_error_m for s in samples], dtype=np.float64)
    heading_deg = np.array([s.heading_error_deg for s in samples], dtype=np.float64)
    req_rate_deg_s = np.array([s.requested_steering_rate_deg_s for s in samples], dtype=np.float64)
    app_steer_rad = np.array([s.applied_steering_rad for s in samples], dtype=np.float64)
    app_steer_deg = np.array([s.applied_steering_deg for s in samples], dtype=np.float64)
    app_rate_rad_s = np.array([s.applied_steering_rate_rad_s for s in samples], dtype=np.float64)
    app_rate_deg_s = np.array([s.applied_steering_rate_deg_s for s in samples], dtype=np.float64)
    carla_steer = np.array([s.normalized_carla_steer for s in samples], dtype=np.float64)
    solver_ms = np.array([s.solver_time_ms for s in samples], dtype=np.float64)
    total_ms = np.array([s.total_controller_time_ms for s in samples], dtype=np.float64)
    fallback_flags = np.array([1 if s.used_fallback else 0 for s in samples], dtype=np.int64)

    angle_violations = int(np.sum(np.abs(app_steer_rad) > MAX_STEERING_RAD + 1e-9))
    rate_violations = int(np.sum(np.abs(app_rate_rad_s) > MAX_STEERING_RATE_RAD_S + 1e-9))
    non_finite_count = sum(0 if is_mpc_sample_finite(s) else 1 for s in samples)
    deadline_misses = int(np.sum(total_ms > CONTROL_PERIOD_MS))

    segment_metrics = calculate_mpc_segment_metrics(
        samples=samples,
        route_details=route_details
    )
    geom = adapter.geometry

    return MPCCarlaTestSummary(
        termination_reason=termination_reason,
        route_completed=bool(route_completed),
        collision_count=int(collision_count),
        total_steps=len(samples),
        simulation_duration_s=float(samples[-1].sim_time_s),
        route_details=route_details,
        wheelbase_m=float(geom.wheelbase),
        rear_axle_offset_x_m=float(geom.rear_axle_offset_x),
        front_axle_offset_x_m=float(geom.front_axle_offset_x),
        max_front_wheel_steer_deg=float(geom.max_front_wheel_steer_deg),
        max_front_wheel_steer_rad=float(geom.max_front_wheel_steer_rad),
        horizon_steps=int(controller.N),
        horizon_duration_s=float(controller.N * controller.dt),
        max_speed_mps=float(np.max(speeds)),
        mean_speed_mps=float(np.mean(speeds)),
        final_speed_mps=float(speeds[-1]),
        rear_axle_rmse_m=float(np.sqrt(np.mean(rear_errs ** 2))),
        max_rear_axle_error_m=float(np.max(np.abs(rear_errs))),
        final_rear_axle_error_m=float(abs(rear_errs[-1])),
        front_cte_rmse_m=float(np.sqrt(np.mean(front_ctes ** 2))),
        max_front_cte_m=float(np.max(np.abs(front_ctes))),
        final_front_cte_m=float(abs(front_ctes[-1])),
        max_heading_error_deg=float(np.max(np.abs(heading_deg))),
        final_heading_error_deg=float(abs(heading_deg[-1])),
        max_requested_rate_deg_s=float(np.max(np.abs(req_rate_deg_s))),
        max_applied_steering_deg=float(np.max(np.abs(app_steer_deg))),
        steering_rms_deg=float(np.sqrt(np.mean(app_steer_deg ** 2))),
        max_steering_rate_deg_s=float(np.max(np.abs(app_rate_deg_s))),
        steering_rate_rms_deg_s=float(np.sqrt(np.mean(app_rate_deg_s ** 2))),
        max_normalized_carla_steer=float(np.max(np.abs(carla_steer))),
        mean_solver_time_ms=float(np.mean(solver_ms)),
        max_solver_time_ms=float(np.max(solver_ms)),
        mean_total_controller_time_ms=float(np.mean(total_ms)),
        max_total_controller_time_ms=float(np.max(total_ms)),
        solver_failure_count=int(controller.solver_failure_count),
        fallback_step_count=int(np.sum(fallback_flags)),
        deadline_misses_50ms=deadline_misses,
        angle_constraint_violations=angle_violations,
        rate_constraint_violations=rate_violations,
        non_finite_values=int(non_finite_count),
        csv_path=csv_path,
        plot_paths=plot_paths,
        segment_metrics=segment_metrics
    )


# ==================================================
# 5. MAIN CLOSED-LOOP CONSTRAINED MPC CARLA EXECUTION
# ==================================================

def run_mpc_carla_test(
    host: str = CARLA_HOST,
    port: int = CARLA_PORT,
    timeout_s: float = CLIENT_TIMEOUT_S,
    target_speed_mps: float = TARGET_SPEED_MPS,
    fixed_delta_seconds: float = FIXED_DELTA_SECONDS,
    horizon_steps: int = MPC_HORIZON_STEPS,
    route_distance_m: float = ROUTE_DISTANCE_M,
    waypoint_spacing_m: float = WAYPOINT_SPACING_M,
    max_simulation_duration_s: float = MAX_SIMULATION_DURATION_S,
    results_dir: Path = RESULTS_DIR,
    file_prefix: str = DEFAULT_MPC_FILE_PREFIX
) -> MPCCarlaTestSummary:
    """
    Connect to CARLA 0.9.16, configure synchronous mode, spawn a Tesla Model 3
    on the validated extended curved route (Spawn Point #61), run the closed-loop
    Constrained MPC lane-keeping controller with N=15 curvature preview until
    completion or safety stop, and always restore world settings and destroy
    all spawned actors.
    """
    if carla is None:
        raise RuntimeError("CARLA Python API (carla) is not available.")

    client = carla.Client(host, port)
    client.set_timeout(timeout_s)

    world = client.get_world()
    original_settings = world.get_settings()

    vehicle: Any = None
    collision_sensor: Any = None
    collision_events: list[Any] = []

    try:
        # 1. Configure synchronous simulation mode with fixed_delta_seconds = 0.05
        sim_settings = world.get_settings()
        sim_settings.synchronous_mode = True
        sim_settings.fixed_delta_seconds = float(fixed_delta_seconds)
        world.apply_settings(sim_settings)

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(VEHICLE_BLUEPRINT_ID)
        collision_bp = blueprint_library.find(COLLISION_SENSOR_BP_ID)

        # 2. Select the validated extended route (Spawn Point #61) and spawn Tesla Model 3
        vehicle, waypoints, route_details = select_spawn_point_and_route(
            world=world,
            vehicle_bp=vehicle_bp,
            route_distance_m=route_distance_m,
            spacing_m=waypoint_spacing_m,
            max_step_yaw_deg=MAX_STEP_YAW_CHANGE_DEG,
            max_total_yaw_deg=MAX_TOTAL_ROUTE_YAW_DEG,
            preferred_spawn_point_index=PREFERRED_SPAWN_POINT_INDEX
        )
        verify_route_continuity(
            waypoints=waypoints,
            expected_spacing_m=waypoint_spacing_m,
            max_step_yaw_deg=MAX_STEP_YAW_CHANGE_DEG,
            allow_contiguous_road_id=(route_distance_m >= 120.0)
        )

        # 3. Disable autopilot and settle vehicle on road surface
        vehicle.set_autopilot(False)
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=False,
                reverse=False
            )
        )
        for _ in range(SETTLE_TICKS):
            world.tick()
        update_spectator_camera(world, vehicle)

        # 4. Attach collision sensor
        collision_sensor = world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=vehicle
        )
        collision_sensor.listen(lambda event: collision_events.append(event))

        # 5. Retrieve live vehicle physics and initialize CarlaControllerAdapter
        adapter = CarlaControllerAdapter(vehicle)
        road_x, road_y, road_yaw = adapter.convert_waypoints(waypoints)
        profile = build_reference_route_profile(
            road_x=road_x,
            road_y=road_y,
            road_yaw=road_yaw
        )
        draw_route_markers(world, waypoints)

        # 6. Initialize Constrained MPCController with live wheelbase and dt = 0.05 s
        mpc_controller = MPCController(
            velocity=target_speed_mps,
            wheelbase=adapter.wheelbase,
            dt=fixed_delta_seconds,
            horizon=horizon_steps,
            max_steering=MAX_STEERING_RAD,
            max_steering_rate=MAX_STEERING_RATE_RAD_S,
            warm_start=True
        )
        speed_controller = ConservativeLongitudinalController(
            target_speed_mps=target_speed_mps
        )

        max_steps = int(round(max_simulation_duration_s / fixed_delta_seconds))
        applied_steering_rad: float = 0.0
        current_min_index: int = 0
        consecutive_failures: int = 0
        samples: list[MPCTelemetrySample] = []
        termination_reason: str = "max_duration_reached"
        route_completed: bool = False

        print("=" * 102)
        print("CARLA 0.9.16 CLOSED-LOOP CONSTRAINED MPC LANE-KEEPING TEST")
        print("=" * 102)
        print(f"Map                      : {world.get_map().name}")
        print(f"Vehicle Blueprint        : {vehicle.type_id} (ID: {vehicle.id})")
        print(
            f"Route                    : SpawnPoint #{route_details.spawn_point_index} | "
            f"Road {route_details.road_id}, Lane {route_details.lane_id} | "
            f"{route_details.num_waypoints} wps ({route_details.total_length_m:.2f} m)"
        )
        print(
            f"Route Geometry           : Straight = {route_details.straight_length_m:.2f} m | "
            f"Curved = {route_details.curved_length_m:.2f} m | "
            f"Max |kappa| = {route_details.max_abs_curvature_1pm:.5f} m^-1 "
            f"(R_min = {route_details.min_turn_radius_m:.2f} m)"
        )
        print(
            f"Live Physics Geometry    : Wheelbase = {adapter.wheelbase:.4f} m | "
            f"Rear-Axle Offset = {adapter.geometry.rear_axle_offset_x:+.4f} m | "
            f"Max Front Steer = {adapter.geometry.max_front_wheel_steer_deg:.2f} deg"
        )
        print(
            f"MPC Configuration        : N = {mpc_controller.N} ({mpc_controller.N * fixed_delta_seconds:.2f} s preview) | "
            f"Target Speed = {target_speed_mps:.1f} m/s | dt = {fixed_delta_seconds:.2f} s"
        )
        print(
            f"Hard QP Constraints      : Max Steer = +/-{MAX_STEERING_DEG:.0f} deg | "
            f"Max Rate = +/-{MAX_STEERING_RATE_DEG_S:.0f} deg/s"
        )
        print("-" * 102)

        for step in range(max_steps + 1):
            sim_time_s = step * fixed_delta_seconds

            sample = compute_mpc_control_step(
                adapter=adapter,
                controller=mpc_controller,
                speed_controller=speed_controller,
                profile=profile,
                previous_applied_steering_rad=applied_steering_rad,
                min_index=current_min_index,
                consecutive_solver_failures=consecutive_failures,
                dt=fixed_delta_seconds,
                step=step,
                sim_time_s=sim_time_s
            )
            samples.append(sample)
            applied_steering_rad = sample.applied_steering_rad
            current_min_index = sample.nearest_index
            consecutive_failures = sample.consecutive_solver_failures

            should_stop, reason, completed = check_mpc_stopping_conditions(
                sample=sample,
                road_x=road_x,
                road_y=road_y,
                collision_count=len(collision_events),
                total_solver_failures=mpc_controller.solver_failure_count,
                max_consecutive_failures=MAX_CONSECUTIVE_SOLVER_FAILURES,
                max_total_failures=MAX_TOTAL_SOLVER_FAILURES,
                max_duration_s=max_simulation_duration_s,
                max_cte_m=MAX_ALLOWED_CTE_M,
                endpoint_tolerance_m=ENDPOINT_TOLERANCE_M
            )

            if step % 20 == 0 or should_stop:
                print(
                    f"[t={sample.sim_time_s:5.2f}s | s={sample.route_progress_m:6.2f}m ({sample.route_progress_pct:5.1f}%)] "
                    f"v={sample.speed_mps:4.2f}m/s | "
                    f"e_y={sample.rear_axle_tracking_error_m:+.4f}m | "
                    f"f_cte={sample.front_cross_track_error_m:+.4f}m | "
                    f"steer={sample.applied_steering_deg:+.2f}deg | "
                    f"rate={sample.applied_steering_rate_deg_s:+.2f}deg/s | "
                    f"qp={sample.solver_time_ms:.2f}/{sample.total_controller_time_ms:.2f}ms"
                )

            if should_stop:
                termination_reason = reason
                route_completed = completed
                break

            # Apply vehicle control via CarlaControllerAdapter
            control_cmd = adapter.to_vehicle_control(
                controller_steering_angle=sample.applied_steering_rad,
                throttle=sample.throttle,
                brake=sample.brake,
                hand_brake=False,
                reverse=False
            )
            vehicle.apply_control(control_cmd)
            world.tick()
            update_spectator_camera(world, vehicle)

        # Bring vehicle to a controlled stop before cleanup
        stop_control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True,
            reverse=False
        )
        vehicle.apply_control(stop_control)
        for _ in range(STOP_BRAKE_TICKS):
            world.tick()
            update_spectator_camera(world, vehicle)

        csv_path = save_mpc_telemetry_to_csv(
            samples=samples,
            csv_path=results_dir / f"{file_prefix}_telemetry.csv"
        )
        plot_paths = generate_mpc_carla_plots(
            samples=samples,
            profile=profile,
            output_dir=results_dir,
            target_speed_mps=target_speed_mps,
            file_prefix=file_prefix
        )

        summary = summarize_mpc_test_run(
            samples=samples,
            termination_reason=termination_reason,
            route_completed=route_completed,
            collision_count=len(collision_events),
            route_details=route_details,
            adapter=adapter,
            controller=mpc_controller,
            csv_path=csv_path,
            plot_paths=plot_paths
        )

        print("-" * 102)
        print(f"Termination Reason       : {summary.termination_reason}")
        print(f"Route Completed          : {summary.route_completed}")
        print(f"Collision Count          : {summary.collision_count}")
        print(f"Simulation Duration      : {summary.simulation_duration_s:.2f} s ({summary.total_steps} steps)")
        print(f"Max / Mean Speed         : {summary.max_speed_mps:.4f} m/s / {summary.mean_speed_mps:.4f} m/s")
        print(f"Rear-Axle Tracking RMSE  : {summary.rear_axle_rmse_m:.4f} m (Max: {summary.max_rear_axle_error_m:.4f} m)")
        print(f"Front-Axle CTE RMSE      : {summary.front_cte_rmse_m:.4f} m (Max: {summary.max_front_cte_m:.4f} m)")
        print(
            f"Steering RMS / Max       : {summary.steering_rms_deg:.4f} deg / {summary.max_applied_steering_deg:.4f} deg"
        )
        print(
            f"Steering Rate RMS / Max  : {summary.steering_rate_rms_deg_s:.4f} deg/s / {summary.max_steering_rate_deg_s:.4f} deg/s"
        )
        print(
            f"OSQP Solver Time         : Mean = {summary.mean_solver_time_ms:.4f} ms | Max = {summary.max_solver_time_ms:.4f} ms"
        )
        print(
            f"Total Controller Time    : Mean = {summary.mean_total_controller_time_ms:.4f} ms | "
            f"Max = {summary.max_total_controller_time_ms:.4f} ms | "
            f"Deadline Misses (>50 ms) = {summary.deadline_misses_50ms}"
        )
        print(
            f"Solver Failures          : {summary.solver_failure_count} (Fallback Steps = {summary.fallback_step_count})"
        )
        print(
            f"Constraint Violations    : Angle={summary.angle_constraint_violations}, "
            f"Rate={summary.rate_constraint_violations}, NonFinite={summary.non_finite_values}"
        )
        print("-" * 102)
        print(
            f"{'Segment':<20} | {'Samples':>7} | {'Dist (m)':>8} | "
            f"{'Rear RMSE':>10} | {'Max |Rear|':>10} | {'Front RMSE':>10} | "
            f"{'Max |Front|':>11} | {'Steer RMS':>10} | {'Max Rate':>10}"
        )
        print("-" * 102)
        for seg_key in (
            "straight_approach",
            "sustained_curve",
            "straight_exit",
            "straight_combined",
            "overall"
        ):
            sm = summary.segment_metrics[seg_key]
            print(
                f"{sm.segment_name:<20} | {sm.sample_count:7d} | {sm.distance_m:8.2f} | "
                f"{sm.rear_axle_rmse_m:10.4f} | {sm.max_rear_axle_error_m:10.4f} | "
                f"{sm.front_cte_rmse_m:10.4f} | {sm.max_front_cte_m:11.4f} | "
                f"{sm.steering_rms_deg:10.4f} | {sm.max_steering_rate_deg_s:10.4f}"
            )
        print("-" * 102)
        print(f"Telemetry CSV Saved      : {summary.csv_path}")
        print("=" * 102)

        return summary

    finally:
        # Always destroy collision sensor and vehicle actor, then restore world settings
        if collision_sensor is not None:
            try:
                collision_sensor.stop()
            except Exception:
                pass
            try:
                if collision_sensor.is_alive:
                    collision_sensor.destroy()
            except Exception:
                pass

        if vehicle is not None:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except Exception:
                pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


def main() -> None:
    """Entry point for running the closed-loop Constrained MPC CARLA test."""
    run_mpc_carla_test()


if __name__ == "__main__":
    main()
