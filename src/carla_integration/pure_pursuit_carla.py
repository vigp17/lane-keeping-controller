"""
Closed-Loop Pure Pursuit Lane-Keeping Controller Integration & Live Test in CARLA 0.9.16.

Connects the existing geometric PurePursuitController and PIDController to a live
Tesla Model 3 in CARLA 0.9.16 (Town10HD_Opt) using CarlaControllerAdapter for
all physics-derived vehicle geometry, rear-axle coordinate transformations, and
steering conversions on the validated 135.24-meter extended curved route.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
import sys
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

from carla_integration.carla_adapter import CarlaControllerAdapter
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
    apply_steering_constraints,
    compute_route_arc_length_and_curvature,
    draw_route_markers,
    select_spawn_point_and_route,
    update_spectator_camera,
    verify_route_continuity,
)
from controllers.pure_pursuit_controller import PurePursuitController
from controllers.stanley_controller import StanleyController
from vehicle import Vehicle


# ==================================================
# 1. PURE PURSUIT CARLA CONFIGURATION
# ==================================================

MIN_LOOKAHEAD_M: float = 3.0
LOOKAHEAD_GAIN_S: float = 0.5
FIXED_LOOKAHEAD_M: float = 10.0
USE_DYNAMIC_LOOKAHEAD: bool = True

STRAIGHT_APPROACH_MAX_S_M: float = 57.50

DEFAULT_PP_FILE_PREFIX: str = "pure_pursuit_carla_8ms"
DEFAULT_PP_CSV_PATH: Path = RESULTS_DIR / f"{DEFAULT_PP_FILE_PREFIX}_telemetry.csv"


# ==================================================
# 2. DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class PurePursuitTelemetrySample:
    """Single closed-loop Pure Pursuit CARLA telemetry sample."""

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
    target_index: int
    target_x_m: float
    target_y_m: float
    desired_lookahead_m: float
    actual_lookahead_m: float
    alpha_rad: float
    alpha_deg: float
    rear_axle_tracking_error_m: float
    front_cross_track_error_m: float
    heading_error_rad: float
    heading_error_deg: float
    requested_steering_rad: float
    requested_steering_deg: float
    angle_limited_steering_rad: float
    applied_steering_rad: float
    applied_steering_deg: float
    steering_rate_rad_s: float
    steering_rate_deg_s: float
    normalized_carla_steer: float
    throttle: float
    brake: float
    road_curvature_1pm: float


@dataclass(frozen=True)
class PurePursuitCarlaTestSummary:
    """Summary metrics and verification results from the Pure Pursuit CARLA run."""

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
    min_lookahead_m: float
    lookahead_gain_s: float
    mean_desired_lookahead_m: float
    mean_actual_lookahead_m: float
    min_actual_lookahead_m: float
    max_actual_lookahead_m: float
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
    max_requested_steering_deg: float
    max_applied_steering_deg: float
    steering_rms_deg: float
    max_steering_rate_deg_s: float
    steering_rate_rms_deg_s: float
    max_normalized_carla_steer: float
    angle_constraint_violations: int
    rate_constraint_violations: int
    non_finite_values: int
    csv_path: Path
    plot_paths: dict[str, Path]
    segment_metrics: dict[str, SegmentPerformanceMetrics]


# ==================================================
# 3. SINGLE-STEP CONTROL & STOPPING CONDITIONS
# ==================================================

def compute_signed_cross_track_error(
    point_x: float,
    point_y: float,
    ref_x: float,
    ref_y: float,
    ref_yaw: float
) -> float:
    """
    Compute signed lateral tracking error relative to the path's local left
    unit normal n = [-sin(ref_yaw), cos(ref_yaw)], matching StanleyController.
    """
    normal_x = -math.sin(ref_yaw)
    normal_y = math.cos(ref_yaw)
    error_x = ref_x - point_x
    error_y = ref_y - point_y
    return float(error_x * normal_x + error_y * normal_y)


def compute_pure_pursuit_control_step(
    adapter: CarlaControllerAdapter,
    controller: PurePursuitController,
    speed_controller: ConservativeLongitudinalController,
    road_x: NDArray[np.float64],
    road_y: NDArray[np.float64],
    road_yaw: NDArray[np.float64],
    previous_applied_steering_rad: float,
    min_index: int,
    dt: float,
    step: int,
    sim_time_s: float,
    road_s: NDArray[np.float64] | None = None,
    road_curvature: NDArray[np.float64] | None = None
) -> PurePursuitTelemetrySample:
    """
    Execute one closed-loop Pure Pursuit + longitudinal control calculation
    using live rear-axle pose and physics-derived geometry from CarlaControllerAdapter.
    """
    rear_x, rear_y, yaw_rad = adapter.get_rear_axle_pose()
    speed_mps = adapter.get_speed()

    if road_s is None or road_curvature is None:
        road_s, road_curvature = compute_route_arc_length_and_curvature(
            road_x=road_x,
            road_y=road_y,
            road_yaw=road_yaw
        )

    # Construct kinematic Vehicle at the rear axle with live physics wheelbase
    kinematic_vehicle = Vehicle(
        x=rear_x,
        y=rear_y,
        yaw=yaw_rad,
        wheelbase=adapter.wheelbase
    )
    front_x, front_y = kinematic_vehicle.get_front_axle_position()

    pp_details = controller.compute_with_details(
        vehicle=kinematic_vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=speed_mps,
        use_dynamic_lookahead=controller.use_dynamic_lookahead,
        min_index=min_index
    )

    requested_steering_rad = float(pp_details["requested_steering"])
    nearest_index = int(pp_details["nearest_index"])
    target_index = int(pp_details["target_index"])
    target_x_m = float(pp_details["target_x"])
    target_y_m = float(pp_details["target_y"])
    desired_lookahead_m = float(pp_details["desired_lookahead_distance"])
    actual_lookahead_m = float(pp_details["actual_lookahead_distance"])
    alpha_rad = float(pp_details["heading_to_target_error"])

    # Rear-axle reference waypoint & tracking error
    ref_x_m = float(road_x[nearest_index])
    ref_y_m = float(road_y[nearest_index])
    ref_yaw_rad = float(road_yaw[nearest_index])
    rear_tracking_error_m = compute_signed_cross_track_error(
        point_x=rear_x,
        point_y=rear_y,
        ref_x=ref_x_m,
        ref_y=ref_y_m,
        ref_yaw=ref_yaw_rad
    )

    # Front-axle nearest waypoint & cross-track error (matching Stanley definition)
    front_dx = road_x - front_x
    front_dy = road_y - front_y
    front_nearest_index = int(np.argmin(front_dx ** 2 + front_dy ** 2))
    front_ref_x = float(road_x[front_nearest_index])
    front_ref_y = float(road_y[front_nearest_index])
    front_ref_yaw = float(road_yaw[front_nearest_index])
    front_cte_m = compute_signed_cross_track_error(
        point_x=front_x,
        point_y=front_y,
        ref_x=front_ref_x,
        ref_y=front_ref_y,
        ref_yaw=front_ref_yaw
    )

    # Heading error relative to local path tangent (matching Stanley definition)
    heading_error_rad = StanleyController.normalize_angle(ref_yaw_rad - yaw_rad)

    # Enforce hard +/-30 deg steering angle and +/-20 deg/s steering rate limits
    applied_steering_rad, angle_limited_rad, steering_rate_rad_s = (
        apply_steering_constraints(
            requested_steering_rad=requested_steering_rad,
            previous_applied_steering_rad=previous_applied_steering_rad,
            dt=dt,
            max_steering_rad=MAX_STEERING_RAD,
            max_steering_rate_rad_s=MAX_STEERING_RATE_RAD_S
        )
    )

    # Convert controller steering radians (+left) to normalized CARLA steer (+right)
    normalized_carla_steer = adapter.to_carla_steer(applied_steering_rad)

    # Compute longitudinal throttle and brake targeting 8.0 m/s
    throttle, brake = speed_controller.compute(
        current_speed_mps=speed_mps,
        dt=dt
    )

    # Progress and curvature indexed consistently with Stanley (front-axle progress along route)
    total_route_len_m = max(float(road_s[-1]), 1e-6)
    route_progress_m = float(road_s[front_nearest_index])
    route_progress_pct = float(
        np.clip(100.0 * route_progress_m / total_route_len_m, 0.0, 100.0)
    )
    road_curvature_1pm = float(road_curvature[front_nearest_index])

    return PurePursuitTelemetrySample(
        step=int(step),
        sim_time_s=float(sim_time_s),
        route_progress_m=route_progress_m,
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
        target_index=target_index,
        target_x_m=target_x_m,
        target_y_m=target_y_m,
        desired_lookahead_m=desired_lookahead_m,
        actual_lookahead_m=actual_lookahead_m,
        alpha_rad=alpha_rad,
        alpha_deg=float(math.degrees(alpha_rad)),
        rear_axle_tracking_error_m=rear_tracking_error_m,
        front_cross_track_error_m=front_cte_m,
        heading_error_rad=heading_error_rad,
        heading_error_deg=float(math.degrees(heading_error_rad)),
        requested_steering_rad=requested_steering_rad,
        requested_steering_deg=float(math.degrees(requested_steering_rad)),
        angle_limited_steering_rad=float(angle_limited_rad),
        applied_steering_rad=float(applied_steering_rad),
        applied_steering_deg=float(math.degrees(applied_steering_rad)),
        steering_rate_rad_s=float(steering_rate_rad_s),
        steering_rate_deg_s=float(math.degrees(steering_rate_rad_s)),
        normalized_carla_steer=float(normalized_carla_steer),
        throttle=float(throttle),
        brake=float(brake),
        road_curvature_1pm=road_curvature_1pm
    )


def is_pp_sample_finite(sample: PurePursuitTelemetrySample) -> bool:
    """Return True if all numerical fields in the telemetry sample are finite."""
    for value in asdict(sample).values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def check_pure_pursuit_stopping_conditions(
    sample: PurePursuitTelemetrySample,
    road_x: NDArray[np.float64],
    road_y: NDArray[np.float64],
    collision_count: int,
    max_duration_s: float = MAX_SIMULATION_DURATION_S,
    max_cte_m: float = MAX_ALLOWED_CTE_M,
    endpoint_tolerance_m: float = ENDPOINT_TOLERANCE_M
) -> tuple[bool, str, bool]:
    """
    Evaluate all safety and mission stopping conditions for Pure Pursuit.

    Returns:
        Tuple of (should_stop, termination_reason, route_completed).
    """
    if not is_pp_sample_finite(sample):
        return True, "invalid_numerical_values", False

    if collision_count > 0:
        return True, "collision_detected", False

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

def _compute_pp_subset_metrics(
    segment_name: str,
    subset: Sequence[PurePursuitTelemetrySample],
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
    steer_rate_deg_s = np.array([s.steering_rate_deg_s for s in subset], dtype=np.float64)

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


def calculate_pure_pursuit_segment_metrics(
    samples: Sequence[PurePursuitTelemetrySample],
    route_details: RouteDetails | None = None,
    curvature_threshold_1pm: float = CURVATURE_THRESHOLD_1PM,
    approach_max_s_m: float = STRAIGHT_APPROACH_MAX_S_M
) -> dict[str, SegmentPerformanceMetrics]:
    """
    Separate Pure Pursuit performance metrics into:
        - "straight_approach": Straight approach segment (s < 57.50 m, |kappa| < threshold)
        - "sustained_curve": Sustained curved segment (|kappa| >= threshold)
        - "straight_exit": Straight exit segment (s >= 57.50 m, |kappa| < threshold)
        - "straight_combined": All straight samples combined
        - "overall": Overall route (all samples)
    """
    if not samples:
        raise ValueError("Cannot calculate segment metrics from empty samples.")

    approach_samples = [
        s for s in samples
        if abs(s.road_curvature_1pm) < curvature_threshold_1pm
        and s.route_progress_m <= approach_max_s_m + 0.5
    ]
    curve_samples = [
        s for s in samples
        if abs(s.road_curvature_1pm) >= curvature_threshold_1pm
    ]
    exit_samples = [
        s for s in samples
        if abs(s.road_curvature_1pm) < curvature_threshold_1pm
        and s.route_progress_m > approach_max_s_m + 0.5
    ]
    straight_all_samples = [
        s for s in samples
        if abs(s.road_curvature_1pm) < curvature_threshold_1pm
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
        "straight_approach": _compute_pp_subset_metrics(
            "Straight Approach", approach_samples, approach_dist
        ),
        "sustained_curve": _compute_pp_subset_metrics(
            "Sustained Curve", curve_samples, curved_dist
        ),
        "straight_exit": _compute_pp_subset_metrics(
            "Straight Exit", exit_samples, exit_dist
        ),
        "straight_combined": _compute_pp_subset_metrics(
            "Straight Combined", straight_all_samples, straight_total_dist
        ),
        "overall": _compute_pp_subset_metrics(
            "Overall Route", samples, overall_dist
        ),
    }


def save_pure_pursuit_telemetry_to_csv(
    samples: Sequence[PurePursuitTelemetrySample],
    csv_path: Path = DEFAULT_PP_CSV_PATH
) -> Path:
    """Save recorded Pure Pursuit telemetry samples to a CSV file."""
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


def generate_pure_pursuit_carla_plots(
    samples: Sequence[PurePursuitTelemetrySample],
    road_x: NDArray[np.float64],
    road_y: NDArray[np.float64],
    output_dir: Path = RESULTS_DIR,
    target_speed_mps: float = TARGET_SPEED_MPS,
    file_prefix: str = DEFAULT_PP_FILE_PREFIX
) -> dict[str, Path]:
    """
    Generate and save the 5 required closed-loop Pure Pursuit CARLA plots:
        1. results/pure_pursuit_carla_8ms_trajectory.png
        2. results/pure_pursuit_carla_8ms_error.png
        3. results/pure_pursuit_carla_8ms_steering.png
        4. results/pure_pursuit_carla_8ms_speed.png
        5. results/pure_pursuit_carla_8ms_lookahead.png
    """
    if not samples:
        raise ValueError("Cannot generate plots from empty telemetry samples.")

    output_dir.mkdir(parents=True, exist_ok=True)

    time_s = np.array([s.sim_time_s for s in samples], dtype=np.float64)
    progress_m = np.array([s.route_progress_m for s in samples], dtype=np.float64)
    curvature_1pm = np.array([s.road_curvature_1pm for s in samples], dtype=np.float64)
    rear_x = np.array([s.rear_x_m for s in samples], dtype=np.float64)
    rear_y = np.array([s.rear_y_m for s in samples], dtype=np.float64)
    front_x = np.array([s.front_x_m for s in samples], dtype=np.float64)
    front_y = np.array([s.front_y_m for s in samples], dtype=np.float64)
    rear_err = np.array([s.rear_axle_tracking_error_m for s in samples], dtype=np.float64)
    front_cte = np.array([s.front_cross_track_error_m for s in samples], dtype=np.float64)
    req_steer_deg = np.array([s.requested_steering_deg for s in samples], dtype=np.float64)
    app_steer_deg = np.array([s.applied_steering_deg for s in samples], dtype=np.float64)
    carla_steer = np.array([s.normalized_carla_steer for s in samples], dtype=np.float64)
    speed_mps = np.array([s.speed_mps for s in samples], dtype=np.float64)
    desired_ld = np.array([s.desired_lookahead_m for s in samples], dtype=np.float64)
    actual_ld = np.array([s.actual_lookahead_m for s in samples], dtype=np.float64)

    trajectory_path = output_dir / f"{file_prefix}_trajectory.png"
    error_path = output_dir / f"{file_prefix}_error.png"
    steering_path = output_dir / f"{file_prefix}_steering.png"
    speed_path = output_dir / f"{file_prefix}_speed.png"
    lookahead_path = output_dir / f"{file_prefix}_lookahead.png"

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
        road_x,
        road_y,
        "k--",
        linewidth=2.0,
        label="Reference Driving Lane (Controller Frame)"
    )
    ax.plot(
        rear_x,
        rear_y,
        color="tab:blue",
        linewidth=2.0,
        label="Pure Pursuit Rear-Axle Trajectory"
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
    ax.set_title("CARLA 0.9.16 Closed-Loop Pure Pursuit Trajectory (Tesla Model 3, Town10HD_Opt)")
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
        label="Rear-Axle Tracking Error (m)"
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
    ax_t.set_title("CARLA 0.9.16 Pure Pursuit Path-Tracking Error vs. Time and Route Distance")
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
        label="Rear-Axle Tracking Error (m)"
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

    # 3. Steering Angle Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(
        time_s,
        req_steer_deg,
        color="tab:gray",
        linestyle="--",
        linewidth=1.5,
        label="Requested Steering (deg, +Left)"
    )
    ax1.plot(
        time_s,
        app_steer_deg,
        color="tab:blue",
        linewidth=2.0,
        label="Applied Steering (deg, +Left)"
    )
    ax1.plot(
        time_s,
        carla_steer * MAX_STEERING_DEG,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label="Normalized CARLA Steer (scaled x30 deg, +Right)"
    )
    ax1.axhline(MAX_STEERING_DEG, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="+/-30 deg Limit")
    ax1.axhline(-MAX_STEERING_DEG, color="red", linestyle="--", linewidth=1.0, alpha=0.6)
    ax1.set_title("CARLA 0.9.16 Pure Pursuit Steering Commands vs. Time")
    ax1.set_xlabel("Simulation Time (s)")
    ax1.set_ylabel("Steering Angle (deg)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best")
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
    ax.set_title("CARLA 0.9.16 Pure Pursuit Longitudinal Speed Tracking vs. Time")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(speed_path, dpi=150)
    plt.close(fig)

    # 5. Lookahead Distance vs. Time Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        time_s,
        desired_ld,
        color="tab:purple",
        linestyle="--",
        linewidth=2.0,
        label="Desired Lookahead Ld_desired = 3.0 + 0.5*v (m)"
    )
    ax.plot(
        time_s,
        actual_ld,
        color="tab:blue",
        linewidth=1.8,
        label="Actual Euclidean Lookahead Ld (m)"
    )
    ax.set_title("CARLA 0.9.16 Pure Pursuit Speed-Dependent Lookahead Distance vs. Time")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Lookahead Distance (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(lookahead_path, dpi=150)
    plt.close(fig)

    return {
        "trajectory": trajectory_path,
        "error": error_path,
        "steering": steering_path,
        "speed": speed_path,
        "lookahead": lookahead_path
    }


def summarize_pure_pursuit_test_run(
    samples: Sequence[PurePursuitTelemetrySample],
    termination_reason: str,
    route_completed: bool,
    collision_count: int,
    route_details: RouteDetails,
    adapter: CarlaControllerAdapter,
    controller: PurePursuitController,
    csv_path: Path,
    plot_paths: dict[str, Path]
) -> PurePursuitCarlaTestSummary:
    """Compute final Pure Pursuit tracking, lookahead, constraint, and safety metrics."""
    if not samples:
        raise ValueError("Cannot summarize an empty sample list.")

    speeds = np.array([s.speed_mps for s in samples], dtype=np.float64)
    rear_errs = np.array([s.rear_axle_tracking_error_m for s in samples], dtype=np.float64)
    front_ctes = np.array([s.front_cross_track_error_m for s in samples], dtype=np.float64)
    heading_deg = np.array([s.heading_error_deg for s in samples], dtype=np.float64)
    req_steer_deg = np.array([s.requested_steering_deg for s in samples], dtype=np.float64)
    app_steer_rad = np.array([s.applied_steering_rad for s in samples], dtype=np.float64)
    app_steer_deg = np.array([s.applied_steering_deg for s in samples], dtype=np.float64)
    steer_rate_rad_s = np.array([s.steering_rate_rad_s for s in samples], dtype=np.float64)
    steer_rate_deg_s = np.array([s.steering_rate_deg_s for s in samples], dtype=np.float64)
    carla_steer = np.array([s.normalized_carla_steer for s in samples], dtype=np.float64)
    desired_ld = np.array([s.desired_lookahead_m for s in samples], dtype=np.float64)
    actual_ld = np.array([s.actual_lookahead_m for s in samples], dtype=np.float64)

    angle_violations = int(np.sum(np.abs(app_steer_rad) > MAX_STEERING_RAD + 1e-9))
    rate_violations = int(np.sum(np.abs(steer_rate_rad_s) > MAX_STEERING_RATE_RAD_S + 1e-9))
    non_finite_count = sum(0 if is_pp_sample_finite(s) else 1 for s in samples)

    segment_metrics = calculate_pure_pursuit_segment_metrics(
        samples=samples,
        route_details=route_details
    )
    geom = adapter.geometry

    return PurePursuitCarlaTestSummary(
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
        min_lookahead_m=float(controller.min_lookahead),
        lookahead_gain_s=float(controller.lookahead_gain),
        mean_desired_lookahead_m=float(np.mean(desired_ld)),
        mean_actual_lookahead_m=float(np.mean(actual_ld)),
        min_actual_lookahead_m=float(np.min(actual_ld)),
        max_actual_lookahead_m=float(np.max(actual_ld)),
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
        max_requested_steering_deg=float(np.max(np.abs(req_steer_deg))),
        max_applied_steering_deg=float(np.max(np.abs(app_steer_deg))),
        steering_rms_deg=float(np.sqrt(np.mean(app_steer_deg ** 2))),
        max_steering_rate_deg_s=float(np.max(np.abs(steer_rate_deg_s))),
        steering_rate_rms_deg_s=float(np.sqrt(np.mean(steer_rate_deg_s ** 2))),
        max_normalized_carla_steer=float(np.max(np.abs(carla_steer))),
        angle_constraint_violations=angle_violations,
        rate_constraint_violations=rate_violations,
        non_finite_values=int(non_finite_count),
        csv_path=csv_path,
        plot_paths=plot_paths,
        segment_metrics=segment_metrics
    )


# ==================================================
# 5. MAIN CLOSED-LOOP PURE PURSUIT CARLA EXECUTION
# ==================================================

def run_pure_pursuit_carla_test(
    host: str = CARLA_HOST,
    port: int = CARLA_PORT,
    timeout_s: float = CLIENT_TIMEOUT_S,
    target_speed_mps: float = TARGET_SPEED_MPS,
    fixed_delta_seconds: float = FIXED_DELTA_SECONDS,
    route_distance_m: float = ROUTE_DISTANCE_M,
    waypoint_spacing_m: float = WAYPOINT_SPACING_M,
    max_simulation_duration_s: float = MAX_SIMULATION_DURATION_S,
    min_lookahead_m: float = MIN_LOOKAHEAD_M,
    lookahead_gain_s: float = LOOKAHEAD_GAIN_S,
    use_dynamic_lookahead: bool = USE_DYNAMIC_LOOKAHEAD,
    results_dir: Path = RESULTS_DIR,
    file_prefix: str = DEFAULT_PP_FILE_PREFIX
) -> PurePursuitCarlaTestSummary:
    """
    Connect to CARLA 0.9.16, configure synchronous mode, spawn a Tesla Model 3
    on the validated extended curved route (Spawn Point #61), run the closed-loop
    Pure Pursuit lane-keeping controller until completion or safety stop, and
    always restore world settings and destroy all spawned actors.
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
        road_s, road_curvature = compute_route_arc_length_and_curvature(
            road_x=road_x,
            road_y=road_y,
            road_yaw=road_yaw
        )
        draw_route_markers(world, waypoints)

        # 6. Initialize PurePursuitController using live physics-derived wheelbase
        pp_controller = PurePursuitController(
            wheelbase=adapter.wheelbase,
            lookahead_distance=FIXED_LOOKAHEAD_M,
            min_lookahead=min_lookahead_m,
            lookahead_gain=lookahead_gain_s,
            use_dynamic_lookahead=use_dynamic_lookahead
        )
        speed_controller = ConservativeLongitudinalController(
            target_speed_mps=target_speed_mps
        )

        max_steps = int(round(max_simulation_duration_s / fixed_delta_seconds))
        applied_steering_rad: float = 0.0
        current_min_index: int = 0
        samples: list[PurePursuitTelemetrySample] = []
        termination_reason: str = "max_duration_reached"
        route_completed: bool = False

        print("=" * 98)
        print("CARLA 0.9.16 CLOSED-LOOP PURE PURSUIT LANE-KEEPING TEST")
        print("=" * 98)
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
            f"Pure Pursuit Config      : Ld_desired = {min_lookahead_m:.1f} + {lookahead_gain_s:.1f}*v "
            f"({min_lookahead_m + lookahead_gain_s * target_speed_mps:.1f} m at {target_speed_mps:.1f} m/s) | "
            f"dt = {fixed_delta_seconds:.2f} s | "
            f"Max Steer = +/-{MAX_STEERING_DEG:.0f} deg | "
            f"Max Rate = +/-{MAX_STEERING_RATE_DEG_S:.0f} deg/s"
        )
        print("-" * 98)

        for step in range(max_steps + 1):
            sim_time_s = step * fixed_delta_seconds

            sample = compute_pure_pursuit_control_step(
                adapter=adapter,
                controller=pp_controller,
                speed_controller=speed_controller,
                road_x=road_x,
                road_y=road_y,
                road_yaw=road_yaw,
                previous_applied_steering_rad=applied_steering_rad,
                min_index=current_min_index,
                dt=fixed_delta_seconds,
                step=step,
                sim_time_s=sim_time_s,
                road_s=road_s,
                road_curvature=road_curvature
            )
            samples.append(sample)
            applied_steering_rad = sample.applied_steering_rad
            current_min_index = sample.nearest_index

            should_stop, reason, completed = check_pure_pursuit_stopping_conditions(
                sample=sample,
                road_x=road_x,
                road_y=road_y,
                collision_count=len(collision_events),
                max_duration_s=max_simulation_duration_s,
                max_cte_m=MAX_ALLOWED_CTE_M,
                endpoint_tolerance_m=ENDPOINT_TOLERANCE_M
            )

            if step % 20 == 0 or should_stop:
                print(
                    f"[t={sample.sim_time_s:5.2f}s | s={sample.route_progress_m:6.2f}m ({sample.route_progress_pct:5.1f}%)] "
                    f"v={sample.speed_mps:4.2f}m/s | "
                    f"Ld={sample.desired_lookahead_m:4.2f}/{sample.actual_lookahead_m:4.2f}m (idx={sample.target_index:3d}) | "
                    f"rear_err={sample.rear_axle_tracking_error_m:+.4f}m | "
                    f"front_cte={sample.front_cross_track_error_m:+.4f}m | "
                    f"steer={sample.applied_steering_deg:+.2f}deg"
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

        csv_path = save_pure_pursuit_telemetry_to_csv(
            samples=samples,
            csv_path=results_dir / f"{file_prefix}_telemetry.csv"
        )
        plot_paths = generate_pure_pursuit_carla_plots(
            samples=samples,
            road_x=road_x,
            road_y=road_y,
            output_dir=results_dir,
            target_speed_mps=target_speed_mps,
            file_prefix=file_prefix
        )

        summary = summarize_pure_pursuit_test_run(
            samples=samples,
            termination_reason=termination_reason,
            route_completed=route_completed,
            collision_count=len(collision_events),
            route_details=route_details,
            adapter=adapter,
            controller=pp_controller,
            csv_path=csv_path,
            plot_paths=plot_paths
        )

        print("-" * 98)
        print(f"Termination Reason       : {summary.termination_reason}")
        print(f"Route Completed          : {summary.route_completed}")
        print(f"Collision Count          : {summary.collision_count}")
        print(f"Simulation Duration      : {summary.simulation_duration_s:.2f} s ({summary.total_steps} steps)")
        print(f"Max / Mean Speed         : {summary.max_speed_mps:.4f} m/s / {summary.mean_speed_mps:.4f} m/s")
        print(
            f"Lookahead (Des / Act)    : Mean Desired = {summary.mean_desired_lookahead_m:.4f} m | "
            f"Mean Actual = {summary.mean_actual_lookahead_m:.4f} m "
            f"[{summary.min_actual_lookahead_m:.2f}, {summary.max_actual_lookahead_m:.2f}] m"
        )
        print(f"Rear-Axle Tracking RMSE  : {summary.rear_axle_rmse_m:.4f} m (Max: {summary.max_rear_axle_error_m:.4f} m)")
        print(f"Front-Axle CTE RMSE      : {summary.front_cte_rmse_m:.4f} m (Max: {summary.max_front_cte_m:.4f} m)")
        print(
            f"Steering RMS / Max       : {summary.steering_rms_deg:.4f} deg / {summary.max_applied_steering_deg:.4f} deg "
            f"(Max Rate: {summary.max_steering_rate_deg_s:.4f} deg/s)"
        )
        print(
            f"Constraint Violations    : Angle={summary.angle_constraint_violations}, "
            f"Rate={summary.rate_constraint_violations}, NonFinite={summary.non_finite_values}"
        )
        print("-" * 98)
        print(
            f"{'Segment':<20} | {'Samples':>7} | {'Dist (m)':>8} | "
            f"{'Rear RMSE':>10} | {'Max |Rear|':>10} | {'Front RMSE':>10} | "
            f"{'Max |Front|':>11} | {'Steer RMS':>10} | {'Max Rate':>10}"
        )
        print("-" * 98)
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
        print("-" * 98)
        print(f"Telemetry CSV Saved      : {summary.csv_path}")
        print("=" * 98)

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
    """Entry point for running the closed-loop Pure Pursuit CARLA test."""
    run_pure_pursuit_carla_test()


if __name__ == "__main__":
    main()
