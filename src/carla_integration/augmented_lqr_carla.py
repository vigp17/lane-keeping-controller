"""
Phase 11E: Closed-Loop Augmented LQR Lane-Keeping Integration in CARLA 0.9.16.

Connects the existing discrete-time AugmentedLQRController (with steering-rate
input) and ConservativeLongitudinalController to a live Tesla Model 3 in
CARLA 0.9.16 (Town10HD_Opt) using CarlaControllerAdapter for all physics-derived
vehicle geometry, rear-axle coordinate transformations, and steering conversions
on the validated 135.24-meter extended curved route.

Model Assumptions and Feedforward Formulation:
    1. Planar Kinematic Bicycle Linearization:
       The AugmentedLQRController is synthesized around a constant forward speed v
       and live physics-derived wheelbase L (~3.0046 m) with state vector
       z = [e_y, e_psi, delta_applied]^T and control input u = d(delta)/dt.
    2. Sign Conventions (matching AugmentedLQRController):
       - e_y: Actual rear-axle position minus reference position projected onto
         the local road left normal n = [-sin(psi_ref), cos(psi_ref)]^T:
             e_y = -(x_rear - x_ref)*sin(psi_ref) + (y_rear - y_ref)*cos(psi_ref)
         (e_y > 0 when the vehicle is to the left of the lane centerline).
       - e_psi: Vehicle yaw minus reference path heading wrapped to [-pi, pi]:
             e_psi = normalize_angle(psi - psi_ref)
         (e_psi > 0 when the vehicle points counterclockwise/left of path tangent).
    3. Curvature Feedforward & Deviation-State LQR:
       On curved road segments with reference curvature kappa_ref(s), the nominal
       kinematic bicycle equilibrium steering angle is:
             delta_ff(s) = atan(L * kappa_ref(s))
       Defining the steering deviation state delta_error = delta_applied - delta_ff,
       the error state vector is:
             z_error = [e_y, e_psi, delta_error]^T
       and the feedback steering-rate command is:
             u_feedback = -K * z_error
       To track time-varying road curvature without steady-state rate lag, the
       analytical chain-rule derivative of delta_ff along the reference path is
       added to the steering-rate command:
             d(delta_ff)/dt = (d(delta_ff)/ds) * (ds/dt)
                            = [L / (1 + (L * kappa_ref)^2)] * (d(kappa_ref)/ds) * v * cos(e_psi)
             u_total = u_feedback + d(delta_ff)/dt
    4. Reference Curvature Estimation:
       To avoid differentiating noisy nearest-waypoint heading jumps, waypoint
       headings are unwrapped along cumulative arc length s, differentiated via
       second-order central differences, and smoothed with a zero-phase Gaussian
       spatial filter over arc length before computing d(kappa_ref)/ds and
       evaluating continuous linear segment projection s_proj.
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
from controllers.augmented_lqr_controller import AugmentedLQRController
from vehicle import Vehicle


# ==================================================
# 1. AUGMENTED LQR CARLA CONFIGURATION
# ==================================================

CURVATURE_FILTER_HALF_WINDOW: int = 5
CURVATURE_FILTER_SIGMA_PTS: float = 2.0
MAX_CURVATURE_DERIVATIVE_1PM2: float = 0.25

STRAIGHT_APPROACH_MAX_S_M: float = 57.50

DEFAULT_LQR_FILE_PREFIX: str = "augmented_lqr_carla_8ms"
DEFAULT_LQR_CSV_PATH: Path = RESULTS_DIR / f"{DEFAULT_LQR_FILE_PREFIX}_telemetry.csv"


# ==================================================
# 2. DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class ReferenceRouteProfile:
    """
    Precomputed arc-length-parameterized reference route profile with smoothed
    curvature and spatial curvature derivative.
    """

    road_x: NDArray[np.float64]
    road_y: NDArray[np.float64]
    road_yaw: NDArray[np.float64]
    unwrapped_yaw: NDArray[np.float64]
    road_s: NDArray[np.float64]
    raw_curvature_1pm: NDArray[np.float64]
    curvature_1pm: NDArray[np.float64]
    d_curvature_ds_1pm2: NDArray[np.float64]


@dataclass(frozen=True)
class AugmentedLQRTelemetrySample:
    """Single closed-loop Augmented LQR CARLA telemetry sample."""

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
    d_curvature_ds_1pm2: float
    feedforward_steering_rad: float
    feedforward_steering_deg: float
    steering_deviation_error_rad: float
    steering_deviation_error_deg: float
    feedforward_steering_rate_rad_s: float
    feedforward_steering_rate_deg_s: float
    feedback_steering_rate_rad_s: float
    feedback_steering_rate_deg_s: float
    requested_steering_rate_rad_s: float
    requested_steering_rate_deg_s: float
    applied_steering_rate_rad_s: float
    applied_steering_rate_deg_s: float
    applied_steering_rad: float
    applied_steering_deg: float
    normalized_carla_steer: float
    throttle: float
    brake: float


@dataclass(frozen=True)
class AugmentedLQRCarlaTestSummary:
    """Summary metrics and verification results from the Augmented LQR CARLA run."""

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
    lqr_gain_k: tuple[float, float, float]
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
    max_feedforward_steering_deg: float
    max_feedforward_rate_deg_s: float
    max_feedback_rate_deg_s: float
    max_requested_rate_deg_s: float
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
# 3. REFERENCE CURVATURE ESTIMATION & FEEDFORWARD
# ==================================================

def build_reference_route_profile(
    road_x: NDArray[np.float64],
    road_y: NDArray[np.float64],
    road_yaw: NDArray[np.float64],
    filter_half_window: int = CURVATURE_FILTER_HALF_WINDOW,
    filter_sigma_pts: float = CURVATURE_FILTER_SIGMA_PTS,
    max_curvature_derivative_1pm2: float = MAX_CURVATURE_DERIVATIVE_1PM2
) -> ReferenceRouteProfile:
    """
    Build an arc-length-parameterized reference route profile with Gaussian-smoothed
    curvature kappa_ref(s) and spatial derivative d(kappa_ref)/ds.

    Curvature Estimation Method:
        1. Compute cumulative chord arc length s along (road_x, road_y).
        2. Unwrap road_yaw to remove +/-pi branch discontinuities.
        3. Compute raw curvature kappa_raw = d(yaw_unwrapped)/ds via central
           differences (np.gradient).
        4. Apply a normalized symmetric Gaussian spatial filter over arc length
           to attenuate waypoint quantization noise while preserving sustained
           road curvature and ensuring C^1 continuity across road transitions.
        5. Compute d(kappa_ref)/ds via central differences over s and verify
           continuity bounds.
    """
    rx = np.asarray(road_x, dtype=np.float64)
    ry = np.asarray(road_y, dtype=np.float64)
    ryaw = np.asarray(road_yaw, dtype=np.float64)

    if rx.ndim != 1 or ry.ndim != 1 or ryaw.ndim != 1:
        raise ValueError("road_x, road_y, and road_yaw must be 1D arrays.")
    if rx.size < 2 or rx.size != ry.size or rx.size != ryaw.size:
        raise ValueError(
            "road_x, road_y, and road_yaw must have matching lengths >= 2."
        )
    if not (np.all(np.isfinite(rx)) and np.all(np.isfinite(ry)) and np.all(np.isfinite(ryaw))):
        raise ValueError("Reference route arrays must contain only finite values.")

    ds = np.hypot(np.diff(rx), np.diff(ry))
    if np.any(ds <= 0.0):
        raise ValueError("Consecutive waypoints must have strictly positive spacing.")

    road_s = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float64)
    unwrapped_yaw = np.unwrap(ryaw).astype(np.float64)

    raw_curvature = np.asarray(
        np.gradient(unwrapped_yaw, road_s),
        dtype=np.float64
    )

    if filter_half_window > 0 and filter_sigma_pts > 0.0 and rx.size >= 3:
        half_w = min(int(filter_half_window), max(1, rx.size // 2))
        grid = np.arange(-half_w, half_w + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (grid / float(filter_sigma_pts)) ** 2)
        weights /= np.sum(weights)
        padded_kappa = np.pad(raw_curvature, half_w, mode="edge")
        smooth_curvature = np.convolve(
            padded_kappa,
            weights,
            mode="valid"
        ).astype(np.float64)
    else:
        smooth_curvature = raw_curvature.copy()

    d_curvature_ds = np.asarray(
        np.gradient(smooth_curvature, road_s),
        dtype=np.float64
    )

    verify_curvature_continuity(
        road_s=road_s,
        curvature_1pm=smooth_curvature,
        d_curvature_ds_1pm2=d_curvature_ds,
        max_curvature_derivative_1pm2=max_curvature_derivative_1pm2
    )

    return ReferenceRouteProfile(
        road_x=rx,
        road_y=ry,
        road_yaw=ryaw,
        unwrapped_yaw=unwrapped_yaw,
        road_s=road_s,
        raw_curvature_1pm=raw_curvature,
        curvature_1pm=smooth_curvature,
        d_curvature_ds_1pm2=d_curvature_ds
    )


def verify_curvature_continuity(
    road_s: NDArray[np.float64],
    curvature_1pm: NDArray[np.float64],
    d_curvature_ds_1pm2: NDArray[np.float64],
    max_curvature_derivative_1pm2: float = MAX_CURVATURE_DERIVATIVE_1PM2
) -> None:
    """
    Verify that estimated reference curvature and its spatial derivative are
    finite and free of unphysical spikes at segment transitions.
    """
    if not (
        np.all(np.isfinite(road_s))
        and np.all(np.isfinite(curvature_1pm))
        and np.all(np.isfinite(d_curvature_ds_1pm2))
    ):
        raise ValueError("Curvature profile contains non-finite values.")

    max_abs_dk_ds = float(np.max(np.abs(d_curvature_ds_1pm2)))
    if max_abs_dk_ds > max_curvature_derivative_1pm2:
        raise ValueError(
            f"Curvature derivative {max_abs_dk_ds:.4f} m^-2 exceeds continuity "
            f"threshold {max_curvature_derivative_1pm2:.4f} m^-2."
        )


def query_reference_route_at_position(
    point_x: float,
    point_y: float,
    profile: ReferenceRouteProfile,
    min_index: int = 0,
    search_window: int = 40
) -> tuple[int, float, float, float, float, float, float]:
    """
    Project (point_x, point_y) onto the local reference route segment and
    interpolate continuous reference quantities:
        (nearest_index, s_proj_m, ref_x_m, ref_y_m, ref_yaw_rad, kappa_ref_1pm, d_kappa_ds_1pm2).
    """
    if not (math.isfinite(point_x) and math.isfinite(point_y)):
        raise ValueError("Query position coordinates must be finite numbers.")

    n_pts = profile.road_x.size
    start_idx = max(0, min(int(min_index), n_pts - 1))
    end_idx = (
        min(n_pts, start_idx + search_window)
        if start_idx > 0
        else n_pts
    )

    dx = profile.road_x[start_idx:end_idx] - point_x
    dy = profile.road_y[start_idx:end_idx] - point_y
    nearest_index = start_idx + int(np.argmin(dx ** 2 + dy ** 2))

    # Orthogonal projection onto the adjacent segment for smooth arc-length progression
    best_s = float(profile.road_s[nearest_index])
    min_dist_sq = float("inf")

    seg_start = max(0, nearest_index - 1)
    seg_end = min(n_pts - 1, nearest_index + 1)

    for idx in range(seg_start, seg_end):
        x0 = float(profile.road_x[idx])
        y0 = float(profile.road_y[idx])
        x1 = float(profile.road_x[idx + 1])
        y1 = float(profile.road_y[idx + 1])
        seg_dx = x1 - x0
        seg_dy = y1 - y0
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq <= 1e-12:
            continue
        t_proj = float(
            np.clip(
                ((point_x - x0) * seg_dx + (point_y - y0) * seg_dy) / seg_len_sq,
                0.0,
                1.0
            )
        )
        proj_x = x0 + t_proj * seg_dx
        proj_y = y0 + t_proj * seg_dy
        dist_sq = (point_x - proj_x) ** 2 + (point_y - proj_y) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            s0 = float(profile.road_s[idx])
            s1 = float(profile.road_s[idx + 1])
            best_s = s0 + t_proj * (s1 - s0)

    ref_x = float(np.interp(best_s, profile.road_s, profile.road_x))
    ref_y = float(np.interp(best_s, profile.road_s, profile.road_y))
    ref_yaw_unwrapped = float(np.interp(best_s, profile.road_s, profile.unwrapped_yaw))
    ref_yaw = normalize_angle(ref_yaw_unwrapped)
    kappa_ref = float(np.interp(best_s, profile.road_s, profile.curvature_1pm))
    d_kappa_ds = float(np.interp(best_s, profile.road_s, profile.d_curvature_ds_1pm2))

    return nearest_index, best_s, ref_x, ref_y, ref_yaw, kappa_ref, d_kappa_ds


def compute_curvature_feedforward_terms(
    wheelbase_m: float,
    reference_curvature_1pm: float,
    d_curvature_ds_1pm2: float,
    speed_mps: float,
    heading_error_rad: float = 0.0
) -> tuple[float, float]:
    """
    Compute nominal kinematic curvature feedforward steering angle delta_ff (rad)
    and feedforward steering rate d(delta_ff)/dt (rad/s):

        delta_ff = atan(L * kappa_ref)
        d(delta_ff)/dt = [L / (1 + (L * kappa_ref)^2)] * (d(kappa_ref)/ds) * s_dot
        where s_dot = max(0.0, v * cos(e_psi)).
    """
    if not (
        math.isfinite(wheelbase_m)
        and math.isfinite(reference_curvature_1pm)
        and math.isfinite(d_curvature_ds_1pm2)
        and math.isfinite(speed_mps)
        and math.isfinite(heading_error_rad)
    ):
        raise ValueError("All feedforward inputs must be finite numbers.")
    if wheelbase_m <= 0.0:
        raise ValueError("wheelbase_m must be strictly positive.")
    if speed_mps < 0.0:
        raise ValueError("speed_mps must be nonnegative.")

    l_kappa = float(wheelbase_m) * float(reference_curvature_1pm)
    delta_ff_rad = float(math.atan(l_kappa))

    d_delta_ff_d_kappa = float(wheelbase_m) / (1.0 + l_kappa * l_kappa)
    s_dot_mps = max(0.0, float(speed_mps) * math.cos(float(heading_error_rad)))
    delta_ff_rate_rad_s = float(
        d_delta_ff_d_kappa * float(d_curvature_ds_1pm2) * s_dot_mps
    )

    return delta_ff_rad, delta_ff_rate_rad_s


def compute_lqr_tracking_errors(
    actual_x: float,
    actual_y: float,
    actual_yaw: float,
    ref_x: float,
    ref_y: float,
    ref_yaw: float,
    applied_steering_rad: float,
    feedforward_steering_rad: float
) -> tuple[float, float, float]:
    """
    Compute the Augmented LQR deviation state components:
        e_y = -(actual_x - ref_x)*sin(ref_yaw) + (actual_y - ref_y)*cos(ref_yaw)
        e_psi = normalize_angle(actual_yaw - ref_yaw)
        delta_error = applied_steering_rad - feedforward_steering_rad
    """
    if not (
        math.isfinite(actual_x)
        and math.isfinite(actual_y)
        and math.isfinite(actual_yaw)
        and math.isfinite(ref_x)
        and math.isfinite(ref_y)
        and math.isfinite(ref_yaw)
        and math.isfinite(applied_steering_rad)
        and math.isfinite(feedforward_steering_rad)
    ):
        raise ValueError("All state error inputs must be finite numbers.")

    dx = float(actual_x) - float(ref_x)
    dy = float(actual_y) - float(ref_y)
    normal_x = -math.sin(float(ref_yaw))
    normal_y = math.cos(float(ref_yaw))

    e_y = float(dx * normal_x + dy * normal_y)
    e_psi = normalize_angle(float(actual_yaw) - float(ref_yaw))
    delta_error = float(applied_steering_rad) - float(feedforward_steering_rad)

    return e_y, e_psi, delta_error


def apply_lqr_steering_rate_and_angle_constraints(
    requested_steering_rate_rad_s: float,
    previous_applied_steering_rad: float,
    dt: float,
    max_steering_rad: float = MAX_STEERING_RAD,
    max_steering_rate_rad_s: float = MAX_STEERING_RATE_RAD_S
) -> tuple[float, float, float]:
    """
    Enforce steering-rate limits, integrate to update steering angle, and
    enforce steering-angle limits:
        1. u_rate_limited = clip(u_requested, -u_max, +u_max)
        2. delta_next = clip(delta_applied + u_rate_limited * dt, -delta_max, +delta_max)
        3. u_applied = (delta_next - delta_applied) / dt

    Returns:
        Tuple of (next_applied_steering_rad, applied_steering_rate_rad_s, rate_limited_cmd_rad_s).
    """
    if not (
        math.isfinite(requested_steering_rate_rad_s)
        and math.isfinite(previous_applied_steering_rad)
        and math.isfinite(dt)
        and math.isfinite(max_steering_rad)
        and math.isfinite(max_steering_rate_rad_s)
    ):
        raise ValueError("All steering constraint inputs must be finite numbers.")
    if dt <= 0.0:
        raise ValueError("Timestep dt must be strictly positive.")
    if max_steering_rad <= 0.0 or max_steering_rate_rad_s <= 0.0:
        raise ValueError("Steering angle and rate limits must be positive.")

    rate_limited_cmd = float(
        np.clip(
            requested_steering_rate_rad_s,
            -max_steering_rate_rad_s,
            max_steering_rate_rad_s
        )
    )
    unclamped_next_steer = float(previous_applied_steering_rad) + rate_limited_cmd * float(dt)
    next_applied_steer = float(
        np.clip(unclamped_next_steer, -max_steering_rad, max_steering_rad)
    )
    applied_rate = float(
        (next_applied_steer - float(previous_applied_steering_rad)) / float(dt)
    )

    return next_applied_steer, applied_rate, rate_limited_cmd


# ==================================================
# 4. SINGLE-STEP CONTROL & STOPPING CONDITIONS
# ==================================================

def compute_augmented_lqr_control_step(
    adapter: CarlaControllerAdapter,
    controller: AugmentedLQRController,
    speed_controller: ConservativeLongitudinalController,
    profile: ReferenceRouteProfile,
    previous_applied_steering_rad: float,
    min_index: int,
    dt: float,
    step: int,
    sim_time_s: float
) -> AugmentedLQRTelemetrySample:
    """
    Execute one closed-loop Augmented LQR + curvature feedforward + longitudinal
    control step using live vehicle state and physics-derived geometry.
    """
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

    # Project rear axle onto reference route for LQR tracking & curvature feedforward
    (
        nearest_index,
        rear_s_m,
        ref_x_m,
        ref_y_m,
        ref_yaw_rad,
        kappa_ref_1pm,
        d_kappa_ds_1pm2
    ) = query_reference_route_at_position(
        point_x=rear_x,
        point_y=rear_y,
        profile=profile,
        min_index=min_index
    )

    heading_error_rad = normalize_angle(yaw_rad - ref_yaw_rad)

    delta_ff_rad, delta_ff_rate_rad_s = compute_curvature_feedforward_terms(
        wheelbase_m=adapter.wheelbase,
        reference_curvature_1pm=kappa_ref_1pm,
        d_curvature_ds_1pm2=d_kappa_ds_1pm2,
        speed_mps=speed_mps,
        heading_error_rad=heading_error_rad
    )

    e_y_m, e_psi_rad, delta_error_rad = compute_lqr_tracking_errors(
        actual_x=rear_x,
        actual_y=rear_y,
        actual_yaw=yaw_rad,
        ref_x=ref_x_m,
        ref_y=ref_y_m,
        ref_yaw=ref_yaw_rad,
        applied_steering_rad=previous_applied_steering_rad,
        feedforward_steering_rad=delta_ff_rad
    )

    # Feedback steering rate from existing AugmentedLQRController: u_fb = -K * z_error
    u_feedback_rad_s = controller.compute(
        lateral_error=e_y_m,
        heading_error=e_psi_rad,
        steering_angle=delta_error_rad
    )

    # Total requested steering rate: u_total = u_feedback + d(delta_ff)/dt
    u_requested_rad_s = float(u_feedback_rad_s + delta_ff_rate_rad_s)

    # Apply steering-rate limits, integrate steering angle, and apply angle limits
    applied_steering_rad, applied_rate_rad_s, _ = (
        apply_lqr_steering_rate_and_angle_constraints(
            requested_steering_rate_rad_s=u_requested_rad_s,
            previous_applied_steering_rad=previous_applied_steering_rad,
            dt=dt,
            max_steering_rad=MAX_STEERING_RAD,
            max_steering_rate_rad_s=MAX_STEERING_RATE_RAD_S
        )
    )

    # Convert applied steering angle (+left, rad) to normalized CARLA steer (+right)
    normalized_carla_steer = adapter.to_carla_steer(applied_steering_rad)

    # Front-axle cross-track error & route progress (consistent with Stanley & Pure Pursuit)
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

    return AugmentedLQRTelemetrySample(
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
        reference_curvature_1pm=kappa_ref_1pm,
        d_curvature_ds_1pm2=d_kappa_ds_1pm2,
        feedforward_steering_rad=delta_ff_rad,
        feedforward_steering_deg=float(math.degrees(delta_ff_rad)),
        steering_deviation_error_rad=delta_error_rad,
        steering_deviation_error_deg=float(math.degrees(delta_error_rad)),
        feedforward_steering_rate_rad_s=delta_ff_rate_rad_s,
        feedforward_steering_rate_deg_s=float(math.degrees(delta_ff_rate_rad_s)),
        feedback_steering_rate_rad_s=float(u_feedback_rad_s),
        feedback_steering_rate_deg_s=float(math.degrees(u_feedback_rad_s)),
        requested_steering_rate_rad_s=u_requested_rad_s,
        requested_steering_rate_deg_s=float(math.degrees(u_requested_rad_s)),
        applied_steering_rate_rad_s=applied_rate_rad_s,
        applied_steering_rate_deg_s=float(math.degrees(applied_rate_rad_s)),
        applied_steering_rad=applied_steering_rad,
        applied_steering_deg=float(math.degrees(applied_steering_rad)),
        normalized_carla_steer=float(normalized_carla_steer),
        throttle=float(throttle),
        brake=float(brake)
    )


def is_lqr_sample_finite(sample: AugmentedLQRTelemetrySample) -> bool:
    """Return True if all numerical fields in the telemetry sample are finite."""
    for value in asdict(sample).values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def check_augmented_lqr_stopping_conditions(
    sample: AugmentedLQRTelemetrySample,
    road_x: NDArray[np.float64],
    road_y: NDArray[np.float64],
    collision_count: int,
    max_duration_s: float = MAX_SIMULATION_DURATION_S,
    max_cte_m: float = MAX_ALLOWED_CTE_M,
    endpoint_tolerance_m: float = ENDPOINT_TOLERANCE_M
) -> tuple[bool, str, bool]:
    """
    Evaluate all safety and mission stopping conditions for Augmented LQR.

    Returns:
        Tuple of (should_stop, termination_reason, route_completed).
    """
    if not is_lqr_sample_finite(sample):
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
# 5. SEGMENT METRICS, CSV EXPORT & PLOTTING
# ==================================================

def _compute_lqr_subset_metrics(
    segment_name: str,
    subset: Sequence[AugmentedLQRTelemetrySample],
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


def calculate_augmented_lqr_segment_metrics(
    samples: Sequence[AugmentedLQRTelemetrySample],
    route_details: RouteDetails | None = None,
    curvature_threshold_1pm: float = CURVATURE_THRESHOLD_1PM,
    approach_max_s_m: float = STRAIGHT_APPROACH_MAX_S_M
) -> dict[str, SegmentPerformanceMetrics]:
    """
    Separate Augmented LQR performance metrics into:
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
        "straight_approach": _compute_lqr_subset_metrics(
            "Straight Approach", approach_samples, approach_dist
        ),
        "sustained_curve": _compute_lqr_subset_metrics(
            "Sustained Curve", curve_samples, curved_dist
        ),
        "straight_exit": _compute_lqr_subset_metrics(
            "Straight Exit", exit_samples, exit_dist
        ),
        "straight_combined": _compute_lqr_subset_metrics(
            "Straight Combined", straight_all_samples, straight_total_dist
        ),
        "overall": _compute_lqr_subset_metrics(
            "Overall Route", samples, overall_dist
        ),
    }


def save_augmented_lqr_telemetry_to_csv(
    samples: Sequence[AugmentedLQRTelemetrySample],
    csv_path: Path = DEFAULT_LQR_CSV_PATH
) -> Path:
    """Save recorded Augmented LQR telemetry samples to a CSV file."""
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


def generate_augmented_lqr_carla_plots(
    samples: Sequence[AugmentedLQRTelemetrySample],
    profile: ReferenceRouteProfile,
    output_dir: Path = RESULTS_DIR,
    target_speed_mps: float = TARGET_SPEED_MPS,
    file_prefix: str = DEFAULT_LQR_FILE_PREFIX
) -> dict[str, Path]:
    """
    Generate and save the 5 required closed-loop Augmented LQR CARLA plots:
        1. results/augmented_lqr_carla_8ms_trajectory.png
        2. results/augmented_lqr_carla_8ms_error.png
        3. results/augmented_lqr_carla_8ms_steering.png
        4. results/augmented_lqr_carla_8ms_speed.png
        5. results/augmented_lqr_carla_8ms_curvature.png
    """
    if not samples:
        raise ValueError("Cannot generate plots from empty telemetry samples.")

    output_dir.mkdir(parents=True, exist_ok=True)

    time_s = np.array([s.sim_time_s for s in samples], dtype=np.float64)
    progress_m = np.array([s.route_progress_m for s in samples], dtype=np.float64)
    curvature_1pm = np.array([s.reference_curvature_1pm for s in samples], dtype=np.float64)
    rear_x = np.array([s.rear_x_m for s in samples], dtype=np.float64)
    rear_y = np.array([s.rear_y_m for s in samples], dtype=np.float64)
    front_x = np.array([s.front_x_m for s in samples], dtype=np.float64)
    front_y = np.array([s.front_y_m for s in samples], dtype=np.float64)
    rear_err = np.array([s.rear_axle_tracking_error_m for s in samples], dtype=np.float64)
    front_cte = np.array([s.front_cross_track_error_m for s in samples], dtype=np.float64)
    ff_steer_deg = np.array([s.feedforward_steering_deg for s in samples], dtype=np.float64)
    app_steer_deg = np.array([s.applied_steering_deg for s in samples], dtype=np.float64)
    ff_rate_deg_s = np.array([s.feedforward_steering_rate_deg_s for s in samples], dtype=np.float64)
    fb_rate_deg_s = np.array([s.feedback_steering_rate_deg_s for s in samples], dtype=np.float64)
    req_rate_deg_s = np.array([s.requested_steering_rate_deg_s for s in samples], dtype=np.float64)
    app_rate_deg_s = np.array([s.applied_steering_rate_deg_s for s in samples], dtype=np.float64)
    carla_steer = np.array([s.normalized_carla_steer for s in samples], dtype=np.float64)
    speed_mps = np.array([s.speed_mps for s in samples], dtype=np.float64)

    trajectory_path = output_dir / f"{file_prefix}_trajectory.png"
    error_path = output_dir / f"{file_prefix}_error.png"
    steering_path = output_dir / f"{file_prefix}_steering.png"
    speed_path = output_dir / f"{file_prefix}_speed.png"
    curvature_path = output_dir / f"{file_prefix}_curvature.png"

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
        label="Augmented LQR Rear-Axle Trajectory"
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
    ax.set_title("CARLA 0.9.16 Closed-Loop Augmented LQR Trajectory (Tesla Model 3, Town10HD_Opt)")
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
    ax_t.set_title("CARLA 0.9.16 Augmented LQR Tracking Error vs. Time and Route Distance")
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
        ff_steer_deg,
        color="tab:green",
        linestyle="--",
        linewidth=1.8,
        label="Curvature Feedforward Steering delta_ff (deg)"
    )
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
    ax_ang.axhline(MAX_STEERING_DEG, color="red", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_ang.axhline(-MAX_STEERING_DEG, color="red", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_ang.set_title("CARLA 0.9.16 Augmented LQR Steering Angle and Steering Rate vs. Time")
    ax_ang.set_ylabel("Steering Angle (deg)")
    ax_ang.grid(True, linestyle="--", alpha=0.5)
    ax_ang.legend(loc="best")

    ax_rate.plot(
        time_s,
        fb_rate_deg_s,
        color="tab:purple",
        linestyle="--",
        linewidth=1.5,
        label="LQR Feedback Rate u_fb (deg/s)"
    )
    ax_rate.plot(
        time_s,
        ff_rate_deg_s,
        color="tab:green",
        linestyle=":",
        linewidth=1.5,
        label="Feedforward Rate d(delta_ff)/dt (deg/s)"
    )
    ax_rate.plot(
        time_s,
        req_rate_deg_s,
        color="tab:gray",
        linestyle="-.",
        linewidth=1.4,
        label="Total Requested Rate u_total (deg/s)"
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
    ax.set_title("CARLA 0.9.16 Augmented LQR Longitudinal Speed Tracking vs. Time")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(speed_path, dpi=150)
    plt.close(fig)

    # 5. Reference Curvature & Feedforward Profile Plot
    fig, (ax_k, ax_dk) = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True)
    ax_k.plot(
        profile.road_s,
        profile.raw_curvature_1pm,
        color="tab:gray",
        linestyle=":",
        linewidth=1.4,
        label="Raw Discrete Waypoint Curvature (m^-1)"
    )
    ax_k.plot(
        profile.road_s,
        profile.curvature_1pm,
        color="tab:red",
        linewidth=2.0,
        label="Gaussian-Smoothed Reference Curvature kappa_ref(s) (m^-1)"
    )
    ax_k.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_k.set_title("CARLA 0.9.16 Reference Route Curvature and Spatial Derivative vs. Arc Length")
    ax_k.set_ylabel("Curvature (m^-1)")
    ax_k.grid(True, linestyle="--", alpha=0.5)
    ax_k.legend(loc="best")

    ax_dk.plot(
        profile.road_s,
        profile.d_curvature_ds_1pm2,
        color="tab:purple",
        linewidth=1.8,
        label="Spatial Curvature Derivative d(kappa_ref)/ds (m^-2)"
    )
    ax_dk.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_dk.set_xlabel("Route Arc-Length Distance s (m)")
    ax_dk.set_ylabel("d(kappa)/ds (m^-2)")
    ax_dk.grid(True, linestyle="--", alpha=0.5)
    ax_dk.legend(loc="best")
    fig.tight_layout()
    fig.savefig(curvature_path, dpi=150)
    plt.close(fig)

    return {
        "trajectory": trajectory_path,
        "error": error_path,
        "steering": steering_path,
        "speed": speed_path,
        "curvature": curvature_path
    }


def summarize_augmented_lqr_test_run(
    samples: Sequence[AugmentedLQRTelemetrySample],
    termination_reason: str,
    route_completed: bool,
    collision_count: int,
    route_details: RouteDetails,
    adapter: CarlaControllerAdapter,
    controller: AugmentedLQRController,
    csv_path: Path,
    plot_paths: dict[str, Path]
) -> AugmentedLQRCarlaTestSummary:
    """Compute final Augmented LQR tracking, feedforward, constraint, and safety metrics."""
    if not samples:
        raise ValueError("Cannot summarize an empty sample list.")

    speeds = np.array([s.speed_mps for s in samples], dtype=np.float64)
    rear_errs = np.array([s.rear_axle_tracking_error_m for s in samples], dtype=np.float64)
    front_ctes = np.array([s.front_cross_track_error_m for s in samples], dtype=np.float64)
    heading_deg = np.array([s.heading_error_deg for s in samples], dtype=np.float64)
    ff_steer_deg = np.array([s.feedforward_steering_deg for s in samples], dtype=np.float64)
    ff_rate_deg_s = np.array([s.feedforward_steering_rate_deg_s for s in samples], dtype=np.float64)
    fb_rate_deg_s = np.array([s.feedback_steering_rate_deg_s for s in samples], dtype=np.float64)
    req_rate_deg_s = np.array([s.requested_steering_rate_deg_s for s in samples], dtype=np.float64)
    app_steer_rad = np.array([s.applied_steering_rad for s in samples], dtype=np.float64)
    app_steer_deg = np.array([s.applied_steering_deg for s in samples], dtype=np.float64)
    app_rate_rad_s = np.array([s.applied_steering_rate_rad_s for s in samples], dtype=np.float64)
    app_rate_deg_s = np.array([s.applied_steering_rate_deg_s for s in samples], dtype=np.float64)
    carla_steer = np.array([s.normalized_carla_steer for s in samples], dtype=np.float64)

    angle_violations = int(np.sum(np.abs(app_steer_rad) > MAX_STEERING_RAD + 1e-9))
    rate_violations = int(np.sum(np.abs(app_rate_rad_s) > MAX_STEERING_RATE_RAD_S + 1e-9))
    non_finite_count = sum(0 if is_lqr_sample_finite(s) else 1 for s in samples)

    segment_metrics = calculate_augmented_lqr_segment_metrics(
        samples=samples,
        route_details=route_details
    )
    geom = adapter.geometry
    k_flat = np.asarray(controller.K, dtype=np.float64).ravel()

    return AugmentedLQRCarlaTestSummary(
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
        lqr_gain_k=(float(k_flat[0]), float(k_flat[1]), float(k_flat[2])),
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
        max_feedforward_steering_deg=float(np.max(np.abs(ff_steer_deg))),
        max_feedforward_rate_deg_s=float(np.max(np.abs(ff_rate_deg_s))),
        max_feedback_rate_deg_s=float(np.max(np.abs(fb_rate_deg_s))),
        max_requested_rate_deg_s=float(np.max(np.abs(req_rate_deg_s))),
        max_applied_steering_deg=float(np.max(np.abs(app_steer_deg))),
        steering_rms_deg=float(np.sqrt(np.mean(app_steer_deg ** 2))),
        max_steering_rate_deg_s=float(np.max(np.abs(app_rate_deg_s))),
        steering_rate_rms_deg_s=float(np.sqrt(np.mean(app_rate_deg_s ** 2))),
        max_normalized_carla_steer=float(np.max(np.abs(carla_steer))),
        angle_constraint_violations=angle_violations,
        rate_constraint_violations=rate_violations,
        non_finite_values=int(non_finite_count),
        csv_path=csv_path,
        plot_paths=plot_paths,
        segment_metrics=segment_metrics
    )


# ==================================================
# 6. MAIN CLOSED-LOOP AUGMENTED LQR CARLA EXECUTION
# ==================================================

def run_augmented_lqr_carla_test(
    host: str = CARLA_HOST,
    port: int = CARLA_PORT,
    timeout_s: float = CLIENT_TIMEOUT_S,
    target_speed_mps: float = TARGET_SPEED_MPS,
    fixed_delta_seconds: float = FIXED_DELTA_SECONDS,
    route_distance_m: float = ROUTE_DISTANCE_M,
    waypoint_spacing_m: float = WAYPOINT_SPACING_M,
    max_simulation_duration_s: float = MAX_SIMULATION_DURATION_S,
    results_dir: Path = RESULTS_DIR,
    file_prefix: str = DEFAULT_LQR_FILE_PREFIX
) -> AugmentedLQRCarlaTestSummary:
    """
    Connect to CARLA 0.9.16, configure synchronous mode, spawn a Tesla Model 3
    on the validated extended curved route (Spawn Point #61), run the closed-loop
    Augmented LQR lane-keeping controller with curvature feedforward until
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

        # 6. Initialize AugmentedLQRController using live physics-derived wheelbase
        lqr_controller = AugmentedLQRController(
            velocity=target_speed_mps,
            wheelbase=adapter.wheelbase,
            dt=fixed_delta_seconds
        )
        speed_controller = ConservativeLongitudinalController(
            target_speed_mps=target_speed_mps
        )

        k_flat = np.asarray(lqr_controller.K, dtype=np.float64).ravel()
        max_steps = int(round(max_simulation_duration_s / fixed_delta_seconds))
        applied_steering_rad: float = 0.0
        current_min_index: int = 0
        samples: list[AugmentedLQRTelemetrySample] = []
        termination_reason: str = "max_duration_reached"
        route_completed: bool = False

        print("=" * 100)
        print("CARLA 0.9.16 CLOSED-LOOP AUGMENTED LQR LANE-KEEPING TEST")
        print("=" * 100)
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
            f"Augmented LQR Gain K     : [{k_flat[0]:.6f}, {k_flat[1]:.6f}, {k_flat[2]:.6f}] | "
            f"Target Speed = {target_speed_mps:.1f} m/s | dt = {fixed_delta_seconds:.2f} s"
        )
        print(
            f"Actuator Constraints     : Max Steer = +/-{MAX_STEERING_DEG:.0f} deg | "
            f"Max Rate = +/-{MAX_STEERING_RATE_DEG_S:.0f} deg/s"
        )
        print("-" * 100)

        for step in range(max_steps + 1):
            sim_time_s = step * fixed_delta_seconds

            sample = compute_augmented_lqr_control_step(
                adapter=adapter,
                controller=lqr_controller,
                speed_controller=speed_controller,
                profile=profile,
                previous_applied_steering_rad=applied_steering_rad,
                min_index=current_min_index,
                dt=fixed_delta_seconds,
                step=step,
                sim_time_s=sim_time_s
            )
            samples.append(sample)
            applied_steering_rad = sample.applied_steering_rad
            current_min_index = sample.nearest_index

            should_stop, reason, completed = check_augmented_lqr_stopping_conditions(
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
                    f"e_y={sample.rear_axle_tracking_error_m:+.4f}m | "
                    f"f_cte={sample.front_cross_track_error_m:+.4f}m | "
                    f"ff={sample.feedforward_steering_deg:+.2f}deg | "
                    f"steer={sample.applied_steering_deg:+.2f}deg | "
                    f"rate={sample.applied_steering_rate_deg_s:+.2f}deg/s"
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

        csv_path = save_augmented_lqr_telemetry_to_csv(
            samples=samples,
            csv_path=results_dir / f"{file_prefix}_telemetry.csv"
        )
        plot_paths = generate_augmented_lqr_carla_plots(
            samples=samples,
            profile=profile,
            output_dir=results_dir,
            target_speed_mps=target_speed_mps,
            file_prefix=file_prefix
        )

        summary = summarize_augmented_lqr_test_run(
            samples=samples,
            termination_reason=termination_reason,
            route_completed=route_completed,
            collision_count=len(collision_events),
            route_details=route_details,
            adapter=adapter,
            controller=lqr_controller,
            csv_path=csv_path,
            plot_paths=plot_paths
        )

        print("-" * 100)
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
            f"Constraint Violations    : Angle={summary.angle_constraint_violations}, "
            f"Rate={summary.rate_constraint_violations}, NonFinite={summary.non_finite_values}"
        )
        print("-" * 100)
        print(
            f"{'Segment':<20} | {'Samples':>7} | {'Dist (m)':>8} | "
            f"{'Rear RMSE':>10} | {'Max |Rear|':>10} | {'Front RMSE':>10} | "
            f"{'Max |Front|':>11} | {'Steer RMS':>10} | {'Max Rate':>10}"
        )
        print("-" * 100)
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
        print("-" * 100)
        print(f"Telemetry CSV Saved      : {summary.csv_path}")
        print("=" * 100)

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
    """Entry point for running the closed-loop Augmented LQR CARLA test."""
    run_augmented_lqr_carla_test()


if __name__ == "__main__":
    main()
