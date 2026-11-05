"""
Closed-Loop Stanley Lane-Keeping Controller Integration & Live Test in CARLA 0.9.16.

Connects the existing geometric StanleyController and PIDController to a live
Tesla Model 3 in CARLA 0.9.16 (Town10HD_Opt) using CarlaControllerAdapter for
all physics-derived vehicle geometry, coordinate frame transformations, and
steering conversions.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
import sys
from pathlib import Path
from typing import Any, Sequence, cast

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
from controllers.pid_controller import PIDController
from controllers.stanley_controller import StanleyController
from vehicle import Vehicle


# ==================================================
# 1. CONFIGURATION CONSTANTS
# ==================================================

CARLA_HOST: str = "127.0.0.1"
CARLA_PORT: int = 2000
CLIENT_TIMEOUT_S: float = 30.0
EXPECTED_MAP_NAME: str = "Town10HD_Opt"

VEHICLE_BLUEPRINT_ID: str = "vehicle.tesla.model3"
COLLISION_SENSOR_BP_ID: str = "sensor.other.collision"

FIXED_DELTA_SECONDS: float = 0.05
TARGET_SPEED_MPS: float = 8.0

STANLEY_GAIN: float = 1.5
STANLEY_EPSILON: float = 1e-3

MAX_STEERING_DEG: float = 30.0
MAX_STEERING_RAD: float = float(np.radians(MAX_STEERING_DEG))

MAX_STEERING_RATE_DEG_S: float = 20.0
MAX_STEERING_RATE_RAD_S: float = float(np.radians(MAX_STEERING_RATE_DEG_S))

WAYPOINT_SPACING_M: float = 0.5
ROUTE_DISTANCE_M: float = 138.0
MIN_EXTENDED_ROUTE_LENGTH_M: float = 120.0
MAX_STEP_YAW_CHANGE_DEG: float = 3.0
MAX_TOTAL_ROUTE_YAW_DEG: float = 100.0
CURVATURE_THRESHOLD_1PM: float = 0.005
PREFERRED_SPAWN_POINT_INDEX: int = 61

ENDPOINT_TOLERANCE_M: float = 1.5
MAX_ALLOWED_CTE_M: float = 2.5
MAX_SIMULATION_DURATION_S: float = 35.0

SETTLE_TICKS: int = 10
STOP_BRAKE_TICKS: int = 20

RESULTS_DIR: Path = Path(__file__).resolve().parents[2] / "results"
DEFAULT_FILE_PREFIX: str = "stanley_carla_extended_8ms"
DEFAULT_CSV_PATH: Path = RESULTS_DIR / f"{DEFAULT_FILE_PREFIX}_telemetry.csv"


# ==================================================
# 2. DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class TelemetrySample:
    """Single closed-loop simulation telemetry sample."""

    step: int
    sim_time_s: float
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
    road_curvature_1pm: float = 0.0
    route_progress_m: float = 0.0
    route_progress_pct: float = 0.0


@dataclass(frozen=True)
class SegmentPerformanceMetrics:
    """Tracking and steering performance metrics for a route segment."""

    segment_name: str
    sample_count: int
    distance_m: float
    mean_speed_mps: float
    max_speed_mps: float
    front_cte_rmse_m: float
    max_front_cte_m: float
    rear_axle_rmse_m: float
    max_rear_axle_error_m: float
    max_heading_error_deg: float
    steering_rms_deg: float
    max_applied_steering_deg: float
    max_steering_rate_deg_s: float
    steering_rate_rms_deg_s: float


@dataclass(frozen=True)
class RouteDetails:
    """Metadata describing the extracted continuous driving-lane route."""

    spawn_point_index: int
    road_id: int
    lane_id: int
    num_waypoints: int
    waypoint_spacing_m: float
    total_length_m: float
    max_step_yaw_change_deg: float
    total_yaw_change_deg: float
    start_x_ctrl: float
    start_y_ctrl: float
    end_x_ctrl: float
    end_y_ctrl: float
    straight_length_m: float = 0.0
    curved_length_m: float = 0.0
    max_abs_curvature_1pm: float = 0.0
    min_turn_radius_m: float = 0.0


@dataclass(frozen=True)
class StanleyCarlaTestSummary:
    """Summary metrics and verification results from the closed-loop run."""

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
    max_speed_mps: float
    mean_speed_mps: float
    final_speed_mps: float
    rear_axle_rmse_m: float
    max_rear_axle_error_m: float
    front_cte_rmse_m: float
    max_front_cte_m: float
    final_front_cte_m: float
    max_heading_error_deg: float
    final_heading_error_deg: float
    max_requested_steering_deg: float
    max_applied_steering_deg: float
    max_steering_rate_deg_s: float
    max_normalized_carla_steer: float
    angle_constraint_violations: int
    rate_constraint_violations: int
    non_finite_values: int
    csv_path: Path
    plot_paths: dict[str, Path]
    steering_rms_deg: float = 0.0
    segment_metrics: dict[str, SegmentPerformanceMetrics] | None = None


# ==================================================
# 3. STEERING CONSTRAINTS & LONGITUDINAL CONTROL
# ==================================================

def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to (-180.0, 180.0]."""
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if wrapped <= -180.0:
        return 180.0
    return float(wrapped)


def apply_steering_constraints(
    requested_steering_rad: float,
    previous_applied_steering_rad: float,
    dt: float,
    max_steering_rad: float = MAX_STEERING_RAD,
    max_steering_rate_rad_s: float = MAX_STEERING_RATE_RAD_S
) -> tuple[float, float, float]:
    """
    Enforce hard steering-angle (+/-30 deg) and steering-rate (+/-20 deg/s)
    constraints on a requested controller steering angle in radians.

    Returns:
        Tuple of (applied_steering_rad, angle_limited_steering_rad, steering_rate_rad_s).
    """
    if not (
        math.isfinite(requested_steering_rad)
        and math.isfinite(previous_applied_steering_rad)
        and math.isfinite(dt)
        and math.isfinite(max_steering_rad)
        and math.isfinite(max_steering_rate_rad_s)
    ):
        raise ValueError("All steering constraint inputs must be finite numbers.")

    if dt <= 0.0:
        raise ValueError("Timestep dt must be strictly positive.")
    if max_steering_rad <= 0.0 or max_steering_rate_rad_s <= 0.0:
        raise ValueError(
            "max_steering_rad and max_steering_rate_rad_s must be positive."
        )

    # 1. Angle saturation
    angle_limited_rad = float(
        np.clip(requested_steering_rad, -max_steering_rad, max_steering_rad)
    )

    # 2. Rate saturation relative to previous applied steering
    max_delta_rad = max_steering_rate_rad_s * dt
    rate_limited_rad = float(
        np.clip(
            angle_limited_rad,
            previous_applied_steering_rad - max_delta_rad,
            previous_applied_steering_rad + max_delta_rad
        )
    )

    # 3. Final safety angle clamp
    applied_rad = float(
        np.clip(rate_limited_rad, -max_steering_rad, max_steering_rad)
    )

    steering_rate_rad_s = float(
        (applied_rad - previous_applied_steering_rad) / dt
    )

    return applied_rad, angle_limited_rad, steering_rate_rad_s


class ConservativeLongitudinalController:
    """
    Conservative PID-based longitudinal speed controller targeting 5.0 m/s.
    Reuses the project's PIDController with integral anti-windup and mutually
    exclusive throttle/brake commands.
    """

    def __init__(
        self,
        target_speed_mps: float = TARGET_SPEED_MPS,
        kp: float = 0.55,
        ki: float = 0.20,
        kd: float = 0.05,
        max_throttle: float = 0.60,
        max_brake: float = 0.50,
        integral_limit: float = 1.5,
        brake_deadband_mps: float = 0.05
    ) -> None:
        if not math.isfinite(target_speed_mps) or target_speed_mps <= 0.0:
            raise ValueError("target_speed_mps must be a finite positive number.")
        if not (0.0 < max_throttle <= 1.0 and 0.0 < max_brake <= 1.0):
            raise ValueError("max_throttle and max_brake must be in (0.0, 1.0].")

        self.target_speed_mps: float = float(target_speed_mps)
        self.max_throttle: float = float(max_throttle)
        self.max_brake: float = float(max_brake)
        self.integral_limit: float = float(integral_limit)
        self.brake_deadband_mps: float = float(brake_deadband_mps)
        self.pid: PIDController = PIDController(kp=kp, ki=ki, kd=kd)

    def compute(
        self,
        current_speed_mps: float,
        dt: float
    ) -> tuple[float, float]:
        """
        Compute conservative (throttle, brake) commands in [0.0, 1.0].
        """
        if not (math.isfinite(current_speed_mps) and math.isfinite(dt)):
            raise ValueError("current_speed_mps and dt must be finite numbers.")
        if dt <= 0.0:
            raise ValueError("dt must be strictly positive.")

        speed_error = self.target_speed_mps - float(current_speed_mps)

        # Anti-windup: pre-clamp integral inside PIDController
        raw_output = self.pid.compute(error=speed_error, dt=dt)
        self.pid.integral = float(
            np.clip(
                self.pid.integral,
                -self.integral_limit,
                self.integral_limit
            )
        )

        if raw_output >= 0.0 and speed_error >= -self.brake_deadband_mps:
            throttle = float(np.clip(raw_output, 0.0, self.max_throttle))
            brake = 0.0
        elif speed_error < -self.brake_deadband_mps:
            throttle = 0.0
            brake = float(np.clip(-raw_output, 0.0, self.max_brake))
        else:
            throttle = 0.0
            brake = 0.0

        return throttle, brake


# ==================================================
# 4. ROUTE EXTRACTION & CONTINUITY VERIFICATION
# ==================================================

def compute_route_arc_length_and_curvature(
    road_x: NDArray[np.float64],
    road_y: NDArray[np.float64],
    road_yaw: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute cumulative route arc-length distance s (m) and signed discrete
    road curvature kappa (m^-1) in controller coordinates.
    """
    rx = np.asarray(road_x, dtype=np.float64)
    ry = np.asarray(road_y, dtype=np.float64)
    ryaw = np.asarray(road_yaw, dtype=np.float64)

    if rx.size < 2:
        return np.zeros_like(rx), np.zeros_like(rx)

    ds = np.hypot(np.diff(rx), np.diff(ry))
    road_s = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float64)

    dyaw = np.arctan2(np.sin(np.diff(ryaw)), np.cos(np.diff(ryaw)))
    road_curvature = np.zeros_like(rx, dtype=np.float64)
    road_curvature[:-1] = dyaw / np.maximum(ds, 1e-6)
    road_curvature[-1] = road_curvature[-2]

    return road_s, road_curvature


def verify_route_continuity(
    waypoints: Sequence[Any],
    expected_spacing_m: float = WAYPOINT_SPACING_M,
    max_step_yaw_deg: float = MAX_STEP_YAW_CHANGE_DEG,
    allow_contiguous_road_id: bool = False
) -> tuple[float, float, float]:
    """
    Verify that a sequence of CARLA waypoints forms a continuous, single-lane,
    non-junction driving route without sharp discontinuities.

    Returns:
        Tuple of (total_length_m, max_step_yaw_change_deg, total_yaw_change_deg).
    """
    if len(waypoints) < 2:
        raise ValueError(
            f"Route must contain at least 2 waypoints, got {len(waypoints)}."
        )

    first_wp = waypoints[0]
    initial_road_id = getattr(first_wp, "road_id", None)
    initial_lane_id = getattr(first_wp, "lane_id", None)
    initial_yaw_deg = float(first_wp.transform.rotation.yaw)

    if bool(getattr(first_wp, "is_junction", False)):
        raise ValueError("Route starts inside a junction/intersection.")

    total_length_m = 0.0
    max_observed_step_yaw_deg = 0.0
    max_total_yaw_deg = 0.0

    min_step_m = 0.05 * expected_spacing_m
    max_step_m = 2.50 * expected_spacing_m

    for idx in range(1, len(waypoints)):
        prev_wp = waypoints[idx - 1]
        curr_wp = waypoints[idx]

        if bool(getattr(curr_wp, "is_junction", False)):
            raise ValueError(
                f"Waypoint {idx} enters a junction/intersection."
            )

        curr_road_id = getattr(curr_wp, "road_id", initial_road_id)
        curr_lane_id = getattr(curr_wp, "lane_id", initial_lane_id)

        road_mismatch = (
            not allow_contiguous_road_id
            and initial_road_id is not None
            and curr_road_id != initial_road_id
        )
        lane_mismatch = (
            initial_lane_id is not None
            and curr_lane_id != initial_lane_id
        )

        if road_mismatch or lane_mismatch:
            raise ValueError(
                f"Discontinuous road/lane change at waypoint {idx}: "
                f"(road={initial_road_id}, lane={initial_lane_id}) -> "
                f"(road={curr_road_id}, lane={curr_lane_id})."
            )

        prev_loc = prev_wp.transform.location
        curr_loc = curr_wp.transform.location

        dx = float(curr_loc.x) - float(prev_loc.x)
        dy = float(curr_loc.y) - float(prev_loc.y)
        step_dist = math.hypot(dx, dy)

        if not math.isfinite(step_dist) or not (min_step_m <= step_dist <= max_step_m):
            raise ValueError(
                f"Invalid waypoint spacing {step_dist:.4f} m at index {idx} "
                f"(expected ~{expected_spacing_m:.2f} m)."
            )

        prev_yaw = float(prev_wp.transform.rotation.yaw)
        curr_yaw = float(curr_wp.transform.rotation.yaw)
        step_yaw_diff = abs(wrap_angle_deg(curr_yaw - prev_yaw))

        if step_yaw_diff > max_step_yaw_deg:
            raise ValueError(
                f"Sharp heading change ({step_yaw_diff:.2f} deg > "
                f"{max_step_yaw_deg:.2f} deg) at waypoint {idx}."
            )

        total_length_m += step_dist
        max_observed_step_yaw_deg = max(max_observed_step_yaw_deg, step_yaw_diff)
        max_total_yaw_deg = max(
            max_total_yaw_deg,
            abs(wrap_angle_deg(curr_yaw - initial_yaw_deg))
        )

    return (
        float(total_length_m),
        float(max_observed_step_yaw_deg),
        float(max_total_yaw_deg)
    )


def extract_continuous_driving_route(
    start_waypoint: Any,
    distance_m: float = ROUTE_DISTANCE_M,
    spacing_m: float = WAYPOINT_SPACING_M,
    max_step_yaw_deg: float = MAX_STEP_YAW_CHANGE_DEG,
    allow_contiguous_road_id: bool = False
) -> list[Any]:
    """
    Extract a continuous single-lane driving route forward from start_waypoint
    without lane changes or junctions.
    """
    if distance_m <= 0.0 or spacing_m <= 0.0:
        raise ValueError("distance_m and spacing_m must be positive.")

    waypoints: list[Any] = [start_waypoint]
    current_wp = start_waypoint
    num_steps = int(round(distance_m / spacing_m))

    target_road_id = getattr(start_waypoint, "road_id", None)
    target_lane_id = getattr(start_waypoint, "lane_id", None)

    for _ in range(num_steps):
        next_candidates = current_wp.next(spacing_m)
        if not next_candidates:
            break

        if allow_contiguous_road_id and len(next_candidates) != 1:
            break

        chosen_wp = None
        for candidate in next_candidates:
            if bool(getattr(candidate, "is_junction", False)):
                continue
            cand_road = getattr(candidate, "road_id", target_road_id)
            cand_lane = getattr(candidate, "lane_id", target_lane_id)
            road_ok = allow_contiguous_road_id or (cand_road == target_road_id)
            if road_ok and cand_lane == target_lane_id:
                chosen_wp = candidate
                break

        if chosen_wp is None:
            break

        waypoints.append(chosen_wp)
        current_wp = chosen_wp

    verify_route_continuity(
        waypoints=waypoints,
        expected_spacing_m=spacing_m,
        max_step_yaw_deg=max_step_yaw_deg,
        allow_contiguous_road_id=allow_contiguous_road_id
    )
    return waypoints


def select_spawn_point_and_route(
    world: Any,
    vehicle_bp: Any,
    route_distance_m: float = ROUTE_DISTANCE_M,
    spacing_m: float = WAYPOINT_SPACING_M,
    max_step_yaw_deg: float = MAX_STEP_YAW_CHANGE_DEG,
    max_total_yaw_deg: float = MAX_TOTAL_ROUTE_YAW_DEG,
    preferred_spawn_point_index: int = PREFERRED_SPAWN_POINT_INDEX
) -> tuple[Any, list[Any], RouteDetails]:
    """
    Search map spawn points for a clean, continuous single-lane driving route
    free of intersections and sharp discontinuities, and spawn the Tesla Model 3.
    """
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in the active CARLA map.")

    is_extended_route = route_distance_m >= MIN_EXTENDED_ROUTE_LENGTH_M
    min_required_length_m = (
        MIN_EXTENDED_ROUTE_LENGTH_M
        if is_extended_route
        else 0.95 * route_distance_m
    )

    candidates: list[
        tuple[int, float, float, int, Any, list[Any], float, float, float]
    ] = []
    driving_lane_type = (
        carla.LaneType.Driving
        if carla is not None and hasattr(carla, "LaneType")
        else None
    )

    for sp_idx, sp in enumerate(spawn_points):
        if driving_lane_type is not None:
            start_wp = carla_map.get_waypoint(
                sp.location,
                project_to_road=True,
                lane_type=driving_lane_type
            )
        else:
            start_wp = carla_map.get_waypoint(
                sp.location,
                project_to_road=True
            )
        if start_wp is None or bool(getattr(start_wp, "is_junction", False)):
            continue

        try:
            route_wps = extract_continuous_driving_route(
                start_waypoint=start_wp,
                distance_m=route_distance_m,
                spacing_m=spacing_m,
                max_step_yaw_deg=max_step_yaw_deg,
                allow_contiguous_road_id=is_extended_route
            )
        except ValueError:
            continue

        total_len, max_step_yaw, total_yaw = verify_route_continuity(
            waypoints=route_wps,
            expected_spacing_m=spacing_m,
            max_step_yaw_deg=max_step_yaw_deg,
            allow_contiguous_road_id=is_extended_route
        )
        if total_len < min_required_length_m or total_yaw > max_total_yaw_deg:
            continue
        if is_extended_route and len(spawn_points) > 1 and total_yaw < 45.0:
            continue

        sp_offset = math.hypot(
            float(sp.location.x) - float(start_wp.transform.location.x),
            float(sp.location.y) - float(start_wp.transform.location.y)
        )
        pref_rank = 0 if sp_idx == preferred_spawn_point_index else 1
        candidates.append(
            (
                pref_rank,
                sp_offset,
                total_yaw,
                sp_idx,
                sp,
                route_wps,
                total_len,
                max_step_yaw,
                total_yaw
            )
        )

    candidates.sort(
        key=lambda item: (item[0], round(item[1], 4), round(item[2], 3), item[3])
    )

    for _, _, _, sp_idx, sp, route_wps, total_len, max_step_yaw, total_yaw in candidates:
        spawned_actor = world.try_spawn_actor(vehicle_bp, sp)
        if spawned_actor is not None:
            vehicle = cast(Any, spawned_actor)
            road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(
                route_wps
            )
            _, road_curvature = compute_route_arc_length_and_curvature(
                road_x, road_y, road_yaw
            )
            ds = np.hypot(np.diff(road_x), np.diff(road_y))
            curved_mask = np.abs(road_curvature[:-1]) >= CURVATURE_THRESHOLD_1PM
            curved_len = float(np.sum(ds[curved_mask]))
            straight_len = float(np.sum(ds[~curved_mask]))
            max_k = float(np.max(np.abs(road_curvature)))
            min_r = float(1.0 / max_k) if max_k > 1e-6 else float("inf")

            first_wp = route_wps[0]
            details = RouteDetails(
                spawn_point_index=sp_idx,
                road_id=int(getattr(first_wp, "road_id", 0)),
                lane_id=int(getattr(first_wp, "lane_id", 0)),
                num_waypoints=len(route_wps),
                waypoint_spacing_m=float(spacing_m),
                total_length_m=float(total_len),
                max_step_yaw_change_deg=float(max_step_yaw),
                total_yaw_change_deg=float(total_yaw),
                start_x_ctrl=float(road_x[0]),
                start_y_ctrl=float(road_y[0]),
                end_x_ctrl=float(road_x[-1]),
                end_y_ctrl=float(road_y[-1]),
                straight_length_m=straight_len,
                curved_length_m=curved_len,
                max_abs_curvature_1pm=max_k,
                min_turn_radius_m=min_r
            )
            return vehicle, route_wps, details

    raise RuntimeError(
        "Unable to find and spawn Tesla Model 3 on a continuous driving-lane route."
    )


# ==================================================
# 5. SINGLE-STEP CONTROL & STOPPING CRITERIA
# ==================================================

def compute_stanley_control_step(
    adapter: CarlaControllerAdapter,
    controller: StanleyController,
    speed_controller: ConservativeLongitudinalController,
    road_x: NDArray[np.float64],
    road_y: NDArray[np.float64],
    road_yaw: NDArray[np.float64],
    previous_applied_steering_rad: float,
    dt: float,
    step: int,
    sim_time_s: float,
    road_s: NDArray[np.float64] | None = None,
    road_curvature: NDArray[np.float64] | None = None
) -> TelemetrySample:
    """
    Execute one closed-loop Stanley + longitudinal control calculation using
    live vehicle state and physics-derived geometry from CarlaControllerAdapter.
    """
    rear_x, rear_y, yaw_rad = adapter.get_rear_axle_pose()
    speed_mps = adapter.get_speed()

    if road_s is None or road_curvature is None:
        road_s, road_curvature = compute_route_arc_length_and_curvature(
            road_x=road_x,
            road_y=road_y,
            road_yaw=road_yaw
        )

    # Construct kinematic Vehicle using the live physics-derived wheelbase (3.0046 m)
    kinematic_vehicle = Vehicle(
        x=rear_x,
        y=rear_y,
        yaw=yaw_rad,
        wheelbase=adapter.wheelbase
    )

    stanley_details = controller.compute_with_details(
        vehicle=kinematic_vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=speed_mps,
        road_yaw=road_yaw
    )

    requested_steering_rad = float(stanley_details["steering"])
    front_cte_m = float(stanley_details["cross_track_error"])
    heading_error_rad = float(stanley_details["heading_error"])
    front_x_m = float(stanley_details["front_x"])
    front_y_m = float(stanley_details["front_y"])
    ref_x_m = float(stanley_details["ref_x"])
    ref_y_m = float(stanley_details["ref_y"])
    ref_yaw_rad = float(stanley_details["ref_yaw"])
    nearest_index = int(stanley_details["nearest_index"])

    # Compute signed rear-axle lateral tracking error relative to nearest rear waypoint
    _, rear_ref_x, rear_ref_y, rear_ref_yaw = controller.find_nearest_waypoint(
        front_x=rear_x,
        front_y=rear_y,
        road_x=road_x,
        road_y=road_y,
        road_yaw=road_yaw
    )
    rear_tracking_error_m = controller.compute_cross_track_error(
        front_x=rear_x,
        front_y=rear_y,
        ref_x=rear_ref_x,
        ref_y=rear_ref_y,
        ref_yaw=rear_ref_yaw
    )

    # Apply +/-30 deg angle and +/-20 deg/s rate constraints
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

    # Compute longitudinal throttle and brake
    throttle, brake = speed_controller.compute(
        current_speed_mps=speed_mps,
        dt=dt
    )

    total_route_len_m = max(float(road_s[-1]), 1e-6)
    route_progress_m = float(road_s[nearest_index])
    route_progress_pct = float(
        np.clip(100.0 * route_progress_m / total_route_len_m, 0.0, 100.0)
    )
    road_curvature_1pm = float(road_curvature[nearest_index])

    return TelemetrySample(
        step=int(step),
        sim_time_s=float(sim_time_s),
        speed_mps=float(speed_mps),
        rear_x_m=float(rear_x),
        rear_y_m=float(rear_y),
        front_x_m=float(front_x_m),
        front_y_m=float(front_y_m),
        yaw_rad=float(yaw_rad),
        ref_x_m=float(ref_x_m),
        ref_y_m=float(ref_y_m),
        ref_yaw_rad=float(ref_yaw_rad),
        nearest_index=int(nearest_index),
        rear_axle_tracking_error_m=float(rear_tracking_error_m),
        front_cross_track_error_m=float(front_cte_m),
        heading_error_rad=float(heading_error_rad),
        heading_error_deg=float(math.degrees(heading_error_rad)),
        requested_steering_rad=float(requested_steering_rad),
        requested_steering_deg=float(math.degrees(requested_steering_rad)),
        angle_limited_steering_rad=float(angle_limited_rad),
        applied_steering_rad=float(applied_steering_rad),
        applied_steering_deg=float(math.degrees(applied_steering_rad)),
        steering_rate_rad_s=float(steering_rate_rad_s),
        steering_rate_deg_s=float(math.degrees(steering_rate_rad_s)),
        normalized_carla_steer=float(normalized_carla_steer),
        throttle=float(throttle),
        brake=float(brake),
        road_curvature_1pm=road_curvature_1pm,
        route_progress_m=route_progress_m,
        route_progress_pct=route_progress_pct
    )


def is_sample_finite(sample: TelemetrySample) -> bool:
    """Return True if all numerical fields in the telemetry sample are finite."""
    for value in asdict(sample).values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def check_stopping_conditions(
    sample: TelemetrySample,
    road_x: NDArray[np.float64],
    road_y: NDArray[np.float64],
    collision_count: int,
    max_duration_s: float = MAX_SIMULATION_DURATION_S,
    max_cte_m: float = MAX_ALLOWED_CTE_M,
    endpoint_tolerance_m: float = ENDPOINT_TOLERANCE_M
) -> tuple[bool, str, bool]:
    """
    Evaluate all safety and mission stopping conditions.

    Returns:
        Tuple of (should_stop, termination_reason, route_completed).
    """
    if not is_sample_finite(sample):
        return True, "invalid_numerical_values", False

    if collision_count > 0:
        return True, "collision_detected", False

    if (
        abs(sample.front_cross_track_error_m) > max_cte_m
        or abs(sample.rear_axle_tracking_error_m) > max_cte_m
    ):
        return True, "left_intended_route", False

    dist_to_endpoint = math.hypot(
        sample.front_x_m - float(road_x[-1]),
        sample.front_y_m - float(road_y[-1])
    )
    if (
        dist_to_endpoint <= endpoint_tolerance_m
        or sample.nearest_index >= len(road_x) - 2
    ):
        return True, "route_endpoint_reached", True

    if sample.sim_time_s >= max_duration_s - 1e-9:
        return True, "max_duration_reached", False

    return False, "running", False


# ==================================================
# 6. SPECTATOR CAMERA & ROUTE VISUALIZATION
# ==================================================

def update_spectator_camera(world: Any, vehicle: Any) -> None:
    """
    Position the spectator camera 8 m behind and 4.5 m above the vehicle,
    aimed along the vehicle's forward heading.
    """
    transform = vehicle.get_transform()
    location = transform.location
    forward = transform.get_forward_vector()

    cam_x = float(location.x) - 8.0 * float(forward.x)
    cam_y = float(location.y) - 8.0 * float(forward.y)
    cam_z = float(location.z) + 4.5

    dx = float(location.x) - cam_x
    dy = float(location.y) - cam_y
    dz = float(location.z) - cam_z

    cam_yaw = math.degrees(math.atan2(dy, dx))
    cam_pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))

    spectator = world.get_spectator()
    if carla is not None:
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=cam_x, y=cam_y, z=cam_z),
                carla.Rotation(pitch=cam_pitch, yaw=cam_yaw, roll=0.0)
            )
        )


def draw_route_markers(world: Any, waypoints: Sequence[Any]) -> None:
    """Draw green debug markers along the extracted CARLA route."""
    if carla is None or not hasattr(world, "debug"):
        return

    for wp in waypoints:
        loc = wp.transform.location
        world.debug.draw_point(
            carla.Location(x=float(loc.x), y=float(loc.y), z=float(loc.z) + 0.3),
            size=0.12,
            color=carla.Color(r=0, g=255, b=0),
            life_time=30.0
        )


# ==================================================
# 7. TELEMETRY CSV EXPORT, METRICS & PLOTTING
# ==================================================

def save_telemetry_to_csv(
    samples: Sequence[TelemetrySample],
    csv_path: Path = DEFAULT_CSV_PATH
) -> Path:
    """Save recorded telemetry samples to a CSV file."""
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


def _compute_subset_metrics(
    segment_name: str,
    subset: Sequence[TelemetrySample],
    segment_distance_m: float
) -> SegmentPerformanceMetrics:
    """Compute tracking and steering metrics over a non-empty sample subset."""
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


def calculate_segment_metrics(
    samples: Sequence[TelemetrySample],
    route_details: RouteDetails | None = None,
    curvature_threshold_1pm: float = CURVATURE_THRESHOLD_1PM
) -> dict[str, SegmentPerformanceMetrics]:
    """
    Separate performance metrics into:
        - "straight": Straight-road segment (|road_curvature_1pm| < threshold)
        - "curved": Curved-road segment (|road_curvature_1pm| >= threshold)
        - "overall": Overall route (all samples)
    """
    if not samples:
        raise ValueError("Cannot calculate segment metrics from empty samples.")

    straight_samples = [
        s for s in samples if abs(s.road_curvature_1pm) < curvature_threshold_1pm
    ]
    curved_samples = [
        s for s in samples if abs(s.road_curvature_1pm) >= curvature_threshold_1pm
    ]

    if route_details is not None and route_details.total_length_m > 0.0:
        overall_dist = route_details.total_length_m
        straight_dist = route_details.straight_length_m
        curved_dist = route_details.curved_length_m
    else:
        progress = np.array([s.route_progress_m for s in samples], dtype=np.float64)
        ds = np.maximum(np.diff(progress, prepend=progress[0]), 0.0)
        kappas = np.array([s.road_curvature_1pm for s in samples], dtype=np.float64)
        curved_mask = np.abs(kappas) >= curvature_threshold_1pm
        overall_dist = float(np.sum(ds))
        straight_dist = float(np.sum(ds[~curved_mask]))
        curved_dist = float(np.sum(ds[curved_mask]))

    return {
        "straight": _compute_subset_metrics(
            "Straight-Road Segment", straight_samples, straight_dist
        ),
        "curved": _compute_subset_metrics(
            "Curved-Road Segment", curved_samples, curved_dist
        ),
        "overall": _compute_subset_metrics(
            "Overall Route", samples, overall_dist
        ),
    }


def generate_stanley_carla_plots(
    samples: Sequence[TelemetrySample],
    road_x: NDArray[np.float64],
    road_y: NDArray[np.float64],
    output_dir: Path = RESULTS_DIR,
    target_speed_mps: float = TARGET_SPEED_MPS,
    file_prefix: str = DEFAULT_FILE_PREFIX
) -> dict[str, Path]:
    """
    Generate and save the closed-loop Stanley CARLA evaluation plots:
        1. Vehicle trajectory plot
        2. Cross-track error plot (vs. time and vs. route distance)
        3. Dedicated tracking error & curvature vs. route distance plot
        4. Steering angle plot
        5. Speed vs time plot
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
    heading_deg = np.array([s.heading_error_deg for s in samples], dtype=np.float64)
    req_steer_deg = np.array([s.requested_steering_deg for s in samples], dtype=np.float64)
    app_steer_deg = np.array([s.applied_steering_deg for s in samples], dtype=np.float64)
    carla_steer = np.array([s.normalized_carla_steer for s in samples], dtype=np.float64)
    speed_mps = np.array([s.speed_mps for s in samples], dtype=np.float64)

    trajectory_path = output_dir / f"{file_prefix}_trajectory.png"
    cte_path = output_dir / f"{file_prefix}_cross_track_error.png"
    error_dist_path = output_dir / f"{file_prefix}_error_vs_distance.png"
    steering_path = output_dir / f"{file_prefix}_steering_angle.png"
    speed_path = output_dir / f"{file_prefix}_speed.png"

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
        label="Rear-Axle Trajectory"
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
    ax.set_title("CARLA 0.9.16 Closed-Loop Stanley Trajectory (Tesla Model 3, Town10HD_Opt)")
    ax.set_xlabel("Controller X (m)")
    ax.set_ylabel("Controller Y (m, +Left)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(trajectory_path, dpi=150)
    plt.close(fig)

    # 2. Cross-Track Error Plot (vs. Time and vs. Route Distance)
    fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(10, 7.5))
    ax_t.plot(
        time_s,
        front_cte,
        color="tab:blue",
        linewidth=2.0,
        label="Front-Axle Cross-Track Error (m)"
    )
    ax_t.plot(
        time_s,
        rear_err,
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label="Rear-Axle Tracking Error (m)"
    )
    ax_t.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_t.set_title("CARLA 0.9.16 Stanley Path-Tracking Error vs. Time and Route Distance")
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
        front_cte,
        color="tab:blue",
        linewidth=2.0,
        label="Front-Axle Cross-Track Error (m)"
    )
    ax_s.plot(
        progress_m,
        rear_err,
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label="Rear-Axle Tracking Error (m)"
    )
    ax_s.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_s.set_xlabel("Route Distance / Progress s (m)")
    ax_s.set_ylabel("Tracking Error (m)")
    ax_s.grid(True, linestyle="--", alpha=0.5)
    ax_s.legend(loc="best")
    fig.tight_layout()
    fig.savefig(cte_path, dpi=150)
    plt.close(fig)

    # 3. Dedicated Tracking Error, Heading/Steering, and Road Curvature vs. Route Distance Plot
    fig, (ax_err, ax_ang, ax_kap) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    if has_curved:
        for ax_sub in (ax_err, ax_ang, ax_kap):
            ax_sub.axvspan(
                curve_start_m,
                curve_end_m,
                color="gold",
                alpha=0.22,
                label="Curved-Road Segment" if ax_sub is ax_err else None
            )
    ax_err.plot(
        progress_m,
        front_cte,
        color="tab:blue",
        linewidth=2.0,
        label="Front-Axle Cross-Track Error (m)"
    )
    ax_err.plot(
        progress_m,
        rear_err,
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label="Rear-Axle Tracking Error (m)"
    )
    ax_err.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_err.set_title("CARLA 0.9.16 Stanley Tracking Performance vs. Route Distance (v = 8.0 m/s)")
    ax_err.set_ylabel("Tracking Error (m)")
    ax_err.grid(True, linestyle="--", alpha=0.5)
    ax_err.legend(loc="best")

    ax_ang.plot(
        progress_m,
        heading_deg,
        color="tab:purple",
        linewidth=1.8,
        label="Heading Error (deg)"
    )
    ax_ang.plot(
        progress_m,
        app_steer_deg,
        color="tab:blue",
        linestyle="--",
        linewidth=2.0,
        label="Applied Steering Angle (deg, +Left)"
    )
    ax_ang.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_ang.set_ylabel("Angle (deg)")
    ax_ang.grid(True, linestyle="--", alpha=0.5)
    ax_ang.legend(loc="best")

    ax_kap.plot(
        progress_m,
        curvature_1pm,
        color="tab:red",
        linewidth=2.0,
        label="Road Curvature kappa(s) (m^-1)"
    )
    ax_kap.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax_kap.set_xlabel("Route Distance s (m)")
    ax_kap.set_ylabel("Curvature (m^-1)")
    ax_kap.grid(True, linestyle="--", alpha=0.5)
    ax_kap.legend(loc="best")
    fig.tight_layout()
    fig.savefig(error_dist_path, dpi=150)
    plt.close(fig)

    # 4. Steering Angle Plot
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
    ax1.set_title("CARLA 0.9.16 Stanley Steering Commands vs. Time")
    ax1.set_xlabel("Simulation Time (s)")
    ax1.set_ylabel("Steering Angle (deg)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best")
    fig.tight_layout()
    fig.savefig(steering_path, dpi=150)
    plt.close(fig)

    # 5. Speed vs. Time Plot
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
    ax.set_title("CARLA 0.9.16 Longitudinal Speed Tracking vs. Time")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(speed_path, dpi=150)
    plt.close(fig)

    return {
        "trajectory": trajectory_path,
        "cross_track_error": cte_path,
        "steering_angle": steering_path,
        "speed": speed_path
    }


def summarize_test_run(
    samples: Sequence[TelemetrySample],
    termination_reason: str,
    route_completed: bool,
    collision_count: int,
    route_details: RouteDetails,
    adapter: CarlaControllerAdapter,
    csv_path: Path,
    plot_paths: dict[str, Path]
) -> StanleyCarlaTestSummary:
    """Compute final tracking, steering constraint, and safety metrics."""
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

    angle_violations = int(np.sum(np.abs(app_steer_rad) > MAX_STEERING_RAD + 1e-9))
    rate_violations = int(np.sum(np.abs(steer_rate_rad_s) > MAX_STEERING_RATE_RAD_S + 1e-9))
    non_finite_count = sum(0 if is_sample_finite(s) else 1 for s in samples)

    segment_metrics = calculate_segment_metrics(
        samples=samples,
        route_details=route_details
    )
    geom = adapter.geometry

    return StanleyCarlaTestSummary(
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
        max_speed_mps=float(np.max(speeds)),
        mean_speed_mps=float(np.mean(speeds)),
        final_speed_mps=float(speeds[-1]),
        rear_axle_rmse_m=float(np.sqrt(np.mean(rear_errs ** 2))),
        max_rear_axle_error_m=float(np.max(np.abs(rear_errs))),
        front_cte_rmse_m=float(np.sqrt(np.mean(front_ctes ** 2))),
        max_front_cte_m=float(np.max(np.abs(front_ctes))),
        final_front_cte_m=float(abs(front_ctes[-1])),
        max_heading_error_deg=float(np.max(np.abs(heading_deg))),
        final_heading_error_deg=float(abs(heading_deg[-1])),
        max_requested_steering_deg=float(np.max(np.abs(req_steer_deg))),
        max_applied_steering_deg=float(np.max(np.abs(app_steer_deg))),
        max_steering_rate_deg_s=float(np.max(np.abs(steer_rate_deg_s))),
        max_normalized_carla_steer=float(np.max(np.abs(carla_steer))),
        angle_constraint_violations=angle_violations,
        rate_constraint_violations=rate_violations,
        non_finite_values=int(non_finite_count),
        csv_path=csv_path,
        plot_paths=plot_paths,
        steering_rms_deg=float(np.sqrt(np.mean(app_steer_deg ** 2))),
        segment_metrics=segment_metrics
    )


# ==================================================
# 8. MAIN CLOSED-LOOP CARLA EXECUTION
# ==================================================

def run_stanley_carla_test(
    host: str = CARLA_HOST,
    port: int = CARLA_PORT,
    timeout_s: float = CLIENT_TIMEOUT_S,
    target_speed_mps: float = TARGET_SPEED_MPS,
    fixed_delta_seconds: float = FIXED_DELTA_SECONDS,
    route_distance_m: float = ROUTE_DISTANCE_M,
    waypoint_spacing_m: float = WAYPOINT_SPACING_M,
    max_simulation_duration_s: float = MAX_SIMULATION_DURATION_S,
    results_dir: Path = RESULTS_DIR,
    file_prefix: str = DEFAULT_FILE_PREFIX
) -> StanleyCarlaTestSummary:
    """
    Connect to CARLA 0.9.16, configure synchronous mode, spawn a Tesla Model 3
    with a collision sensor, run the closed-loop Stanley lane-keeping controller
    until completion or safety stop, and always restore world settings and
    destroy all spawned actors.
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

        # 2. Select continuous driving-lane route and spawn Tesla Model 3
        vehicle, waypoints, route_details = select_spawn_point_and_route(
            world=world,
            vehicle_bp=vehicle_bp,
            route_distance_m=route_distance_m,
            spacing_m=waypoint_spacing_m
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

        # 5. Retrieve vehicle physics and initialize CarlaControllerAdapter
        adapter = CarlaControllerAdapter(vehicle)
        road_x, road_y, road_yaw = adapter.convert_waypoints(waypoints)
        road_s, road_curvature = compute_route_arc_length_and_curvature(
            road_x=road_x,
            road_y=road_y,
            road_yaw=road_yaw
        )
        draw_route_markers(world, waypoints)

        # 6. Initialize StanleyController using its actual constructor signature
        stanley_controller = StanleyController(
            stanley_gain=STANLEY_GAIN,
            epsilon=STANLEY_EPSILON
        )
        speed_controller = ConservativeLongitudinalController(
            target_speed_mps=target_speed_mps
        )

        max_steps = int(round(max_simulation_duration_s / fixed_delta_seconds))
        applied_steering_rad: float = 0.0
        samples: list[TelemetrySample] = []
        termination_reason: str = "max_duration_reached"
        route_completed: bool = False

        print("=" * 95)
        print("CARLA 0.9.16 CLOSED-LOOP STANLEY LANE-KEEPING TEST")
        print("=" * 95)
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
            f"Control Configuration    : Target Speed = {target_speed_mps:.1f} m/s | "
            f"dt = {fixed_delta_seconds:.2f} s | "
            f"Max Steer = +/-{MAX_STEERING_DEG:.0f} deg | "
            f"Max Rate = +/-{MAX_STEERING_RATE_DEG_S:.0f} deg/s"
        )
        print("-" * 95)

        for step in range(max_steps + 1):
            sim_time_s = step * fixed_delta_seconds

            sample = compute_stanley_control_step(
                adapter=adapter,
                controller=stanley_controller,
                speed_controller=speed_controller,
                road_x=road_x,
                road_y=road_y,
                road_yaw=road_yaw,
                previous_applied_steering_rad=applied_steering_rad,
                dt=fixed_delta_seconds,
                step=step,
                sim_time_s=sim_time_s,
                road_s=road_s,
                road_curvature=road_curvature
            )
            samples.append(sample)
            applied_steering_rad = sample.applied_steering_rad

            should_stop, reason, completed = check_stopping_conditions(
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
                    f"kappa={sample.road_curvature_1pm:+.4f}m^-1 | "
                    f"front_cte={sample.front_cross_track_error_m:+.4f}m | "
                    f"rear_err={sample.rear_axle_tracking_error_m:+.4f}m | "
                    f"head_err={sample.heading_error_deg:+.2f}deg | "
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

        csv_path = save_telemetry_to_csv(
            samples=samples,
            csv_path=results_dir / f"{file_prefix}_telemetry.csv"
        )
        plot_paths = generate_stanley_carla_plots(
            samples=samples,
            road_x=road_x,
            road_y=road_y,
            output_dir=results_dir,
            target_speed_mps=target_speed_mps,
            file_prefix=file_prefix
        )

        summary = summarize_test_run(
            samples=samples,
            termination_reason=termination_reason,
            route_completed=route_completed,
            collision_count=len(collision_events),
            route_details=route_details,
            adapter=adapter,
            csv_path=csv_path,
            plot_paths=plot_paths
        )

        print("-" * 95)
        print(f"Termination Reason       : {summary.termination_reason}")
        print(f"Route Completed          : {summary.route_completed}")
        print(f"Collision Count          : {summary.collision_count}")
        print(f"Simulation Duration      : {summary.simulation_duration_s:.2f} s ({summary.total_steps} steps)")
        print(f"Max / Mean Speed         : {summary.max_speed_mps:.4f} m/s / {summary.mean_speed_mps:.4f} m/s")
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
        if summary.segment_metrics is not None:
            print("-" * 95)
            print(
                f"{'Segment':<24} | {'Samples':>7} | {'Dist (m)':>8} | "
                f"{'Front RMSE':>10} | {'Max |CTE|':>10} | {'Rear RMSE':>10} | "
                f"{'Steer RMS':>10} | {'Max Rate':>10}"
            )
            print("-" * 95)
            for seg_key in ("straight", "curved", "overall"):
                sm = summary.segment_metrics[seg_key]
                print(
                    f"{sm.segment_name:<24} | {sm.sample_count:7d} | {sm.distance_m:8.2f} | "
                    f"{sm.front_cte_rmse_m:10.4f} | {sm.max_front_cte_m:10.4f} | "
                    f"{sm.rear_axle_rmse_m:10.4f} | {sm.steering_rms_deg:10.4f} | "
                    f"{sm.max_steering_rate_deg_s:10.4f}"
                )
        print("-" * 95)
        print(f"Telemetry CSV Saved      : {summary.csv_path}")
        print("=" * 95)

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
    """Entry point for running the closed-loop Stanley CARLA test."""
    run_stanley_carla_test()


if __name__ == "__main__":
    main()
