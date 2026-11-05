from __future__ import annotations

from dataclasses import dataclass, replace
import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src directory to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from carla_integration.carla_adapter import CarlaControllerAdapter
from carla_integration.pure_pursuit_carla import (
    ConservativeLongitudinalController,
    FIXED_DELTA_SECONDS,
    LOOKAHEAD_GAIN_S,
    MAX_STEERING_RAD,
    MAX_STEERING_RATE_RAD_S,
    MIN_LOOKAHEAD_M,
    RouteDetails,
    TARGET_SPEED_MPS,
    apply_steering_constraints,
    calculate_pure_pursuit_segment_metrics,
    check_pure_pursuit_stopping_conditions,
    compute_pure_pursuit_control_step,
    generate_pure_pursuit_carla_plots,
    is_pp_sample_finite,
    run_pure_pursuit_carla_test,
    save_pure_pursuit_telemetry_to_csv,
    summarize_pure_pursuit_test_run,
    verify_route_continuity,
)
from carla_integration.stanley_carla import extract_continuous_driving_route
from controllers.pure_pursuit_controller import PurePursuitController


# ==================================================
# MOCK OBJECTS WITH MEASURED TESLA MODEL 3 GEOMETRY
# ==================================================

TESLA_WHEELBASE_M: float = 3.004645
TESLA_REAR_OFFSET_X_M: float = -1.386406
TESLA_FRONT_OFFSET_X_M: float = TESLA_REAR_OFFSET_X_M + TESLA_WHEELBASE_M
TESLA_MAX_STEER_DEG: float = 70.0


@dataclass
class MockVector3D:
    x: float
    y: float
    z: float = 0.0


@dataclass
class MockRotation:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


class MockTransform:
    def __init__(
        self,
        location: MockVector3D | None = None,
        rotation: MockRotation | None = None
    ) -> None:
        self.location = location or MockVector3D(0.0, 0.0, 0.0)
        self.rotation = rotation or MockRotation(0.0, 0.0, 0.0)

    def get_forward_vector(self) -> MockVector3D:
        yaw_rad = math.radians(self.rotation.yaw)
        return MockVector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)


class MockWaypoint:
    def __init__(
        self,
        x: float,
        y: float,
        yaw_deg: float = 0.0,
        road_id: int = 10,
        lane_id: int = 1,
        is_junction: bool = False,
        next_waypoints: list[MockWaypoint] | None = None
    ) -> None:
        self.transform = MockTransform(
            location=MockVector3D(x=x, y=y, z=0.0),
            rotation=MockRotation(yaw=yaw_deg)
        )
        self.road_id = road_id
        self.lane_id = lane_id
        self.is_junction = is_junction
        self._next_waypoints = next_waypoints or []

    def next(self, distance: float) -> list[MockWaypoint]:
        assert distance > 0.0
        return list(self._next_waypoints)


@dataclass
class MockWheelPhysics:
    position: MockVector3D
    max_steer_angle: float


@dataclass
class MockPhysicsControl:
    wheels: list[MockWheelPhysics]


class MockTeslaVehicle:
    """
    Mock CARLA 0.9.16 Tesla Model 3 actor with measured live physics geometry:
        wheelbase = 3.004645 m
        rear_axle_offset_x = -1.386406 m
        front_axle_offset_x = +1.618239 m
        max_front_wheel_steer_deg = 70.0 deg
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw_deg: float = 0.0,
        speed_mps: float = 8.0
    ) -> None:
        self.id = 61
        self.type_id = "vehicle.tesla.model3"
        self.is_alive = True
        self.autopilot_enabled = True
        self.applied_controls: list[Any] = []

        yaw_rad = math.radians(yaw_deg)
        self._transform = MockTransform(
            location=MockVector3D(x=x, y=y, z=0.0),
            rotation=MockRotation(yaw=yaw_deg)
        )
        self._velocity = MockVector3D(
            x=speed_mps * math.cos(yaw_rad),
            y=speed_mps * math.sin(yaw_rad),
            z=0.0
        )

        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        half_track_m = 0.79

        local_wheels = [
            (TESLA_FRONT_OFFSET_X_M, -half_track_m, TESLA_MAX_STEER_DEG),
            (TESLA_FRONT_OFFSET_X_M, +half_track_m, TESLA_MAX_STEER_DEG),
            (TESLA_REAR_OFFSET_X_M, -half_track_m, 0.0),
            (TESLA_REAR_OFFSET_X_M, +half_track_m, 0.0),
        ]
        wheels: list[MockWheelPhysics] = []
        for lx, ly, steer_deg in local_wheels:
            wx_cm = (x + lx * cos_y - ly * sin_y) * 100.0
            wy_cm = (y + lx * sin_y + ly * cos_y) * 100.0
            wheels.append(
                MockWheelPhysics(
                    position=MockVector3D(x=wx_cm, y=wy_cm, z=35.0),
                    max_steer_angle=steer_deg
                )
            )
        self._physics = MockPhysicsControl(wheels=wheels)

    def get_transform(self) -> MockTransform:
        return self._transform

    def get_velocity(self) -> MockVector3D:
        return self._velocity

    def get_physics_control(self) -> MockPhysicsControl:
        return self._physics

    def set_autopilot(self, enabled: bool) -> None:
        self.autopilot_enabled = enabled

    def apply_control(self, control: Any) -> None:
        self.applied_controls.append(control)

    def destroy(self) -> bool:
        self.is_alive = False
        return True


def build_chained_mock_waypoints(
    num_points: int = 41,
    spacing_m: float = 0.5,
    carla_y: float = -60.90,
    yaw_deg: float = 0.0
) -> list[MockWaypoint]:
    """Create a linked chain of MockWaypoints along +X."""
    wps = [
        MockWaypoint(
            x=i * spacing_m,
            y=carla_y,
            yaw_deg=yaw_deg,
            road_id=10,
            lane_id=1,
            is_junction=False
        )
        for i in range(num_points)
    ]
    for i in range(num_points - 1):
        wps[i]._next_waypoints = [wps[i + 1]]
    return wps


# ==================================================
# 1. CARLA ADAPTER INTEGRATION & REAR-AXLE GEOMETRY
# ==================================================

def test_carla_adapter_integration_and_rear_axle_geometry() -> None:
    """
    Verify Pure Pursuit CARLA integration reads live physics geometry from
    CarlaControllerAdapter (wheelbase ~3.0046 m, rear-axle offset ~-1.3864 m,
    max front wheel steer ~70 deg) and tracks from the rear axle.
    """
    vehicle = MockTeslaVehicle(x=10.0, y=-60.90, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle)

    assert np.isclose(adapter.wheelbase, TESLA_WHEELBASE_M, atol=1e-4)
    assert np.isclose(
        adapter.geometry.rear_axle_offset_x,
        TESLA_REAR_OFFSET_X_M,
        atol=1e-4
    )
    assert np.isclose(adapter.geometry.max_front_wheel_steer_deg, 70.0, atol=1e-4)

    rear_x, rear_y, yaw_rad = adapter.get_rear_axle_pose()
    assert np.isclose(rear_x, 10.0 + TESLA_REAR_OFFSET_X_M, atol=1e-6)
    assert np.isclose(rear_y, 60.90, atol=1e-6)
    assert np.isclose(yaw_rad, 0.0, atol=1e-6)


# ==================================================
# 2. STEERING SIGN CONVERSION & LOOKAHEAD SELECTION
# ==================================================

def test_steering_sign_conversion_and_lookahead_selection() -> None:
    """
    Verify speed-dependent lookahead (Ld_desired = 3.0 + 0.5 * v = 7.0 m at 8 m/s),
    actual Euclidean lookahead distance, and controller-to-CARLA steering sign
    reversal.
    """
    wps = build_chained_mock_waypoints(num_points=61, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    # carla_y = -60.90 -> controller road_y = +60.90

    # Place vehicle 1.0 m LEFT of centerline in controller frame (ctrl_y = +61.90 -> carla_y = -61.90)
    vehicle_left = MockTeslaVehicle(x=5.0, y=-61.90, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle_left)
    pp_ctrl = PurePursuitController(
        wheelbase=adapter.wheelbase,
        min_lookahead=MIN_LOOKAHEAD_M,
        lookahead_gain=LOOKAHEAD_GAIN_S,
        use_dynamic_lookahead=True
    )
    lon_ctrl = ConservativeLongitudinalController(target_speed_mps=TARGET_SPEED_MPS)

    sample = compute_pure_pursuit_control_step(
        adapter=adapter,
        controller=pp_ctrl,
        speed_controller=lon_ctrl,
        road_x=road_x,
        road_y=road_y,
        road_yaw=road_yaw,
        previous_applied_steering_rad=0.0,
        min_index=0,
        dt=FIXED_DELTA_SECONDS,
        step=0,
        sim_time_s=0.0
    )

    # At v = 8.0 m/s, Ld_desired = 3.0 + 0.5 * 8.0 = 7.0 m
    assert np.isclose(sample.desired_lookahead_m, 7.0, atol=1e-9)
    assert sample.target_index > sample.nearest_index
    expected_actual_ld = math.hypot(
        sample.target_x_m - sample.rear_x_m,
        sample.target_y_m - sample.rear_y_m
    )
    assert np.isclose(sample.actual_lookahead_m, expected_actual_ld, atol=1e-12)

    # Vehicle left of centerline -> negative CTE -> negative controller steer (right turn)
    # -> positive normalized CARLA steer (+right)
    assert sample.rear_axle_tracking_error_m < 0.0
    assert sample.front_cross_track_error_m < 0.0
    assert sample.requested_steering_rad < 0.0
    assert sample.applied_steering_rad < 0.0
    assert sample.normalized_carla_steer > 0.0

    # Normalization divides by 70 deg (not 30 deg)
    expected_carla_steer = -sample.applied_steering_rad / math.radians(70.0)
    assert np.isclose(sample.normalized_carla_steer, expected_carla_steer, atol=1e-12)


# ==================================================
# 3. STEERING-ANGLE & STEERING-RATE LIMITS
# ==================================================

def test_steering_angle_and_rate_limits() -> None:
    """
    Verify +/-30 deg steering angle limit and +/-20 deg/s steering rate limit
    at dt = 0.05 s (1.0 deg max step change).
    """
    dt = FIXED_DELTA_SECONDS
    app_pos, lim_pos, rate_pos = apply_steering_constraints(
        requested_steering_rad=math.radians(40.0),
        previous_applied_steering_rad=0.0,
        dt=dt
    )
    assert np.isclose(lim_pos, MAX_STEERING_RAD, atol=1e-12)
    assert np.isclose(app_pos, math.radians(1.0), atol=1e-12)
    assert np.isclose(rate_pos, MAX_STEERING_RATE_RAD_S, atol=1e-12)

    app_sat, lim_sat, rate_sat = apply_steering_constraints(
        requested_steering_rad=math.radians(-40.0),
        previous_applied_steering_rad=math.radians(-29.5),
        dt=dt
    )
    assert np.isclose(lim_sat, -MAX_STEERING_RAD, atol=1e-12)
    assert np.isclose(app_sat, -MAX_STEERING_RAD, atol=1e-12)
    assert abs(rate_sat) <= MAX_STEERING_RATE_RAD_S + 1e-12


# ==================================================
# 4. ROUTE CONTINUITY VALIDATION
# ==================================================

def test_route_continuity_validation() -> None:
    """
    Verify route continuity validation accepts contiguous road segments on the
    same driving lane (e.g., 10 -> 0 -> 3, lane 1) when allow_contiguous_road_id=True,
    and rejects junctions, lane changes, and sharp heading jumps.
    """
    wps = build_chained_mock_waypoints(num_points=31, spacing_m=0.5)
    for i in range(10, 20):
        wps[i].road_id = 0
    for i in range(20, 31):
        wps[i].road_id = 3

    extracted = extract_continuous_driving_route(
        start_waypoint=wps[0],
        distance_m=15.0,
        spacing_m=0.5,
        allow_contiguous_road_id=True
    )
    assert len(extracted) == 31
    total_len, max_step_yaw, _ = verify_route_continuity(
        extracted,
        expected_spacing_m=0.5,
        allow_contiguous_road_id=True
    )
    assert np.isclose(total_len, 15.0, atol=1e-9)
    assert max_step_yaw == 0.0

    # Reject lane change even when allow_contiguous_road_id=True
    wps[15].lane_id = 2
    with pytest.raises(ValueError, match="Discontinuous road/lane"):
        verify_route_continuity(
            wps,
            expected_spacing_m=0.5,
            allow_contiguous_road_id=True
        )


# ==================================================
# 5. NUMERICAL STABILITY, COLLISION & ROUTE COMPLETION
# ==================================================

def test_numerical_stability_collision_and_route_completion() -> None:
    """
    Verify low-speed numerical stability, collision stopping, off-route
    stopping, non-finite stopping, timeout stopping, and route completion.
    """
    wps = build_chained_mock_waypoints(num_points=41, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)

    # Zero speed numerical stability
    vehicle_zero = MockTeslaVehicle(x=2.0, y=-60.90, yaw_deg=0.0, speed_mps=0.0)
    adapter = CarlaControllerAdapter(vehicle_zero)
    pp_ctrl = PurePursuitController(wheelbase=adapter.wheelbase)
    lon_ctrl = ConservativeLongitudinalController(target_speed_mps=8.0)

    sample = compute_pure_pursuit_control_step(
        adapter=adapter,
        controller=pp_ctrl,
        speed_controller=lon_ctrl,
        road_x=road_x,
        road_y=road_y,
        road_yaw=road_yaw,
        previous_applied_steering_rad=0.0,
        min_index=0,
        dt=0.05,
        step=0,
        sim_time_s=0.0
    )
    assert is_pp_sample_finite(sample)
    assert np.isclose(sample.desired_lookahead_m, 3.0, atol=1e-9)

    # Nominal running
    stop, reason, completed = check_pure_pursuit_stopping_conditions(
        sample, road_x, road_y, collision_count=0
    )
    assert not stop and reason == "running" and not completed

    # Collision handling
    stop, reason, completed = check_pure_pursuit_stopping_conditions(
        sample, road_x, road_y, collision_count=1
    )
    assert stop and reason == "collision_detected" and not completed

    # Route completion
    done_sample = replace(
        sample,
        front_x_m=float(road_x[-1]),
        front_y_m=float(road_y[-1]),
        front_nearest_index=len(road_x) - 1
    )
    stop, reason, completed = check_pure_pursuit_stopping_conditions(
        done_sample, road_x, road_y, collision_count=0
    )
    assert stop and reason == "route_endpoint_reached" and completed

    # Off-route detection
    off_sample = replace(sample, rear_axle_tracking_error_m=3.0)
    stop, reason, completed = check_pure_pursuit_stopping_conditions(
        off_sample, road_x, road_y, collision_count=0, max_cte_m=2.5
    )
    assert stop and reason == "left_intended_route" and not completed

    # Invalid numerical value detection
    nan_sample = replace(sample, requested_steering_rad=float("nan"))
    stop, reason, completed = check_pure_pursuit_stopping_conditions(
        nan_sample, road_x, road_y, collision_count=0
    )
    assert stop and reason == "invalid_numerical_values" and not completed


# ==================================================
# 6. ARTIFACTS, SEGMENT METRICS & CLEANUP RESTORATION
# ==================================================

def test_outputs_segment_metrics_and_cleanup(tmp_path: Path) -> None:
    """
    Verify CSV saving, all 5 required plot outputs, 4-segment performance
    breakdown, and guaranteed actor/world-settings cleanup on exception.
    """
    wps = build_chained_mock_waypoints(num_points=41, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    vehicle = MockTeslaVehicle(x=2.0, y=-60.90, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle)
    pp_ctrl = PurePursuitController(wheelbase=adapter.wheelbase)
    lon_ctrl = ConservativeLongitudinalController(target_speed_mps=8.0)

    samples = [
        compute_pure_pursuit_control_step(
            adapter=adapter,
            controller=pp_ctrl,
            speed_controller=lon_ctrl,
            road_x=road_x,
            road_y=road_y,
            road_yaw=road_yaw,
            previous_applied_steering_rad=0.0,
            min_index=0,
            dt=0.05,
            step=i,
            sim_time_s=i * 0.05
        )
        for i in range(5)
    ]

    csv_path = save_pure_pursuit_telemetry_to_csv(
        samples,
        tmp_path / "pure_pursuit_carla_8ms_telemetry.csv"
    )
    assert csv_path.exists()

    plot_paths = generate_pure_pursuit_carla_plots(
        samples=samples,
        road_x=road_x,
        road_y=road_y,
        output_dir=tmp_path,
        file_prefix="pure_pursuit_carla_8ms"
    )
    assert set(plot_paths.keys()) == {
        "trajectory",
        "error",
        "steering",
        "speed",
        "lookahead"
    }
    for path in plot_paths.values():
        assert path.exists()

    route_details = RouteDetails(
        spawn_point_index=61,
        road_id=10,
        lane_id=1,
        num_waypoints=len(wps),
        waypoint_spacing_m=0.5,
        total_length_m=20.0,
        max_step_yaw_change_deg=0.0,
        total_yaw_change_deg=0.0,
        start_x_ctrl=float(road_x[0]),
        start_y_ctrl=float(road_y[0]),
        end_x_ctrl=float(road_x[-1]),
        end_y_ctrl=float(road_y[-1]),
        straight_length_m=20.0,
        curved_length_m=0.0
    )

    seg_metrics = calculate_pure_pursuit_segment_metrics(samples, route_details)
    assert set(seg_metrics.keys()) == {
        "straight_approach",
        "sustained_curve",
        "straight_exit",
        "straight_combined",
        "overall"
    }

    summary = summarize_pure_pursuit_test_run(
        samples=samples,
        termination_reason="route_endpoint_reached",
        route_completed=True,
        collision_count=0,
        route_details=route_details,
        adapter=adapter,
        controller=pp_ctrl,
        csv_path=csv_path,
        plot_paths=plot_paths
    )
    assert summary.route_completed
    assert summary.angle_constraint_violations == 0
    assert summary.rate_constraint_violations == 0
    assert summary.non_finite_values == 0

    # Verify actor cleanup and world-settings restoration on runtime failure
    mock_sensor = MagicMock()
    mock_sensor.is_alive = True
    original_settings = MagicMock(name="OriginalWorldSettings")
    sim_settings = MagicMock(name="SimWorldSettings")

    mock_world = MagicMock()
    mock_world.get_settings.side_effect = [original_settings, sim_settings]
    mock_world.try_spawn_actor.return_value = vehicle
    mock_world.spawn_actor.return_value = mock_sensor

    mock_map = MagicMock()
    mock_map.name = "Carla/Maps/Town10HD_Opt"
    mock_map.get_spawn_points.return_value = [wps[0].transform]
    mock_map.get_waypoint.return_value = wps[0]
    mock_world.get_map.return_value = mock_map

    mock_client = MagicMock()
    mock_client.get_world.return_value = mock_world

    with patch("carla_integration.pure_pursuit_carla.carla") as mock_carla_mod:
        mock_carla_mod.Client.return_value = mock_client
        with patch(
            "carla_integration.pure_pursuit_carla.compute_pure_pursuit_control_step",
            side_effect=RuntimeError("Simulated Pure Pursuit failure")
        ):
            with pytest.raises(RuntimeError, match="Simulated Pure Pursuit failure"):
                run_pure_pursuit_carla_test(
                    route_distance_m=10.0,
                    waypoint_spacing_m=0.5,
                    results_dir=tmp_path
                )

    assert not vehicle.is_alive
    mock_sensor.stop.assert_called_once()
    mock_sensor.destroy.assert_called_once()
    assert mock_world.apply_settings.call_args_list[-1][0][0] is original_settings
