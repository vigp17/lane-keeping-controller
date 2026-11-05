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
from carla_integration.stanley_carla import (
    ConservativeLongitudinalController,
    FIXED_DELTA_SECONDS,
    MAX_STEERING_RAD,
    MAX_STEERING_RATE_RAD_S,
    RouteDetails,
    TARGET_SPEED_MPS,
    apply_steering_constraints,
    check_stopping_conditions,
    compute_stanley_control_step,
    extract_continuous_driving_route,
    generate_stanley_carla_plots,
    run_stanley_carla_test,
    save_telemetry_to_csv,
    summarize_test_run,
    verify_route_continuity,
)
from controllers.stanley_controller import StanleyController


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

    def next(self, _distance: float) -> list[MockWaypoint]:
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
        speed_mps: float = 5.0
    ) -> None:
        self.id = 42
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
    num_points: int = 21,
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
# 1. STEERING ANGLE & RATE CONSTRAINT TESTS
# ==================================================

def test_steering_angle_and_rate_constraints() -> None:
    """
    Verify +/-30 deg steering angle limit and +/-20 deg/s steering rate limit
    at dt = 0.05 s (max 1.0 deg change per step).
    """
    dt = FIXED_DELTA_SECONDS  # 0.05 s -> max 1.0 deg per step
    prev_steer = 0.0

    # Request +45 deg (exceeds both 30 deg angle limit and 1 deg/step rate limit)
    applied_1, angle_lim_1, rate_1 = apply_steering_constraints(
        requested_steering_rad=math.radians(45.0),
        previous_applied_steering_rad=prev_steer,
        dt=dt
    )
    assert np.isclose(angle_lim_1, MAX_STEERING_RAD, atol=1e-12)
    assert np.isclose(applied_1, math.radians(1.0), atol=1e-12)
    assert np.isclose(rate_1, MAX_STEERING_RATE_RAD_S, atol=1e-12)

    # Request -45 deg from +29.5 deg
    applied_2, angle_lim_2, rate_2 = apply_steering_constraints(
        requested_steering_rad=math.radians(-45.0),
        previous_applied_steering_rad=math.radians(29.5),
        dt=dt
    )
    assert np.isclose(angle_lim_2, -MAX_STEERING_RAD, atol=1e-12)
    assert np.isclose(applied_2, math.radians(28.5), atol=1e-12)
    assert np.isclose(rate_2, -MAX_STEERING_RATE_RAD_S, atol=1e-12)

    # Invalid inputs raise ValueError
    with pytest.raises(ValueError):
        apply_steering_constraints(float("nan"), 0.0, dt)
    with pytest.raises(ValueError):
        apply_steering_constraints(0.1, 0.0, 0.0)


# ==================================================
# 2. CONSERVATIVE LONGITUDINAL CONTROLLER TESTS
# ==================================================

def test_conservative_longitudinal_controller() -> None:
    """
    Verify longitudinal speed controller commands bounded throttle below
    target speed and bounded brake above target speed without simultaneous actuation.
    """
    lon_ctrl = ConservativeLongitudinalController(target_speed_mps=TARGET_SPEED_MPS)

    # Below target speed
    throttle_low, brake_low = lon_ctrl.compute(current_speed_mps=0.0, dt=0.05)
    assert 0.0 < throttle_low <= lon_ctrl.max_throttle
    assert brake_low == 0.0

    # Above target speed
    throttle_high, brake_high = lon_ctrl.compute(
        current_speed_mps=TARGET_SPEED_MPS + 1.0,
        dt=0.05
    )
    assert throttle_high == 0.0
    assert 0.0 < brake_high <= lon_ctrl.max_brake


# ==================================================
# 3. ROUTE CONTINUITY & EXTRACTION TESTS
# ==================================================

def test_route_continuity_and_extraction() -> None:
    """
    Verify route extraction and continuity validation reject junctions,
    lane changes, and sharp heading discontinuities.
    """
    wps = build_chained_mock_waypoints(num_points=21, spacing_m=0.5)
    extracted = extract_continuous_driving_route(
        start_waypoint=wps[0],
        distance_m=10.0,
        spacing_m=0.5
    )
    assert len(extracted) == 21
    total_len, max_step_yaw, total_yaw = verify_route_continuity(
        extracted,
        expected_spacing_m=0.5
    )
    assert np.isclose(total_len, 10.0, atol=1e-9)
    assert max_step_yaw == 0.0
    assert total_yaw == 0.0

    # Junction waypoint must be rejected
    junction_wps = build_chained_mock_waypoints(num_points=5, spacing_m=0.5)
    junction_wps[2].is_junction = True
    with pytest.raises(ValueError, match="junction"):
        verify_route_continuity(junction_wps, expected_spacing_m=0.5)

    # Lane change must be rejected
    lane_change_wps = build_chained_mock_waypoints(num_points=5, spacing_m=0.5)
    lane_change_wps[3].lane_id = 2
    with pytest.raises(ValueError, match="Discontinuous road/lane"):
        verify_route_continuity(lane_change_wps, expected_spacing_m=0.5)

    # Sharp turn must be rejected
    sharp_turn_wps = build_chained_mock_waypoints(num_points=5, spacing_m=0.5)
    sharp_turn_wps[2].transform.rotation.yaw = 25.0
    with pytest.raises(ValueError, match="Sharp heading change"):
        verify_route_continuity(sharp_turn_wps, expected_spacing_m=0.5, max_step_yaw_deg=3.0)


# ==================================================
# 4. STANLEY CONTROL STEP & GEOMETRY/STEERING TESTS
# ==================================================

def test_stanley_control_step_geometry_and_steering_signs() -> None:
    """
    Verify that compute_stanley_control_step uses the live physics-derived
    3.0046 m wheelbase, -1.3864 m rear-axle offset, and 70 deg max steering,
    and reverses steering sign properly between controller and CARLA.
    """
    wps = build_chained_mock_waypoints(num_points=41, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    # Note: carla_y = -60.90 -> controller road_y = +60.90

    # Place vehicle 1.0 m to the LEFT of the road in controller coordinates:
    # controller_y = +61.90 -> carla_y = -61.90
    vehicle_left = MockTeslaVehicle(x=5.0, y=-61.90, yaw_deg=0.0, speed_mps=5.0)
    adapter = CarlaControllerAdapter(vehicle_left)

    assert np.isclose(adapter.wheelbase, TESLA_WHEELBASE_M, atol=1e-4)
    assert np.isclose(adapter.geometry.rear_axle_offset_x, TESLA_REAR_OFFSET_X_M, atol=1e-4)
    assert np.isclose(adapter.geometry.max_front_wheel_steer_deg, 70.0, atol=1e-4)
    assert not np.isclose(adapter.wheelbase, 2.8, atol=1e-2)

    stanley = StanleyController(stanley_gain=1.5, epsilon=1e-3)
    lon_ctrl = ConservativeLongitudinalController(target_speed_mps=5.0)

    sample = compute_stanley_control_step(
        adapter=adapter,
        controller=stanley,
        speed_controller=lon_ctrl,
        road_x=road_x,
        road_y=road_y,
        road_yaw=road_yaw,
        previous_applied_steering_rad=0.0,
        dt=0.05,
        step=0,
        sim_time_s=0.0
    )

    # Vehicle is left of centerline -> negative CTE -> negative controller steer (right)
    # -> positive CARLA normalized steer (+right)
    assert sample.front_cross_track_error_m < 0.0
    assert sample.rear_axle_tracking_error_m < 0.0
    assert sample.requested_steering_rad < 0.0
    assert sample.applied_steering_rad < 0.0
    assert sample.normalized_carla_steer > 0.0

    # Verify normalization uses 70 deg (not 30 deg)
    expected_carla_steer = -sample.applied_steering_rad / math.radians(70.0)
    assert np.isclose(sample.normalized_carla_steer, expected_carla_steer, atol=1e-12)


# ==================================================
# 5. STOPPING CONDITIONS TESTS
# ==================================================

def test_stopping_conditions() -> None:
    """
    Verify all 5 stopping conditions:
        1. Collision occurs
        2. Route endpoint reached
        3. Vehicle leaves intended route
        4. Numerical values become invalid
        5. Maximum simulation duration reached
    """
    wps = build_chained_mock_waypoints(num_points=41, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    vehicle = MockTeslaVehicle(x=2.0, y=-60.90, yaw_deg=0.0, speed_mps=5.0)
    adapter = CarlaControllerAdapter(vehicle)
    stanley = StanleyController()
    lon_ctrl = ConservativeLongitudinalController()

    nominal_sample = compute_stanley_control_step(
        adapter=adapter,
        controller=stanley,
        speed_controller=lon_ctrl,
        road_x=road_x,
        road_y=road_y,
        road_yaw=road_yaw,
        previous_applied_steering_rad=0.0,
        dt=0.05,
        step=1,
        sim_time_s=0.05
    )

    # Normal running state
    stop, reason, completed = check_stopping_conditions(
        nominal_sample, road_x, road_y, collision_count=0
    )
    assert not stop
    assert reason == "running"
    assert not completed

    # 1. Collision detected
    stop, reason, completed = check_stopping_conditions(
        nominal_sample, road_x, road_y, collision_count=1
    )
    assert stop and reason == "collision_detected" and not completed

    # 2. Route endpoint reached
    endpoint_sample = replace(
        nominal_sample,
        front_x_m=float(road_x[-1]),
        front_y_m=float(road_y[-1]),
        nearest_index=len(road_x) - 1
    )
    stop, reason, completed = check_stopping_conditions(
        endpoint_sample, road_x, road_y, collision_count=0
    )
    assert stop and reason == "route_endpoint_reached" and completed

    # 3. Vehicle leaves intended route
    off_route_sample = replace(nominal_sample, front_cross_track_error_m=3.5)
    stop, reason, completed = check_stopping_conditions(
        off_route_sample, road_x, road_y, collision_count=0, max_cte_m=2.5
    )
    assert stop and reason == "left_intended_route" and not completed

    # 4. Invalid numerical values
    nan_sample = replace(nominal_sample, speed_mps=float("nan"))
    stop, reason, completed = check_stopping_conditions(
        nan_sample, road_x, road_y, collision_count=0
    )
    assert stop and reason == "invalid_numerical_values" and not completed

    # 5. Maximum simulation duration reached
    timeout_sample = replace(nominal_sample, sim_time_s=25.0)
    stop, reason, completed = check_stopping_conditions(
        timeout_sample, road_x, road_y, collision_count=0, max_duration_s=25.0
    )
    assert stop and reason == "max_duration_reached" and not completed


# ==================================================
# 6. CSV EXPORT, PLOTTING & SUMMARY TESTS
# ==================================================

def test_csv_plots_and_summary_generation(tmp_path: Path) -> None:
    """Verify CSV saving, plot creation, and test summary metrics."""
    wps = build_chained_mock_waypoints(num_points=21, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    vehicle = MockTeslaVehicle(x=2.0, y=-60.90, yaw_deg=0.0, speed_mps=5.0)
    adapter = CarlaControllerAdapter(vehicle)
    stanley = StanleyController()
    lon_ctrl = ConservativeLongitudinalController()

    samples = [
        compute_stanley_control_step(
            adapter=adapter,
            controller=stanley,
            speed_controller=lon_ctrl,
            road_x=road_x,
            road_y=road_y,
            road_yaw=road_yaw,
            previous_applied_steering_rad=0.0,
            dt=0.05,
            step=i,
            sim_time_s=i * 0.05
        )
        for i in range(5)
    ]

    csv_path = save_telemetry_to_csv(samples, tmp_path / "telemetry.csv")
    assert csv_path.exists()

    plot_paths = generate_stanley_carla_plots(samples, road_x, road_y, tmp_path)
    assert set(plot_paths.keys()) == {
        "trajectory",
        "cross_track_error",
        "steering_angle",
        "speed"
    }
    for path in plot_paths.values():
        assert path.exists()

    route_details = RouteDetails(
        spawn_point_index=61,
        road_id=10,
        lane_id=1,
        num_waypoints=len(wps),
        waypoint_spacing_m=0.5,
        total_length_m=10.0,
        max_step_yaw_change_deg=0.0,
        total_yaw_change_deg=0.0,
        start_x_ctrl=float(road_x[0]),
        start_y_ctrl=float(road_y[0]),
        end_x_ctrl=float(road_x[-1]),
        end_y_ctrl=float(road_y[-1])
    )
    summary = summarize_test_run(
        samples=samples,
        termination_reason="route_endpoint_reached",
        route_completed=True,
        collision_count=0,
        route_details=route_details,
        adapter=adapter,
        csv_path=csv_path,
        plot_paths=plot_paths
    )
    assert summary.route_completed
    assert summary.angle_constraint_violations == 0
    assert summary.rate_constraint_violations == 0
    assert summary.non_finite_values == 0
    assert np.isclose(summary.wheelbase_m, TESLA_WHEELBASE_M, atol=1e-4)


# ==================================================
# 7. CLEANUP & SETTINGS RESTORATION ON FAILURE TEST
# ==================================================

def test_cleanup_and_settings_restored_on_exception(tmp_path: Path) -> None:
    """
    Verify that spawned vehicle, collision sensor, and synchronous CARLA world
    settings are always restored in try/finally even if an error occurs.
    """
    wps = build_chained_mock_waypoints(num_points=21, spacing_m=0.5, carla_y=-60.90)
    mock_vehicle = MockTeslaVehicle(x=0.0, y=-60.90, yaw_deg=0.0, speed_mps=5.0)

    mock_sensor = MagicMock()
    mock_sensor.is_alive = True

    original_settings = MagicMock(name="OriginalWorldSettings")
    sim_settings = MagicMock(name="SimWorldSettings")

    mock_world = MagicMock()
    mock_world.get_settings.side_effect = [original_settings, sim_settings]
    mock_world.try_spawn_actor.return_value = mock_vehicle
    mock_world.spawn_actor.return_value = mock_sensor

    mock_map = MagicMock()
    mock_map.name = "Carla/Maps/Town10HD_Opt"
    mock_map.get_spawn_points.return_value = [wps[0].transform]
    mock_map.get_waypoint.return_value = wps[0]
    mock_world.get_map.return_value = mock_map

    mock_client = MagicMock()
    mock_client.get_world.return_value = mock_world

    with patch("carla_integration.stanley_carla.carla") as mock_carla_mod:
        mock_carla_mod.Client.return_value = mock_client
        # Force an exception after actors are spawned during control step
        with patch(
            "carla_integration.stanley_carla.compute_stanley_control_step",
            side_effect=RuntimeError("Simulated failure")
        ):
            with pytest.raises(RuntimeError, match="Simulated failure"):
                run_stanley_carla_test(
                    route_distance_m=10.0,
                    waypoint_spacing_m=0.5,
                    results_dir=tmp_path
                )

    assert not mock_vehicle.is_alive
    mock_sensor.stop.assert_called_once()
    mock_sensor.destroy.assert_called_once()
    assert mock_world.apply_settings.call_args_list[-1][0][0] is original_settings
