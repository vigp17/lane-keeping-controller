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

from carla_integration.augmented_lqr_carla import build_reference_route_profile
from carla_integration.baseline_carla import (
    BASELINE_KD,
    BASELINE_KI,
    BASELINE_KP,
    BASELINE_KPSI,
    ConservativeLongitudinalController,
    FIXED_DELTA_SECONDS,
    MAX_STEERING_RAD,
    MAX_STEERING_RATE_RAD_S,
    RouteDetails,
    TARGET_SPEED_MPS,
    apply_steering_constraints,
    calculate_baseline_segment_metrics,
    check_baseline_stopping_conditions,
    compute_baseline_control_step,
    compute_baseline_steering_components,
    generate_baseline_carla_plots,
    is_baseline_sample_finite,
    run_baseline_carla_test,
    save_baseline_telemetry_to_csv,
    summarize_baseline_test_run,
    verify_route_continuity,
)
from carla_integration.carla_adapter import CarlaControllerAdapter
from carla_integration.stanley_carla import extract_continuous_driving_route
from controllers.pid_controller import PIDController


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

def test_carla_adapter_and_rear_axle_geometry() -> None:
    """
    Verify Baseline CARLA integration reads live physics geometry from
    CarlaControllerAdapter (wheelbase ~3.0046 m, rear-axle offset ~-1.3864 m,
    max front wheel steer ~70 deg) rather than hardcoded 2.8 m values.
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
    assert not np.isclose(adapter.wheelbase, 2.8, atol=1e-2)

    rear_x, rear_y, yaw_rad = adapter.get_rear_axle_pose()
    assert np.isclose(rear_x, 10.0 + TESLA_REAR_OFFSET_X_M, atol=1e-6)
    assert np.isclose(rear_y, 60.90, atol=1e-6)
    assert np.isclose(yaw_rad, 0.0, atol=1e-6)


# ==================================================
# 2. LATERAL FEEDBACK, HEADING FEEDBACK & CURVATURE FEEDFORWARD
# ==================================================

def test_lateral_heading_feedback_signs_and_curvature_feedforward() -> None:
    """
    Verify:
        1. Positive lateral error (e_y > 0, vehicle left of centerline) produces
           negative lateral feedback steering (-Kp * e_y).
        2. Positive heading error (e_psi > 0, vehicle pointing left of path)
           produces negative heading feedback steering (-Kpsi * e_psi).
        3. Curvature feedforward equals atan(wheelbase * kappa_ref) and sums
           with feedback steering: delta_command = delta_ff + delta_feedback.
    """
    lateral_pid = PIDController(kp=BASELINE_KP, ki=BASELINE_KI, kd=BASELINE_KD)

    # 1. Pure lateral offset (e_y = +2.0 m, e_psi = 0, kappa = 0)
    req_1, ff_1, lat_fb_1, head_fb_1, fb_1 = compute_baseline_steering_components(
        lateral_pid=lateral_pid,
        e_y_m=2.0,
        e_psi_rad=0.0,
        wheelbase_m=TESLA_WHEELBASE_M,
        reference_curvature_1pm=0.0,
        dt=FIXED_DELTA_SECONDS,
        kpsi=BASELINE_KPSI
    )
    assert np.isclose(ff_1, 0.0, atol=1e-12)
    assert np.isclose(lat_fb_1, -0.05 * 2.0, atol=1e-12)
    assert lat_fb_1 < 0.0
    assert np.isclose(head_fb_1, 0.0, atol=1e-12)
    assert np.isclose(req_1, fb_1, atol=1e-12)

    # 2. Pure heading error (e_y = 0, e_psi = +0.1 rad, kappa = 0)
    req_2, ff_2, lat_fb_2, head_fb_2, fb_2 = compute_baseline_steering_components(
        lateral_pid=lateral_pid,
        e_y_m=0.0,
        e_psi_rad=0.1,
        wheelbase_m=TESLA_WHEELBASE_M,
        reference_curvature_1pm=0.0,
        dt=FIXED_DELTA_SECONDS,
        kpsi=BASELINE_KPSI
    )
    assert np.isclose(ff_2, 0.0, atol=1e-12)
    assert np.isclose(lat_fb_2, 0.0, atol=1e-12)
    assert np.isclose(head_fb_2, -0.8 * 0.1, atol=1e-12)
    assert head_fb_2 < 0.0
    assert np.isclose(req_2, fb_2, atol=1e-12)

    # 3. Combined curvature feedforward + feedback on right-hand curve (kappa = -0.02655 m^-1)
    kappa_ref = -0.02655
    req_3, ff_3, lat_fb_3, head_fb_3, fb_3 = compute_baseline_steering_components(
        lateral_pid=lateral_pid,
        e_y_m=0.5,
        e_psi_rad=-0.02,
        wheelbase_m=TESLA_WHEELBASE_M,
        reference_curvature_1pm=kappa_ref,
        dt=FIXED_DELTA_SECONDS,
        kpsi=BASELINE_KPSI
    )
    expected_ff = math.atan(TESLA_WHEELBASE_M * kappa_ref)
    assert np.isclose(ff_3, expected_ff, atol=1e-12)
    assert np.isclose(fb_3, lat_fb_3 + head_fb_3, atol=1e-12)
    assert np.isclose(req_3, ff_3 + fb_3, atol=1e-12)


# ==================================================
# 3. STEERING-ANGLE & STEERING-RATE CONSTRAINTS
# ==================================================

def test_steering_angle_and_rate_constraints() -> None:
    """
    Verify +/-30 deg steering angle limit and +/-20 deg/s steering rate limit
    at dt = 0.05 s (maximum 1.0 deg change per timestep).
    """
    dt = FIXED_DELTA_SECONDS  # 0.05 s -> max 1.0 deg per step
    app_1, lim_1, rate_1 = apply_steering_constraints(
        requested_steering_rad=math.radians(40.0),
        previous_applied_steering_rad=0.0,
        dt=dt
    )
    assert np.isclose(lim_1, MAX_STEERING_RAD, atol=1e-12)
    assert np.isclose(app_1, math.radians(1.0), atol=1e-12)
    assert np.isclose(rate_1, MAX_STEERING_RATE_RAD_S, atol=1e-12)

    app_2, lim_2, rate_2 = apply_steering_constraints(
        requested_steering_rad=math.radians(-40.0),
        previous_applied_steering_rad=math.radians(-29.5),
        dt=dt
    )
    assert np.isclose(lim_2, -MAX_STEERING_RAD, atol=1e-12)
    assert np.isclose(app_2, -MAX_STEERING_RAD, atol=1e-12)
    assert abs(rate_2) <= MAX_STEERING_RATE_RAD_S + 1e-12


# ==================================================
# 4. ROUTE CONTINUITY, STABILITY, COLLISION & COMPLETION
# ==================================================

def test_route_continuity_stability_collision_and_completion() -> None:
    """
    Verify route continuity validation, numerical stability, corrective
    CARLA steering sign reversal, collision stopping, off-route stopping,
    and route endpoint completion.
    """
    wps = build_chained_mock_waypoints(num_points=41, spacing_m=0.5, carla_y=-60.90)
    extracted = extract_continuous_driving_route(
        start_waypoint=wps[0],
        distance_m=20.0,
        spacing_m=0.5,
        allow_contiguous_road_id=True
    )
    total_len, max_step_yaw, _ = verify_route_continuity(
        extracted,
        expected_spacing_m=0.5,
        allow_contiguous_road_id=True
    )
    assert np.isclose(total_len, 20.0, atol=1e-9)
    assert max_step_yaw == 0.0

    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    profile = build_reference_route_profile(road_x, road_y, road_yaw)

    # Vehicle 1.0 m LEFT of centerline in controller frame (ctrl_y = +61.90 -> carla_y = -61.90)
    vehicle = MockTeslaVehicle(x=4.0, y=-61.90, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle)
    lateral_pid = PIDController(kp=BASELINE_KP, ki=BASELINE_KI, kd=BASELINE_KD)
    lon_ctrl = ConservativeLongitudinalController(target_speed_mps=TARGET_SPEED_MPS)

    sample = compute_baseline_control_step(
        adapter=adapter,
        lateral_pid=lateral_pid,
        speed_controller=lon_ctrl,
        profile=profile,
        previous_applied_steering_rad=0.0,
        min_index=0,
        dt=FIXED_DELTA_SECONDS,
        step=0,
        sim_time_s=0.0
    )
    assert is_baseline_sample_finite(sample)
    # Vehicle left of centerline -> e_y > 0 -> negative controller steering (right turn)
    # -> positive normalized CARLA steer (+right)
    assert sample.rear_axle_tracking_error_m > 0.0
    assert sample.lateral_feedback_steering_rad < 0.0
    assert sample.requested_steering_rad < 0.0
    assert sample.applied_steering_rad < 0.0
    assert sample.normalized_carla_steer > 0.0

    # Normal running
    stop, reason, completed = check_baseline_stopping_conditions(
        sample, road_x, road_y, collision_count=0
    )
    assert not stop and reason == "running" and not completed

    # Collision stopping
    stop, reason, completed = check_baseline_stopping_conditions(
        sample, road_x, road_y, collision_count=1
    )
    assert stop and reason == "collision_detected" and not completed

    # Off-route stopping
    off_sample = replace(sample, rear_axle_tracking_error_m=3.2)
    stop, reason, completed = check_baseline_stopping_conditions(
        off_sample, road_x, road_y, collision_count=0, max_cte_m=2.5
    )
    assert stop and reason == "left_intended_route" and not completed

    # Route endpoint completion
    done_sample = replace(
        sample,
        front_x_m=float(road_x[-1]),
        front_y_m=float(road_y[-1]),
        front_nearest_index=len(road_x) - 1
    )
    stop, reason, completed = check_baseline_stopping_conditions(
        done_sample, road_x, road_y, collision_count=0
    )
    assert stop and reason == "route_endpoint_reached" and completed


# ==================================================
# 5. OUTPUT ARTIFACTS, SEGMENT METRICS & CLEANUP
# ==================================================

def test_outputs_segment_metrics_and_cleanup_restoration(tmp_path: Path) -> None:
    """
    Verify CSV saving, all 5 required plot outputs, segment metric partitioning,
    and guaranteed actor/world-settings cleanup in try/finally on failure.
    """
    wps = build_chained_mock_waypoints(num_points=41, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    profile = build_reference_route_profile(road_x, road_y, road_yaw)

    vehicle = MockTeslaVehicle(x=2.0, y=-60.90, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle)
    lateral_pid = PIDController(kp=BASELINE_KP, ki=BASELINE_KI, kd=BASELINE_KD)
    lon_ctrl = ConservativeLongitudinalController(target_speed_mps=TARGET_SPEED_MPS)

    samples = []
    app_steer = 0.0
    for i in range(5):
        sm = compute_baseline_control_step(
            adapter=adapter,
            lateral_pid=lateral_pid,
            speed_controller=lon_ctrl,
            profile=profile,
            previous_applied_steering_rad=app_steer,
            min_index=0,
            dt=FIXED_DELTA_SECONDS,
            step=i,
            sim_time_s=i * FIXED_DELTA_SECONDS
        )
        samples.append(sm)
        app_steer = sm.applied_steering_rad

    csv_path = save_baseline_telemetry_to_csv(
        samples,
        tmp_path / "baseline_carla_8ms_telemetry.csv"
    )
    assert csv_path.exists()

    plot_paths = generate_baseline_carla_plots(
        samples=samples,
        profile=profile,
        output_dir=tmp_path,
        file_prefix="baseline_carla_8ms"
    )
    assert set(plot_paths.keys()) == {
        "trajectory",
        "error",
        "steering",
        "speed",
        "curvature"
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

    seg_metrics = calculate_baseline_segment_metrics(samples, route_details)
    assert set(seg_metrics.keys()) == {
        "straight_approach",
        "sustained_curve",
        "straight_exit",
        "straight_combined",
        "overall"
    }

    summary = summarize_baseline_test_run(
        samples=samples,
        termination_reason="route_endpoint_reached",
        route_completed=True,
        collision_count=0,
        route_details=route_details,
        adapter=adapter,
        lateral_pid=lateral_pid,
        kpsi=BASELINE_KPSI,
        csv_path=csv_path,
        plot_paths=plot_paths
    )
    assert summary.route_completed
    assert summary.angle_constraint_violations == 0
    assert summary.rate_constraint_violations == 0
    assert summary.non_finite_values == 0

    # Verify actor cleanup and world-settings restoration on runtime exception
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

    with patch("carla_integration.baseline_carla.carla") as mock_carla_mod:
        mock_carla_mod.Client.return_value = mock_client
        with patch(
            "carla_integration.baseline_carla.compute_baseline_control_step",
            side_effect=RuntimeError("Simulated Baseline failure")
        ):
            with pytest.raises(RuntimeError, match="Simulated Baseline failure"):
                run_baseline_carla_test(
                    route_distance_m=10.0,
                    waypoint_spacing_m=0.5,
                    results_dir=tmp_path
                )

    assert not vehicle.is_alive
    mock_sensor.stop.assert_called_once()
    mock_sensor.destroy.assert_called_once()
    assert mock_world.apply_settings.call_args_list[-1][0][0] is original_settings
