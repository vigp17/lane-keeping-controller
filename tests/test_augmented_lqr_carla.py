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

from carla_integration.augmented_lqr_carla import (
    ConservativeLongitudinalController,
    FIXED_DELTA_SECONDS,
    MAX_STEERING_RAD,
    MAX_STEERING_RATE_RAD_S,
    RouteDetails,
    TARGET_SPEED_MPS,
    apply_lqr_steering_rate_and_angle_constraints,
    build_reference_route_profile,
    calculate_augmented_lqr_segment_metrics,
    check_augmented_lqr_stopping_conditions,
    compute_augmented_lqr_control_step,
    compute_curvature_feedforward_terms,
    compute_lqr_tracking_errors,
    generate_augmented_lqr_carla_plots,
    is_lqr_sample_finite,
    run_augmented_lqr_carla_test,
    save_augmented_lqr_telemetry_to_csv,
    summarize_augmented_lqr_test_run,
    verify_curvature_continuity,
    verify_route_continuity,
)
from carla_integration.carla_adapter import CarlaControllerAdapter, normalize_angle
from carla_integration.stanley_carla import extract_continuous_driving_route
from controllers.augmented_lqr_controller import AugmentedLQRController


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
    Verify Augmented LQR CARLA integration extracts live physics geometry
    (wheelbase ~3.0046 m, rear-axle offset ~-1.3864 m, max front steer ~70 deg)
    and initializes AugmentedLQRController with the measured wheelbase.
    """
    vehicle = MockTeslaVehicle(x=12.0, y=-60.90, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle)

    assert np.isclose(adapter.wheelbase, TESLA_WHEELBASE_M, atol=1e-4)
    assert np.isclose(
        adapter.geometry.rear_axle_offset_x,
        TESLA_REAR_OFFSET_X_M,
        atol=1e-4
    )
    assert np.isclose(adapter.geometry.max_front_wheel_steer_deg, 70.0, atol=1e-4)
    assert not np.isclose(adapter.wheelbase, 2.8, atol=1e-2)

    lqr = AugmentedLQRController(
        velocity=TARGET_SPEED_MPS,
        wheelbase=adapter.wheelbase,
        dt=FIXED_DELTA_SECONDS
    )
    assert np.isclose(lqr.wheelbase, TESLA_WHEELBASE_M, atol=1e-6)
    assert lqr.K.shape == (1, 3)
    assert np.all(lqr.K > 0.0)


# ==================================================
# 2. HEADING ERROR NORMALIZATION & LQR STATE CONSTRUCTION
# ==================================================

def test_heading_error_normalization_and_lqr_state_construction() -> None:
    """
    Verify heading error normalization across +/-pi boundaries and verify
    z_error = [e_y, e_psi, delta_applied - delta_ff] sign conventions match
    AugmentedLQRController.
    """
    assert np.isclose(normalize_angle(2.5 * math.pi), 0.5 * math.pi, atol=1e-12)
    assert np.isclose(normalize_angle(-2.5 * math.pi), -0.5 * math.pi, atol=1e-12)

    # Vehicle 1.2 m LEFT of straight road (actual_y > ref_y with ref_yaw = 0)
    e_y, e_psi, delta_err = compute_lqr_tracking_errors(
        actual_x=10.0,
        actual_y=62.10,
        actual_yaw=math.radians(5.0),
        ref_x=10.0,
        ref_y=60.90,
        ref_yaw=0.0,
        applied_steering_rad=math.radians(2.0),
        feedforward_steering_rad=math.radians(-4.5)
    )
    assert np.isclose(e_y, 1.20, atol=1e-12)
    assert np.isclose(e_psi, math.radians(5.0), atol=1e-12)
    assert np.isclose(delta_err, math.radians(6.5), atol=1e-12)

    # Positive e_y, e_psi, delta_err must produce negative LQR feedback steering rate
    lqr = AugmentedLQRController(
        velocity=TARGET_SPEED_MPS,
        wheelbase=TESLA_WHEELBASE_M,
        dt=FIXED_DELTA_SECONDS
    )
    u_fb = lqr.compute(
        lateral_error=e_y,
        heading_error=e_psi,
        steering_angle=delta_err
    )
    assert u_fb < 0.0


# ==================================================
# 3. CURVATURE ESTIMATION & CONTINUITY CHECKS
# ==================================================

def test_curvature_estimation_and_continuity() -> None:
    """
    Verify arc-length curvature estimation recovers the true curvature of a
    circular arc (R = 37.66 m -> kappa = -0.02655 m^-1) and rejects unphysical
    curvature derivative spikes.
    """
    radius_m = 37.66
    expected_kappa = -1.0 / radius_m  # right-hand turn in controller frame
    arc_lengths = np.linspace(0.0, 50.0, 101)
    angles = expected_kappa * arc_lengths

    road_x = radius_m * np.sin(-angles)
    road_y = radius_m * np.cos(-angles)
    road_yaw = angles.copy()

    profile = build_reference_route_profile(
        road_x=road_x,
        road_y=road_y,
        road_yaw=road_yaw
    )
    assert np.allclose(profile.curvature_1pm[10:-10], expected_kappa, atol=1e-4)
    assert np.allclose(profile.d_curvature_ds_1pm2[10:-10], 0.0, atol=1e-4)

    # Discontinuous curvature derivative must be rejected
    with pytest.raises(ValueError, match="exceeds continuity threshold"):
        verify_curvature_continuity(
            road_s=profile.road_s,
            curvature_1pm=profile.curvature_1pm,
            d_curvature_ds_1pm2=np.full_like(profile.road_s, 1.5),
            max_curvature_derivative_1pm2=0.25
        )


# ==================================================
# 4. FEEDFORWARD STEERING & FEEDFORWARD STEERING-RATE
# ==================================================

def test_feedforward_steering_and_rate_calculation() -> None:
    """
    Verify nominal feedforward steering delta_ff = atan(L * kappa_ref) and
    analytical chain-rule derivative d(delta_ff)/dt = [L / (1 + (L*kappa)^2)] * (dkappa/ds) * v * cos(e_psi).
    """
    kappa_ref = -0.02655
    dkappa_ds = -0.010
    speed_mps = 8.0
    heading_err = math.radians(3.0)

    delta_ff, delta_ff_rate = compute_curvature_feedforward_terms(
        wheelbase_m=TESLA_WHEELBASE_M,
        reference_curvature_1pm=kappa_ref,
        d_curvature_ds_1pm2=dkappa_ds,
        speed_mps=speed_mps,
        heading_error_rad=heading_err
    )

    expected_delta_ff = math.atan(TESLA_WHEELBASE_M * kappa_ref)
    expected_rate = (
        TESLA_WHEELBASE_M
        / (1.0 + (TESLA_WHEELBASE_M * kappa_ref) ** 2)
        * dkappa_ds
        * speed_mps
        * math.cos(heading_err)
    )
    assert np.isclose(delta_ff, expected_delta_ff, atol=1e-12)
    assert np.isclose(delta_ff_rate, expected_rate, atol=1e-12)


# ==================================================
# 5. STEERING-ANGLE & STEERING-RATE CONSTRAINTS
# ==================================================

def test_steering_rate_and_angle_constraints() -> None:
    """
    Verify LQR steering-rate saturation (+/-20 deg/s), steering-angle
    integration, and steering-angle saturation (+/-30 deg) without resetting
    steering to zero.
    """
    dt = FIXED_DELTA_SECONDS  # 0.05 s -> max 1.0 deg per step

    # Rate limit test from +5.0 deg with +100 deg/s request -> +6.0 deg
    next_steer, app_rate, lim_rate = apply_lqr_steering_rate_and_angle_constraints(
        requested_steering_rate_rad_s=math.radians(100.0),
        previous_applied_steering_rad=math.radians(5.0),
        dt=dt
    )
    assert np.isclose(lim_rate, MAX_STEERING_RATE_RAD_S, atol=1e-12)
    assert np.isclose(app_rate, MAX_STEERING_RATE_RAD_S, atol=1e-12)
    assert np.isclose(next_steer, math.radians(6.0), atol=1e-12)

    # Angle saturation test from +29.6 deg with +20 deg/s request -> clamps at +30.0 deg
    next_steer_sat, app_rate_sat, _ = apply_lqr_steering_rate_and_angle_constraints(
        requested_steering_rate_rad_s=math.radians(20.0),
        previous_applied_steering_rad=math.radians(29.6),
        dt=dt
    )
    assert np.isclose(next_steer_sat, MAX_STEERING_RAD, atol=1e-12)
    assert np.isclose(app_rate_sat, math.radians(8.0), atol=1e-12)


# ==================================================
# 6. ROUTE CONTINUITY, STABILITY, COLLISION & COMPLETION
# ==================================================

def test_route_continuity_stability_collision_and_completion() -> None:
    """
    Verify route continuity validation, numerical stability at zero/nominal
    speeds, collision stopping, off-route stopping, and endpoint completion.
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

    vehicle = MockTeslaVehicle(x=2.0, y=-61.90, yaw_deg=0.0, speed_mps=0.0)
    adapter = CarlaControllerAdapter(vehicle)
    lqr = AugmentedLQRController(
        velocity=TARGET_SPEED_MPS,
        wheelbase=adapter.wheelbase,
        dt=FIXED_DELTA_SECONDS
    )
    lon = ConservativeLongitudinalController(target_speed_mps=TARGET_SPEED_MPS)

    sample = compute_augmented_lqr_control_step(
        adapter=adapter,
        controller=lqr,
        speed_controller=lon,
        profile=profile,
        previous_applied_steering_rad=0.0,
        min_index=0,
        dt=FIXED_DELTA_SECONDS,
        step=0,
        sim_time_s=0.0
    )
    assert is_lqr_sample_finite(sample)
    # Vehicle is 1.0 m left of centerline -> e_y > 0 -> u_fb < 0 -> applied_steer < 0 -> carla_steer > 0
    assert sample.rear_axle_tracking_error_m > 0.0
    assert sample.feedback_steering_rate_rad_s < 0.0
    assert sample.applied_steering_rad < 0.0
    assert sample.normalized_carla_steer > 0.0

    # Stopping conditions
    stop, reason, completed = check_augmented_lqr_stopping_conditions(
        sample, road_x, road_y, collision_count=0
    )
    assert not stop and reason == "running" and not completed

    stop, reason, completed = check_augmented_lqr_stopping_conditions(
        sample, road_x, road_y, collision_count=1
    )
    assert stop and reason == "collision_detected" and not completed

    done_sample = replace(
        sample,
        front_x_m=float(road_x[-1]),
        front_y_m=float(road_y[-1]),
        front_nearest_index=len(road_x) - 1
    )
    stop, reason, completed = check_augmented_lqr_stopping_conditions(
        done_sample, road_x, road_y, collision_count=0
    )
    assert stop and reason == "route_endpoint_reached" and completed


# ==================================================
# 7. OUTPUT ARTIFACTS, METRICS & ACTOR/SETTINGS CLEANUP
# ==================================================

def test_outputs_metrics_and_cleanup_restoration(tmp_path: Path) -> None:
    """
    Verify CSV saving, all 5 required plot outputs, segment metric partitioning,
    and guaranteed actor/world-settings cleanup in try/finally on failure.
    """
    wps = build_chained_mock_waypoints(num_points=41, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    profile = build_reference_route_profile(road_x, road_y, road_yaw)

    vehicle = MockTeslaVehicle(x=2.0, y=-60.90, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle)
    lqr = AugmentedLQRController(
        velocity=TARGET_SPEED_MPS,
        wheelbase=adapter.wheelbase,
        dt=FIXED_DELTA_SECONDS
    )
    lon = ConservativeLongitudinalController(target_speed_mps=TARGET_SPEED_MPS)

    samples = []
    app_steer = 0.0
    for i in range(5):
        sm = compute_augmented_lqr_control_step(
            adapter=adapter,
            controller=lqr,
            speed_controller=lon,
            profile=profile,
            previous_applied_steering_rad=app_steer,
            min_index=0,
            dt=FIXED_DELTA_SECONDS,
            step=i,
            sim_time_s=i * FIXED_DELTA_SECONDS
        )
        samples.append(sm)
        app_steer = sm.applied_steering_rad

    csv_path = save_augmented_lqr_telemetry_to_csv(
        samples,
        tmp_path / "augmented_lqr_carla_8ms_telemetry.csv"
    )
    assert csv_path.exists()

    plot_paths = generate_augmented_lqr_carla_plots(
        samples=samples,
        profile=profile,
        output_dir=tmp_path,
        file_prefix="augmented_lqr_carla_8ms"
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

    seg_metrics = calculate_augmented_lqr_segment_metrics(samples, route_details)
    assert set(seg_metrics.keys()) == {
        "straight_approach",
        "sustained_curve",
        "straight_exit",
        "straight_combined",
        "overall"
    }

    summary = summarize_augmented_lqr_test_run(
        samples=samples,
        termination_reason="route_endpoint_reached",
        route_completed=True,
        collision_count=0,
        route_details=route_details,
        adapter=adapter,
        controller=lqr,
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

    with patch("carla_integration.augmented_lqr_carla.carla") as mock_carla_mod:
        mock_carla_mod.Client.return_value = mock_client
        with patch(
            "carla_integration.augmented_lqr_carla.compute_augmented_lqr_control_step",
            side_effect=RuntimeError("Simulated Augmented LQR failure")
        ):
            with pytest.raises(RuntimeError, match="Simulated Augmented LQR failure"):
                run_augmented_lqr_carla_test(
                    route_distance_m=10.0,
                    waypoint_spacing_m=0.5,
                    results_dir=tmp_path
                )

    assert not vehicle.is_alive
    mock_sensor.stop.assert_called_once()
    mock_sensor.destroy.assert_called_once()
    assert mock_world.apply_settings.call_args_list[-1][0][0] is original_settings
