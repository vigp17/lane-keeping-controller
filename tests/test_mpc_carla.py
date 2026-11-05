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
from carla_integration.carla_adapter import CarlaControllerAdapter
from carla_integration.mpc_carla import (
    ConservativeLongitudinalController,
    FIXED_DELTA_SECONDS,
    MAX_STEERING_RAD,
    MAX_STEERING_RATE_RAD_S,
    MPC_HORIZON_DURATION_S,
    MPC_HORIZON_STEPS,
    RouteDetails,
    TARGET_SPEED_MPS,
    calculate_mpc_segment_metrics,
    check_mpc_stopping_conditions,
    compute_curvature_preview_horizon,
    compute_mpc_control_step,
    generate_mpc_carla_plots,
    is_mpc_sample_finite,
    run_mpc_carla_test,
    save_mpc_telemetry_to_csv,
    summarize_mpc_test_run,
    verify_route_continuity,
)
from carla_integration.stanley_carla import extract_continuous_driving_route
from controllers.mpc_controller import MPCController


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
    num_points: int = 61,
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
# 1. ADAPTER INTEGRATION, GEOMETRY & PREDICTION TIMESTEP
# ==================================================

def test_adapter_geometry_and_mpc_prediction_model_timestep() -> None:
    """
    Verify CarlaControllerAdapter extracts live Tesla geometry (~3.0046 m
    wheelbase, ~-1.3864 m rear-axle offset, 70 deg max front steer) and
    MPCController recalculates Ad, Bd, Ed for dt = 0.05 s and N = 15 (0.75 s)
    rather than reusing offline dt = 0.1 s or L = 2.8 m matrices.
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
    assert np.isclose(MPC_HORIZON_DURATION_S, 0.75, atol=1e-12)

    mpc_carla = MPCController(
        velocity=TARGET_SPEED_MPS,
        wheelbase=adapter.wheelbase,
        dt=FIXED_DELTA_SECONDS,
        horizon=MPC_HORIZON_STEPS
    )
    mpc_offline = MPCController(
        velocity=TARGET_SPEED_MPS,
        wheelbase=2.8,
        dt=0.1,
        horizon=MPC_HORIZON_STEPS
    )

    assert mpc_carla.N == 15
    assert np.isclose(mpc_carla.dt, 0.05, atol=1e-12)
    assert np.isclose(mpc_carla.wheelbase, TESLA_WHEELBASE_M, atol=1e-6)

    # Verify exact ZOH entries for v = 8.0, dt = 0.05, L = 3.004645
    v = TARGET_SPEED_MPS
    T = FIXED_DELTA_SECONDS
    L = adapter.wheelbase
    assert np.isclose(mpc_carla.Ad[0, 1], v * T, atol=1e-12)
    assert np.isclose(mpc_carla.Ad[0, 2], (v ** 2 * T ** 2) / (2.0 * L), atol=1e-12)
    assert np.isclose(mpc_carla.Bd[2, 0], T, atol=1e-12)
    assert np.isclose(mpc_carla.Ed[0, 0], -0.5 * (v ** 2) * (T ** 2), atol=1e-12)
    assert np.isclose(mpc_carla.Ed[1, 0], -v * T, atol=1e-12)

    # Must differ from dt = 0.1, L = 2.8 offline matrices
    assert not np.allclose(mpc_carla.Ad, mpc_offline.Ad)
    assert not np.allclose(mpc_carla.Bd, mpc_offline.Bd)
    assert not np.allclose(mpc_carla.Ed, mpc_offline.Ed)


# ==================================================
# 2. CURVATURE PREVIEW HORIZON EVALUATION
# ==================================================

def test_curvature_preview_horizon_varies_along_arc_length() -> None:
    """
    Verify compute_curvature_preview_horizon advances s_preview[0:N] along
    cumulative arc length and interpolates future curvature values without
    repeating the current curvature value across a straight-to-curve transition.
    """
    # Build a 40 m straight road followed by a 40 m circular curve (R = 37.66 m)
    s_straight = np.linspace(0.0, 40.0, 81)
    x_straight = s_straight.copy()
    y_straight = np.zeros_like(s_straight)
    yaw_straight = np.zeros_like(s_straight)

    radius_m = 37.66
    kappa_curve = -1.0 / radius_m
    s_curve_rel = np.linspace(0.5, 40.0, 80)
    angles = kappa_curve * s_curve_rel
    x_curve = 40.0 + radius_m * np.sin(-angles)
    y_curve = -radius_m * (1.0 - np.cos(-angles))
    yaw_curve = angles.copy()

    road_x = np.concatenate([x_straight, x_curve])
    road_y = np.concatenate([y_straight, y_curve])
    road_yaw = np.concatenate([yaw_straight, yaw_curve])

    profile = build_reference_route_profile(road_x, road_y, road_yaw)

    # Query at s = 37.5 m (2.5 m before the curve entry; over 0.75 s at 8 m/s = 6.0 m lookahead)
    kappa_preview, s_preview = compute_curvature_preview_horizon(
        profile=profile,
        current_s_m=37.5,
        speed_mps=8.0,
        heading_error_rad=0.0,
        dt=0.05,
        horizon=15
    )

    assert kappa_preview.shape == (15,)
    assert s_preview.shape == (15,)
    assert np.all(np.diff(s_preview) > 0.0)
    assert np.isclose(s_preview[0], 37.5, atol=1e-9)
    assert np.isclose(s_preview[-1], 37.5 + 14 * 8.0 * 0.05, atol=1e-9)

    # Start of horizon is near straight (|kappa[0]| ~ 0), end of horizon is inside curve (kappa[-1] ~ -0.0265)
    assert abs(float(kappa_preview[0])) < 0.002
    assert np.isclose(float(kappa_preview[-1]), kappa_curve, atol=1e-3)
    assert float(np.std(kappa_preview)) > 0.005


# ==================================================
# 3. MPC STATE CONSTRUCTION & STEERING CONSTRAINTS
# ==================================================

def test_mpc_state_construction_and_hard_constraints() -> None:
    """
    Verify MPC state z0 = [e_y, e_psi, delta_applied] construction, corrective
    steering direction, initial steering continuity, and hard +/-30 deg angle
    and +/-20 deg/s rate limits.
    """
    wps = build_chained_mock_waypoints(num_points=61, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    profile = build_reference_route_profile(road_x, road_y, road_yaw)

    # Place vehicle 1.2 m LEFT of centerline in controller frame (ctrl_y = +62.10 -> carla_y = -62.10)
    vehicle_left = MockTeslaVehicle(x=5.0, y=-62.10, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle_left)
    mpc = MPCController(
        velocity=TARGET_SPEED_MPS,
        wheelbase=adapter.wheelbase,
        dt=FIXED_DELTA_SECONDS,
        horizon=MPC_HORIZON_STEPS,
        max_steering=MAX_STEERING_RAD,
        max_steering_rate=MAX_STEERING_RATE_RAD_S
    )
    lon = ConservativeLongitudinalController(target_speed_mps=TARGET_SPEED_MPS)

    prev_steer = math.radians(-3.0)
    sample = compute_mpc_control_step(
        adapter=adapter,
        controller=mpc,
        speed_controller=lon,
        profile=profile,
        previous_applied_steering_rad=prev_steer,
        min_index=0,
        consecutive_solver_failures=0,
        dt=FIXED_DELTA_SECONDS,
        step=0,
        sim_time_s=0.0
    )

    assert sample.solver_status in ("optimal", "optimal_inaccurate")
    assert not sample.used_fallback
    assert sample.rear_axle_tracking_error_m > 0.0
    # Left of centerline -> negative steering rate -> more negative steering angle -> positive CARLA steer
    assert sample.applied_steering_rate_rad_s < 0.0
    assert sample.applied_steering_rad < prev_steer
    assert sample.normalized_carla_steer > 0.0

    # Verify rate and angle bounds
    assert abs(sample.applied_steering_rate_rad_s) <= MAX_STEERING_RATE_RAD_S + 1e-9
    assert abs(sample.applied_steering_rad) <= MAX_STEERING_RAD + 1e-9
    assert np.isclose(
        sample.applied_steering_rad,
        prev_steer + sample.applied_steering_rate_rad_s * FIXED_DELTA_SECONDS,
        atol=1e-9
    )


# ==================================================
# 4. SOLVER FAILURE FALLBACK & REPEATED FAILURE STOP
# ==================================================

def test_solver_failure_fallback_and_repeated_failure_stop() -> None:
    """
    Verify that when an optimization failure occurs:
        1. Failure is recorded (used_fallback=True, solver_failure_count incremented).
        2. Fallback command is finite and respects +/-30 deg and +/-20 deg/s limits.
        3. Repeated solver failures trigger check_mpc_stopping_conditions with
           reason 'repeated_solver_failures'.
    """
    wps = build_chained_mock_waypoints(num_points=41, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    profile = build_reference_route_profile(road_x, road_y, road_yaw)

    vehicle = MockTeslaVehicle(x=4.0, y=-61.50, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle)
    mpc = MPCController(
        velocity=TARGET_SPEED_MPS,
        wheelbase=adapter.wheelbase,
        dt=FIXED_DELTA_SECONDS,
        horizon=MPC_HORIZON_STEPS
    )
    lon = ConservativeLongitudinalController(target_speed_mps=TARGET_SPEED_MPS)

    fallback_sample = compute_mpc_control_step(
        adapter=adapter,
        controller=mpc,
        speed_controller=lon,
        profile=profile,
        previous_applied_steering_rad=math.radians(5.0),
        min_index=0,
        consecutive_solver_failures=4,
        dt=FIXED_DELTA_SECONDS,
        step=10,
        sim_time_s=0.50,
        force_solver_failure=True
    )

    assert fallback_sample.used_fallback is True
    assert mpc.solver_failure_count == 1
    assert fallback_sample.consecutive_solver_failures == 5
    assert is_mpc_sample_finite(fallback_sample)
    assert abs(fallback_sample.applied_steering_rad) <= MAX_STEERING_RAD + 1e-9
    assert abs(fallback_sample.applied_steering_rate_rad_s) <= MAX_STEERING_RATE_RAD_S + 1e-9

    # Consecutive failures reached threshold (5) -> must stop
    stop, reason, completed = check_mpc_stopping_conditions(
        sample=fallback_sample,
        road_x=road_x,
        road_y=road_y,
        collision_count=0,
        total_solver_failures=mpc.solver_failure_count,
        max_consecutive_failures=5
    )
    assert stop is True
    assert reason == "repeated_solver_failures"
    assert completed is False


# ==================================================
# 5. ROUTE CONTINUITY, COLLISION & ROUTE COMPLETION
# ==================================================

def test_route_continuity_collision_and_completion() -> None:
    """
    Verify route continuity validation, collision stopping, off-route stopping,
    non-finite stopping, and route endpoint completion.
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

    vehicle = MockTeslaVehicle(x=2.0, y=-60.90, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle)
    mpc = MPCController(
        velocity=TARGET_SPEED_MPS,
        wheelbase=adapter.wheelbase,
        dt=FIXED_DELTA_SECONDS,
        horizon=MPC_HORIZON_STEPS
    )
    lon = ConservativeLongitudinalController(target_speed_mps=TARGET_SPEED_MPS)

    nominal = compute_mpc_control_step(
        adapter=adapter,
        controller=mpc,
        speed_controller=lon,
        profile=profile,
        previous_applied_steering_rad=0.0,
        min_index=0,
        consecutive_solver_failures=0,
        dt=FIXED_DELTA_SECONDS,
        step=0,
        sim_time_s=0.0
    )

    # Normal running
    stop, reason, completed = check_mpc_stopping_conditions(
        nominal, road_x, road_y, collision_count=0
    )
    assert not stop and reason == "running" and not completed

    # Collision detected
    stop, reason, completed = check_mpc_stopping_conditions(
        nominal, road_x, road_y, collision_count=1
    )
    assert stop and reason == "collision_detected" and not completed

    # Route completion
    done_sample = replace(
        nominal,
        front_x_m=float(road_x[-1]),
        front_y_m=float(road_y[-1]),
        front_nearest_index=len(road_x) - 1
    )
    stop, reason, completed = check_mpc_stopping_conditions(
        done_sample, road_x, road_y, collision_count=0
    )
    assert stop and reason == "route_endpoint_reached" and completed


# ==================================================
# 6. ARTIFACTS, TIMING METRICS & CLEANUP RESTORATION
# ==================================================

def test_outputs_timing_metrics_and_cleanup_restoration(tmp_path: Path) -> None:
    """
    Verify CSV saving, all 6 required plot outputs (including solver_time),
    segment metric partitioning, and guaranteed actor/world-settings cleanup
    in try/finally on exception.
    """
    wps = build_chained_mock_waypoints(num_points=41, spacing_m=0.5, carla_y=-60.90)
    road_x, road_y, road_yaw = CarlaControllerAdapter.convert_waypoints(wps)
    profile = build_reference_route_profile(road_x, road_y, road_yaw)

    vehicle = MockTeslaVehicle(x=2.0, y=-60.90, yaw_deg=0.0, speed_mps=8.0)
    adapter = CarlaControllerAdapter(vehicle)
    mpc = MPCController(
        velocity=TARGET_SPEED_MPS,
        wheelbase=adapter.wheelbase,
        dt=FIXED_DELTA_SECONDS,
        horizon=MPC_HORIZON_STEPS
    )
    lon = ConservativeLongitudinalController(target_speed_mps=TARGET_SPEED_MPS)

    samples = []
    app_steer = 0.0
    for i in range(5):
        sm = compute_mpc_control_step(
            adapter=adapter,
            controller=mpc,
            speed_controller=lon,
            profile=profile,
            previous_applied_steering_rad=app_steer,
            min_index=0,
            consecutive_solver_failures=0,
            dt=FIXED_DELTA_SECONDS,
            step=i,
            sim_time_s=i * FIXED_DELTA_SECONDS
        )
        samples.append(sm)
        app_steer = sm.applied_steering_rad

    csv_path = save_mpc_telemetry_to_csv(
        samples,
        tmp_path / "mpc_carla_8ms_telemetry.csv"
    )
    assert csv_path.exists()

    plot_paths = generate_mpc_carla_plots(
        samples=samples,
        profile=profile,
        output_dir=tmp_path,
        file_prefix="mpc_carla_8ms"
    )
    assert set(plot_paths.keys()) == {
        "trajectory",
        "error",
        "steering",
        "speed",
        "curvature",
        "solver_time"
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

    seg_metrics = calculate_mpc_segment_metrics(samples, route_details)
    assert set(seg_metrics.keys()) == {
        "straight_approach",
        "sustained_curve",
        "straight_exit",
        "straight_combined",
        "overall"
    }

    summary = summarize_mpc_test_run(
        samples=samples,
        termination_reason="route_endpoint_reached",
        route_completed=True,
        collision_count=0,
        route_details=route_details,
        adapter=adapter,
        controller=mpc,
        csv_path=csv_path,
        plot_paths=plot_paths
    )
    assert summary.route_completed
    assert summary.solver_failure_count == 0
    assert summary.fallback_step_count == 0
    assert summary.angle_constraint_violations == 0
    assert summary.rate_constraint_violations == 0
    assert summary.non_finite_values == 0
    assert summary.mean_solver_time_ms > 0.0
    assert summary.mean_total_controller_time_ms > 0.0

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

    with patch("carla_integration.mpc_carla.carla") as mock_carla_mod:
        mock_carla_mod.Client.return_value = mock_client
        with patch(
            "carla_integration.mpc_carla.compute_mpc_control_step",
            side_effect=RuntimeError("Simulated MPC failure")
        ):
            with pytest.raises(RuntimeError, match="Simulated MPC failure"):
                run_mpc_carla_test(
                    route_distance_m=10.0,
                    waypoint_spacing_m=0.5,
                    results_dir=tmp_path
                )

    assert not vehicle.is_alive
    mock_sensor.stop.assert_called_once()
    mock_sensor.destroy.assert_called_once()
    assert mock_world.apply_settings.call_args_list[-1][0][0] is original_settings
