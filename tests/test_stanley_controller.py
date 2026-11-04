import sys
from pathlib import Path

import numpy as np
import pytest


# Add src directory to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllers.stanley_controller import StanleyController
from road import Road
from stanley_simulation import (
    MAX_STEERING,
    MAX_STEERING_RATE,
    TEST_SPEEDS,
    calculate_stanley_metrics,
    run_stanley_simulation,
)
from vehicle import Vehicle


# ============================================================
# 1. FRONT-AXLE GEOMETRY TESTS
# ============================================================

@pytest.mark.parametrize(
    ("yaw_deg", "expected_fx", "expected_fy"),
    [
        (0.0, 10.0 + 2.8, 5.0),
        (90.0, 10.0, 5.0 + 2.8),
        (-90.0, 10.0, 5.0 - 2.8),
    ]
)
def test_front_axle_geometry(
    yaw_deg: float,
    expected_fx: float,
    expected_fy: float
) -> None:
    """
    Verify front-axle position calculation at 0, +90, and -90 degrees yaw.
    """
    vehicle = Vehicle(
        x=10.0,
        y=5.0,
        yaw=float(np.radians(yaw_deg)),
        wheelbase=2.8
    )

    front_x, front_y = vehicle.get_front_axle_position()

    assert np.isclose(front_x, expected_fx, atol=1e-12)
    assert np.isclose(front_y, expected_fy, atol=1e-12)


# ============================================================
# 2. POSITIVE LATERAL OFFSET STEERING DIRECTION
# ============================================================

def test_positive_lateral_offset_steering_direction() -> None:
    """
    For a straight road along the X axis with vehicle.y > 0 and yaw = 0,
    Stanley must command negative (rightward) steering toward the centerline.
    """
    road = Road(length=200.0)
    road_x, road_y = road.generate_straight_road(num_points=1000)
    road_yaw = np.zeros_like(road_x)

    vehicle = Vehicle(x=10.0, y=2.0, yaw=0.0, wheelbase=2.8)
    controller = StanleyController(stanley_gain=1.5, epsilon=1e-3)

    details = controller.compute_with_details(
        vehicle=vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=15.0,
        road_yaw=road_yaw
    )

    assert float(details["cross_track_error"]) < 0.0
    assert float(details["steering"]) < 0.0


# ============================================================
# 3. NEGATIVE LATERAL OFFSET STEERING DIRECTION
# ============================================================

def test_negative_lateral_offset_steering_direction() -> None:
    """
    For a straight road along the X axis with vehicle.y < 0 and yaw = 0,
    Stanley must command positive (leftward) steering toward the centerline.
    """
    road = Road(length=200.0)
    road_x, road_y = road.generate_straight_road(num_points=1000)
    road_yaw = np.zeros_like(road_x)

    vehicle = Vehicle(x=10.0, y=-2.0, yaw=0.0, wheelbase=2.8)
    controller = StanleyController(stanley_gain=1.5, epsilon=1e-3)

    details = controller.compute_with_details(
        vehicle=vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=15.0,
        road_yaw=road_yaw
    )

    assert float(details["cross_track_error"]) > 0.0
    assert float(details["steering"]) > 0.0


# ============================================================
# 4. ZERO-ERROR STEERING RESPONSE
# ============================================================

def test_zero_error_steering_response() -> None:
    """
    When the vehicle lies on the straight reference road with zero heading
    error, cross-track error and steering output must both be zero.
    """
    road = Road(length=200.0)
    road_x, road_y = road.generate_straight_road(num_points=1000)
    road_yaw = np.zeros_like(road_x)

    vehicle = Vehicle(x=20.0, y=0.0, yaw=0.0, wheelbase=2.8)
    controller = StanleyController(stanley_gain=1.5, epsilon=1e-3)

    steering = controller.compute(
        vehicle=vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=20.0,
        road_yaw=road_yaw
    )

    assert np.isclose(steering, 0.0, atol=1e-12)


# ============================================================
# 5. HEADING ERROR NORMALIZATION
# ============================================================

@pytest.mark.parametrize(
    ("raw_angle", "expected_normalized"),
    [
        (0.0, 0.0),
        (2.0 * np.pi, 0.0),
        (-2.0 * np.pi, 0.0),
        (1.5 * np.pi, -0.5 * np.pi),
        (-1.5 * np.pi, 0.5 * np.pi),
        (3.25 * np.pi, -0.75 * np.pi),
    ]
)
def test_heading_error_normalization(
    raw_angle: float,
    expected_normalized: float
) -> None:
    """
    Ensure angle normalization wraps any angle in radians into [-pi, pi].
    """
    normalized = StanleyController.normalize_angle(raw_angle)

    assert -np.pi - 1e-12 <= normalized <= np.pi + 1e-12
    assert np.isclose(normalized, expected_normalized, atol=1e-12)


# ============================================================
# 6. LOW-SPEED NUMERICAL STABILITY
# ============================================================

@pytest.mark.parametrize("low_speed", [0.0, 1e-6, 0.01, 0.5])
def test_low_speed_numerical_stability(low_speed: float) -> None:
    """
    Verify that low and zero forward speeds do not cause division by zero
    and produce finite, bounded steering commands.
    """
    road = Road(length=100.0)
    road_x, road_y = road.generate_straight_road(num_points=500)
    road_yaw = np.zeros_like(road_x)

    vehicle = Vehicle(x=5.0, y=1.5, yaw=0.0, wheelbase=2.8)
    controller = StanleyController(stanley_gain=1.5, epsilon=1e-3)

    steering = controller.compute(
        vehicle=vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=low_speed,
        road_yaw=road_yaw
    )

    assert np.isfinite(steering)
    assert -0.5 * np.pi <= steering <= 0.5 * np.pi
    assert steering < 0.0


# ============================================================
# 7. STEERING-ANGLE CONSTRAINT
# ============================================================

@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_steering_angle_constraint(
    road_type: str,
    speed: float
) -> None:
    """
    Verify applied steering angle never exceeds +/-30 degrees.
    """
    results = run_stanley_simulation(
        road_type=road_type,
        velocity=speed
    )

    steering = results["steering"]

    assert np.all(np.abs(steering) <= MAX_STEERING + 1e-9)


# ============================================================
# 8. STEERING-RATE CONSTRAINT
# ============================================================

@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_steering_rate_constraint(
    road_type: str,
    speed: float
) -> None:
    """
    Verify applied steering rate never exceeds +/-20 degrees/second,
    including the initial step from 0 radians.
    """
    results = run_stanley_simulation(
        road_type=road_type,
        velocity=speed
    )

    steering_rate = results["steering_rate"]

    assert np.all(
        np.abs(steering_rate) <= MAX_STEERING_RATE + 1e-9
    )


# ============================================================
# 9. FINITE NUMERICAL OUTPUTS
# ============================================================

@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_finite_numerical_outputs(
    road_type: str,
    speed: float
) -> None:
    """
    Verify all recorded simulation arrays and computed metrics are finite.
    """
    results = run_stanley_simulation(
        road_type=road_type,
        velocity=speed
    )
    metrics = calculate_stanley_metrics(results)

    for arr in results.values():
        assert np.all(np.isfinite(arr))

    assert metrics["Non-Finite Values"] == 0
    assert metrics["Constraint Violations"] == 0


# ============================================================
# 10. STRAIGHT-ROAD CONVERGENCE
# ============================================================

def test_straight_road_convergence() -> None:
    """
    Verify straight-road tracking converges from a 2.0 m lateral offset
    to within 0.01 m final lateral error and 0.1 deg final heading error.
    """
    results = run_stanley_simulation(
        road_type="straight",
        velocity=20.0
    )
    metrics = calculate_stanley_metrics(results)

    assert float(metrics["Final Lateral Error (m)"]) < 0.01
    assert float(metrics["Final Heading Error (deg)"]) < 0.1
    assert float(metrics["Lateral RMSE (m)"]) < 0.55


# ============================================================
# 11. CURVED-ROAD TRACKING
# ============================================================

def test_curved_road_tracking() -> None:
    """
    Verify sinusoidal curved-road tracking from a 2.0 m lateral offset.

    Engineering Justification for Thresholds (at v = 20 m/s):
    - Kinematic bicycle without curvature feedforward exhibits a small
      steady-state geometric lag proportional to wheelbase * curvature
      (L * kappa_max = 2.8 * 0.0008 = 0.0022 m).
    - Final lateral error threshold: < 0.02 m (2 cm)
    - Final heading error threshold: < 0.25 deg
    - Transient Rear-Axle Lateral RMSE (including 2.0 m initial offset and
      20 deg/s rate limit): < 0.50 m
    - Front-Axle CTE RMSE: < 0.45 m
    """
    results = run_stanley_simulation(
        road_type="curved",
        velocity=20.0
    )
    metrics = calculate_stanley_metrics(results)

    assert float(metrics["Final Lateral Error (m)"]) < 0.02
    assert float(metrics["Final Heading Error (deg)"]) < 0.25
    assert float(metrics["Lateral RMSE (m)"]) < 0.50
    assert float(metrics["Front-Axle CTE RMSE (m)"]) < 0.45


# ============================================================
# 12. REFERENCE WAYPOINT SELECTION
# ============================================================

def test_reference_waypoint_selection() -> None:
    """
    Verify nearest waypoint lookup selects the exact closest point on the path.
    """
    road = Road(length=100.0)
    road_x, road_y = road.generate_curved_road(num_points=501)
    road_yaw = road.reference_heading(road_x)

    controller = StanleyController()

    target_index = 250
    query_x = float(road_x[target_index])
    query_y = float(road_y[target_index])

    idx, ref_x, ref_y, ref_yaw = controller.find_nearest_waypoint(
        front_x=query_x,
        front_y=query_y,
        road_x=road_x,
        road_y=road_y,
        road_yaw=road_yaw
    )

    assert idx == target_index
    assert np.isclose(ref_x, query_x)
    assert np.isclose(ref_y, query_y)
    assert np.isclose(ref_yaw, float(road_yaw[target_index]))


# ============================================================
# 13. PATH ENDPOINT HANDLING
# ============================================================

def test_path_endpoint_handling() -> None:
    """
    Verify that vehicle front-axle positions before the start of the path
    (x < 0) and beyond the end of the path (x > length) clamp cleanly to
    the first and last waypoints without index errors or NaNs.
    """
    road = Road(length=100.0)
    road_x, road_y = road.generate_straight_road(num_points=200)
    road_yaw = np.zeros_like(road_x)

    controller = StanleyController()

    # Before start of path
    vehicle_before = Vehicle(x=-25.0, y=1.0, yaw=0.0, wheelbase=2.8)
    details_before = controller.compute_with_details(
        vehicle=vehicle_before,
        road_x=road_x,
        road_y=road_y,
        velocity=10.0,
        road_yaw=road_yaw
    )
    assert int(details_before["nearest_index"]) == 0
    assert np.isfinite(float(details_before["steering"]))
    assert float(details_before["steering"]) < 0.0

    # Beyond end of path
    vehicle_after = Vehicle(x=150.0, y=-1.0, yaw=0.0, wheelbase=2.8)
    details_after = controller.compute_with_details(
        vehicle=vehicle_after,
        road_x=road_x,
        road_y=road_y,
        velocity=10.0,
        road_yaw=road_yaw
    )
    assert int(details_after["nearest_index"]) == len(road_x) - 1
    assert np.isfinite(float(details_after["steering"]))
    assert float(details_after["steering"]) > 0.0


# ============================================================
# 14. MULTIPLE VEHICLE SPEEDS
# ============================================================

@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_multiple_vehicle_speeds(
    road_type: str,
    speed: float
) -> None:
    """
    Verify Stanley tracking across all required speeds (5, 10, 20, 30 m/s)
    on both straight and curved roads.
    """
    results = run_stanley_simulation(
        road_type=road_type,
        velocity=speed
    )
    metrics = calculate_stanley_metrics(results)

    # All speeds must converge to < 2.5 cm final lateral error and < 0.25 deg heading error
    assert float(metrics["Final Lateral Error (m)"]) < 0.025
    assert float(metrics["Final Heading Error (deg)"]) < 0.25
    assert float(metrics["Lateral RMSE (m)"]) < 0.70
    assert int(metrics["Constraint Violations"]) == 0
    assert int(metrics["Non-Finite Values"]) == 0
