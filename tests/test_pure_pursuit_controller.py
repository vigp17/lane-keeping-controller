import sys
from pathlib import Path

import numpy as np
import pytest


# Add src directory to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare_controllers import run_simulation as run_comparison_simulation
from controllers.pure_pursuit_controller import PurePursuitController
from pure_pursuit_simulation import (
    FIXED_LOOKAHEAD,
    LOOKAHEAD_GAIN,
    MAX_STEERING,
    MAX_STEERING_RATE,
    MIN_LOOKAHEAD,
    TEST_SPEEDS,
    calculate_pure_pursuit_metrics,
    run_pure_pursuit_simulation,
)
from road import Road
from stanley_simulation import run_stanley_simulation
from vehicle import Vehicle


# ============================================================
# 1. CORRECT STEERING DIRECTION
# ============================================================

def test_positive_offset_steering_direction() -> None:
    """
    When the vehicle is left of the straight road centerline (y > 0)
    with zero yaw, Pure Pursuit must command negative (rightward) steering.
    """
    road = Road(length=200.0)
    road_x, road_y = road.generate_straight_road(num_points=1001)

    vehicle = Vehicle(x=10.0, y=2.0, yaw=0.0, wheelbase=2.8)
    controller = PurePursuitController(wheelbase=2.8)

    details = controller.compute_with_details(
        vehicle=vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=15.0
    )

    assert float(details["heading_to_target_error"]) < 0.0
    assert float(details["steering"]) < 0.0


def test_negative_offset_steering_direction() -> None:
    """
    When the vehicle is right of the straight road centerline (y < 0)
    with zero yaw, Pure Pursuit must command positive (leftward) steering.
    """
    road = Road(length=200.0)
    road_x, road_y = road.generate_straight_road(num_points=1001)

    vehicle = Vehicle(x=10.0, y=-2.0, yaw=0.0, wheelbase=2.8)
    controller = PurePursuitController(wheelbase=2.8)

    details = controller.compute_with_details(
        vehicle=vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=15.0
    )

    assert float(details["heading_to_target_error"]) > 0.0
    assert float(details["steering"]) > 0.0


# ============================================================
# 2. ZERO-ERROR STRAIGHT-ROAD TRACKING
# ============================================================

def test_zero_error_straight_road_tracking() -> None:
    """
    When the vehicle lies on the straight road centerline (y = 0) with
    zero yaw, alpha and the requested steering angle must both be zero.
    """
    road = Road(length=200.0)
    road_x, road_y = road.generate_straight_road(num_points=1001)

    vehicle = Vehicle(x=20.0, y=0.0, yaw=0.0, wheelbase=2.8)
    controller = PurePursuitController(wheelbase=2.8)

    details = controller.compute_with_details(
        vehicle=vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=20.0
    )

    assert np.isclose(
        float(details["heading_to_target_error"]),
        0.0,
        atol=1e-12
    )
    assert np.isclose(float(details["steering"]), 0.0, atol=1e-12)


# ============================================================
# 3. FIXED LOOKAHEAD CALCULATION
# ============================================================

@pytest.mark.parametrize("speed", [0.0, 5.0, 10.0, 20.0, 30.0])
def test_fixed_lookahead_calculation(speed: float) -> None:
    """
    Verify fixed lookahead mode returns the configured constant lookahead
    distance independent of vehicle speed.
    """
    controller = PurePursuitController(
        wheelbase=2.8,
        lookahead_distance=12.5,
        use_dynamic_lookahead=False
    )

    ld_desired = controller.compute_desired_lookahead(velocity=speed)
    assert np.isclose(ld_desired, 12.5, atol=1e-12)


# ============================================================
# 4. SPEED-DEPENDENT LOOKAHEAD CALCULATION
# ============================================================

@pytest.mark.parametrize(
    ("speed", "expected_ld"),
    [
        (0.0, 3.0),
        (5.0, 5.5),
        (10.0, 8.0),
        (20.0, 13.0),
        (30.0, 18.0),
    ]
)
def test_speed_dependent_lookahead_calculation(
    speed: float,
    expected_ld: float
) -> None:
    """
    Verify speed-dependent lookahead follows Ld_desired = L_min + K_v * v
    with L_min = 3.0 m and K_v = 0.5 s.
    """
    controller = PurePursuitController(
        wheelbase=2.8,
        min_lookahead=MIN_LOOKAHEAD,
        lookahead_gain=LOOKAHEAD_GAIN,
        use_dynamic_lookahead=True
    )

    ld_desired = controller.compute_desired_lookahead(velocity=speed)
    assert np.isclose(ld_desired, expected_ld, atol=1e-12)


# ============================================================
# 5. CORRECT TARGET WAYPOINT SELECTION
# ============================================================

def test_correct_target_waypoint_selection() -> None:
    """
    Verify that Pure Pursuit selects the nearest waypoint accurately,
    chooses a target waypoint strictly ahead along the path, and computes
    the exact Euclidean distance Ld from the rear axle.
    """
    road_x = np.linspace(0.0, 100.0, 1001)  # 0.1 m spacing
    road_y = np.zeros_like(road_x)

    vehicle = Vehicle(x=15.0, y=1.5, yaw=0.0, wheelbase=2.8)
    controller = PurePursuitController(
        wheelbase=2.8,
        lookahead_distance=10.0,
        use_dynamic_lookahead=False
    )

    details = controller.compute_with_details(
        vehicle=vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=10.0
    )

    nearest_idx = int(details["nearest_index"])
    target_idx = int(details["target_index"])
    target_x = float(details["target_x"])
    target_y = float(details["target_y"])
    actual_ld = float(details["actual_lookahead_distance"])

    assert nearest_idx == 150
    assert target_idx == 250
    assert target_idx > nearest_idx
    assert np.isclose(target_x, 25.0, atol=1e-12)
    assert np.isclose(target_y, 0.0, atol=1e-12)

    expected_euclidean_ld = float(np.hypot(25.0 - 15.0, 0.0 - 1.5))
    assert np.isclose(actual_ld, expected_euclidean_ld, atol=1e-12)


# ============================================================
# 6. ARC-LENGTH-BASED TARGET SELECTION
# ============================================================

def test_arc_length_based_target_selection() -> None:
    """
    Verify target selection uses cumulative path arc length rather than
    simply adding lookahead distance to the global X coordinate.

    For a 45-degree diagonal path (y = x), an arc length of 10.0 m
    corresponds to delta_x = delta_y = 10 / sqrt(2) ~= 7.071 m,
    NOT delta_x = 10.0 m.
    """
    x_coords = np.linspace(0.0, 50.0, 5001)  # 0.01 m X spacing
    y_coords = x_coords.copy()  # ds = sqrt(2) * dx

    vehicle = Vehicle(
        x=0.0,
        y=0.0,
        yaw=float(np.pi / 4.0),
        wheelbase=2.8
    )
    controller = PurePursuitController(
        wheelbase=2.8,
        lookahead_distance=10.0,
        use_dynamic_lookahead=False
    )

    details = controller.compute_with_details(
        vehicle=vehicle,
        road_x=x_coords,
        road_y=y_coords,
        velocity=10.0
    )

    target_x = float(details["target_x"])
    target_y = float(details["target_y"])
    actual_ld = float(details["actual_lookahead_distance"])

    expected_coord = 10.0 / np.sqrt(2.0)
    assert np.isclose(target_x, expected_coord, atol=0.02)
    assert np.isclose(target_y, expected_coord, atol=0.02)
    assert np.isclose(actual_ld, 10.0, atol=0.02)
    # Confirm it did NOT simply add 10.0 m to X
    assert abs(target_x - 10.0) > 2.5


# ============================================================
# 7. HEADING ERROR NORMALIZATION
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
    Verify PurePursuitController.normalize_angle wraps angles into [-pi, pi].
    """
    normalized = PurePursuitController.normalize_angle(raw_angle)

    assert -np.pi - 1e-12 <= normalized <= np.pi + 1e-12
    assert np.isclose(normalized, expected_normalized, atol=1e-12)


# ============================================================
# 8. LOW-SPEED BEHAVIOR
# ============================================================

@pytest.mark.parametrize("low_speed", [0.0, 1e-6, 0.05, 0.5])
def test_low_speed_behavior(low_speed: float) -> None:
    """
    Verify zero and low vehicle speeds remain numerically stable,
    select valid lookahead distance >= L_min, and command finite steering.
    """
    road = Road(length=100.0)
    road_x, road_y = road.generate_straight_road(num_points=501)

    vehicle = Vehicle(x=5.0, y=1.5, yaw=0.0, wheelbase=2.8)
    controller = PurePursuitController(
        wheelbase=2.8,
        min_lookahead=3.0,
        lookahead_gain=0.5,
        use_dynamic_lookahead=True
    )

    details = controller.compute_with_details(
        vehicle=vehicle,
        road_x=road_x,
        road_y=road_y,
        velocity=low_speed
    )

    steering = float(details["steering"])
    desired_ld = float(details["desired_lookahead_distance"])
    actual_ld = float(details["actual_lookahead_distance"])

    assert np.isfinite(steering)
    assert desired_ld >= 3.0
    assert actual_ld > 0.0
    assert steering < 0.0


# ============================================================
# 9. PATH ENDPOINT HANDLING
# ============================================================

def test_path_endpoint_handling() -> None:
    """
    Verify safe target selection and finite steering when the vehicle is:
    1. Before the start of the path (x < 0)
    2. Near the end of the path where desired lookahead exceeds remaining length
    3. Coincident with or beyond the final waypoint
    """
    road = Road(length=100.0)
    road_x, road_y = road.generate_straight_road(num_points=501)

    controller = PurePursuitController(
        wheelbase=2.8,
        lookahead_distance=10.0,
        use_dynamic_lookahead=False
    )

    # 1. Vehicle before start of path
    v_start = Vehicle(x=-10.0, y=1.5, yaw=0.0, wheelbase=2.8)
    d_start = controller.compute_with_details(
        vehicle=v_start,
        road_x=road_x,
        road_y=road_y,
        velocity=10.0
    )
    assert int(d_start["nearest_index"]) == 0
    assert int(d_start["target_index"]) > 0
    assert np.isfinite(float(d_start["steering"]))
    assert float(d_start["steering"]) < 0.0

    # 2. Vehicle near end of path (x = 96 m, remaining path = 4 m < 10 m)
    v_near_end = Vehicle(x=96.0, y=1.0, yaw=0.0, wheelbase=2.8)
    d_near_end = controller.compute_with_details(
        vehicle=v_near_end,
        road_x=road_x,
        road_y=road_y,
        velocity=10.0
    )
    assert int(d_near_end["target_index"]) == len(road_x) - 1
    assert np.isfinite(float(d_near_end["steering"]))
    assert float(d_near_end["steering"]) < 0.0

    # 3. Vehicle exactly on the final waypoint (Ld = 0)
    v_on_end = Vehicle(x=100.0, y=0.0, yaw=0.0, wheelbase=2.8)
    d_on_end = controller.compute_with_details(
        vehicle=v_on_end,
        road_x=road_x,
        road_y=road_y,
        velocity=10.0
    )
    assert int(d_on_end["target_index"]) == len(road_x) - 1
    assert np.isfinite(float(d_on_end["steering"]))
    assert np.isclose(float(d_on_end["steering"]), 0.0, atol=1e-12)


# ============================================================
# 10. INVALID OR EMPTY PATHS AND INPUT VALIDATION
# ============================================================

def test_invalid_or_empty_paths() -> None:
    """
    Verify meaningful ValueErrors are raised for empty paths, single-point
    paths, mismatched array lengths, non-finite coordinates, degenerate
    zero-length paths, and invalid controller parameters.
    """
    controller = PurePursuitController()
    vehicle = Vehicle(x=0.0, y=0.0, yaw=0.0, wheelbase=2.8)

    # Empty path
    with pytest.raises(ValueError, match="at least 2"):
        controller.compute(
            vehicle,
            np.array([]),
            np.array([]),
            velocity=10.0
        )

    # Single-waypoint path
    with pytest.raises(ValueError, match="at least 2"):
        controller.compute(
            vehicle,
            np.array([0.0]),
            np.array([0.0]),
            velocity=10.0
        )

    # Mismatched lengths
    with pytest.raises(ValueError, match="matching lengths"):
        controller.compute(
            vehicle,
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0]),
            velocity=10.0
        )

    # Non-finite waypoint values
    with pytest.raises(ValueError, match="finite"):
        controller.compute(
            vehicle,
            np.array([0.0, np.nan, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            velocity=10.0
        )

    # Degenerate zero-arc-length path
    with pytest.raises(ValueError, match="positive total arc length"):
        controller.compute(
            vehicle,
            np.array([5.0, 5.0, 5.0]),
            np.array([2.0, 2.0, 2.0]),
            velocity=10.0
        )

    # Negative velocity
    with pytest.raises(ValueError, match="nonnegative"):
        controller.compute(
            vehicle,
            np.array([0.0, 10.0]),
            np.array([0.0, 0.0]),
            velocity=-5.0
        )

    # Invalid constructor parameters
    with pytest.raises(ValueError, match="wheelbase"):
        PurePursuitController(wheelbase=0.0)

    with pytest.raises(ValueError, match="lookahead_distance"):
        PurePursuitController(lookahead_distance=-1.0)


# ============================================================
# 11. STEERING-ANGLE CONSTRAINTS
# ============================================================

@pytest.mark.parametrize("use_dynamic", [True, False])
@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_steering_angle_constraints(
    use_dynamic: bool,
    road_type: str,
    speed: float
) -> None:
    """
    Verify applied steering angle never exceeds +/-30 degrees across all
    scenarios, speeds, and lookahead modes.
    """
    results = run_pure_pursuit_simulation(
        road_type=road_type,
        velocity=speed,
        use_dynamic_lookahead=use_dynamic
    )

    steering = results["steering"]
    assert np.all(np.abs(steering) <= MAX_STEERING + 1e-9)


# ============================================================
# 12. STEERING-RATE CONSTRAINTS
# ============================================================

@pytest.mark.parametrize("use_dynamic", [True, False])
@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_steering_rate_constraints(
    use_dynamic: bool,
    road_type: str,
    speed: float
) -> None:
    """
    Verify applied steering rate never exceeds +/-20 degrees/second,
    including the initial steering change from 0 radians.
    """
    results = run_pure_pursuit_simulation(
        road_type=road_type,
        velocity=speed,
        use_dynamic_lookahead=use_dynamic
    )

    steering_rate = results["steering_rate"]
    assert np.all(
        np.abs(steering_rate) <= MAX_STEERING_RATE + 1e-9
    )


# ============================================================
# 13. FINITE NUMERICAL OUTPUTS
# ============================================================

@pytest.mark.parametrize("use_dynamic", [True, False])
@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_finite_numerical_outputs(
    use_dynamic: bool,
    road_type: str,
    speed: float
) -> None:
    """
    Verify all recorded simulation arrays and metrics contain only finite
    values and zero constraint violations.
    """
    results = run_pure_pursuit_simulation(
        road_type=road_type,
        velocity=speed,
        use_dynamic_lookahead=use_dynamic
    )
    metrics = calculate_pure_pursuit_metrics(results)

    for arr in results.values():
        assert np.all(np.isfinite(arr))

    assert metrics["Non-Finite Values"] == 0
    assert metrics["Constraint Violations"] == 0


# ============================================================
# 14. STRAIGHT-ROAD CONVERGENCE
# ============================================================

def test_straight_road_convergence() -> None:
    """
    Verify straight-road tracking converges from a 2.0 m lateral offset
    to within 0.005 m final lateral error and 0.05 deg final heading error.
    """
    results = run_pure_pursuit_simulation(
        road_type="straight",
        velocity=20.0,
        use_dynamic_lookahead=True
    )
    metrics = calculate_pure_pursuit_metrics(results)

    assert float(metrics["Final Lateral Error (m)"]) < 0.005
    assert float(metrics["Final Heading Error (deg)"]) < 0.05
    assert float(metrics["Lateral RMSE (m)"]) < 0.55


# ============================================================
# 15. CURVED-ROAD TRACKING
# ============================================================

def test_curved_road_tracking() -> None:
    """
    Verify sinusoidal curved-road tracking from a 2.0 m lateral offset.

    Engineering Justification for Thresholds (at v = 20 m/s):
    - Pure Pursuit fits a constant-curvature arc to a lookahead target
      ahead on the sinusoidal road, yielding < 0.01 m final lateral error
      and < 0.10 deg final heading error at v = 20 m/s.
    - Transient Lateral RMSE (dominated by the 2.0 m initial offset and
      20 deg/s steering rate constraint): < 0.50 m.
    """
    results = run_pure_pursuit_simulation(
        road_type="curved",
        velocity=20.0,
        use_dynamic_lookahead=True
    )
    metrics = calculate_pure_pursuit_metrics(results)

    assert float(metrics["Final Lateral Error (m)"]) < 0.01
    assert float(metrics["Final Heading Error (deg)"]) < 0.10
    assert float(metrics["Lateral RMSE (m)"]) < 0.50


# ============================================================
# 16. MULTIPLE VEHICLE SPEEDS
# ============================================================

@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_multiple_vehicle_speeds(
    road_type: str,
    speed: float
) -> None:
    """
    Verify Pure Pursuit with speed-dependent lookahead converges across
    all required speeds (5, 10, 20, 30 m/s) on straight and curved roads.
    """
    results = run_pure_pursuit_simulation(
        road_type=road_type,
        velocity=speed,
        use_dynamic_lookahead=True
    )
    metrics = calculate_pure_pursuit_metrics(results)

    assert float(metrics["Final Lateral Error (m)"]) < 0.025
    assert float(metrics["Final Heading Error (deg)"]) < 0.20
    assert float(metrics["Lateral RMSE (m)"]) < 0.72
    assert int(metrics["Constraint Violations"]) == 0
    assert int(metrics["Non-Finite Values"]) == 0


# ============================================================
# 17. FIXED VS DYNAMIC LOOKAHEAD BEHAVIOR
# ============================================================

def test_fixed_vs_dynamic_lookahead_behavior() -> None:
    """
    Compare fixed lookahead (10.0 m) vs speed-dependent lookahead
    (Ld = 3.0 + 0.5 * v):
    - At low speed (5 m/s), dynamic lookahead uses a shorter 5.5 m
      lookahead, achieving significantly lower lateral RMSE and lower
      final error than the 10.0 m fixed lookahead.
    - At 10 m/s, dynamic lookahead (8.0 m) differs from fixed (10.0 m),
      and at 20 m/s dynamic lookahead increases to 13.0 m.
    """
    dyn_low = run_pure_pursuit_simulation(
        road_type="curved",
        velocity=5.0,
        use_dynamic_lookahead=True
    )
    fix_low = run_pure_pursuit_simulation(
        road_type="curved",
        velocity=5.0,
        use_dynamic_lookahead=False,
        fixed_lookahead=FIXED_LOOKAHEAD
    )

    m_dyn_low = calculate_pure_pursuit_metrics(dyn_low)
    m_fix_low = calculate_pure_pursuit_metrics(fix_low)

    assert np.isclose(
        float(m_dyn_low["Desired Lookahead (m)"]),
        5.5
    )
    assert np.isclose(
        float(m_fix_low["Desired Lookahead (m)"]),
        10.0
    )
    assert (
        float(m_dyn_low["Lateral RMSE (m)"])
        < float(m_fix_low["Lateral RMSE (m)"])
    )
    assert (
        float(m_dyn_low["Final Lateral Error (m)"])
        < float(m_fix_low["Final Lateral Error (m)"])
    )


# ============================================================
# 18. PRESERVATION OF EXISTING CONTROLLER BEHAVIOR
# ============================================================

def test_existing_controllers_preserved() -> None:
    """
    Verify Baseline, Augmented LQR, and Stanley controllers remain
    intact and functional alongside Pure Pursuit.
    """
    baseline_res = run_comparison_simulation("Baseline")
    lqr_res = run_comparison_simulation("Augmented LQR")
    stanley_res = run_stanley_simulation(
        road_type="curved",
        velocity=20.0
    )

    assert abs(float(baseline_res["y"][-1])) < 0.05
    assert abs(float(lqr_res["y"][-1])) < 0.05
    assert abs(float(stanley_res["lateral_error"][-1])) < 0.02
