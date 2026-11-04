import sys
from pathlib import Path

import numpy as np


# Add src directory to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road import Road


def test_straight_road_backward_compatibility() -> None:

    road = Road(length=100.0)

    x, y = road.generate_straight_road()

    assert len(x) == 500
    assert len(y) == 500
    assert np.isclose(x[0], 0.0)
    assert np.isclose(x[-1], 100.0)
    assert np.allclose(y, 0.0)


def test_reference_lateral_position() -> None:

    road = Road(length=100.0)

    assert np.isclose(road.A, 2.0)
    assert np.isclose(road.k, 0.02)

    # At x = 0, sin(0) = 0 -> y_ref = 0
    assert np.isclose(
        road.reference_lateral_position(0.0),
        0.0
    )

    # At quarter wavelength k*x = pi/2, sin(pi/2) = 1 -> y_ref = A
    x_peak = float(np.pi / (2.0 * road.k))
    assert np.isclose(
        road.reference_lateral_position(x_peak),
        road.A
    )

    x_samples, y_samples = road.generate_curved_road()
    expected_y = road.A * np.sin(road.k * x_samples)
    assert np.allclose(y_samples, expected_y)


def test_reference_heading_calculation() -> None:

    road = Road(
        length=100.0,
        amplitude=2.0,
        frequency=0.02
    )

    # At x = 0, yaw_ref = atan(A * k * cos(0)) = atan(A * k)
    expected_initial_yaw = float(
        np.arctan(road.A * road.k)
    )
    assert np.isclose(
        road.reference_heading(0.0),
        expected_initial_yaw
    )

    # At peak k*x = pi/2, slope is zero -> yaw_ref = 0
    x_peak = float(np.pi / (2.0 * road.k))
    assert np.isclose(
        road.reference_heading(x_peak),
        0.0,
        atol=1e-12
    )

    # Array evaluation matches analytical formula
    x = np.linspace(0.0, road.length, 500)
    expected_yaw = np.arctan(
        road.A * road.k * np.cos(road.k * x)
    )
    assert np.allclose(
        road.reference_heading(x),
        expected_yaw
    )


def test_curvature_calculation() -> None:

    road = Road(
        length=100.0,
        amplitude=2.0,
        frequency=0.02
    )

    # At x = 0, sin(0) = 0 -> curvature = 0
    assert np.isclose(
        road.road_curvature(0.0),
        0.0,
        atol=1e-12
    )

    # At peak k*x = pi/2, sin(pi/2) = 1 and cos(pi/2) = 0
    # curvature = -A * k^2
    x_peak = float(np.pi / (2.0 * road.k))
    expected_peak_curvature = -road.A * (road.k ** 2)
    assert np.isclose(
        road.road_curvature(x_peak),
        expected_peak_curvature
    )

    # Array evaluation matches analytical formula
    x = np.linspace(0.0, road.length, 500)
    dy_dx = road.A * road.k * np.cos(road.k * x)
    d2y_dx2 = -road.A * (road.k ** 2) * np.sin(road.k * x)
    expected_curvature = d2y_dx2 / (
        (1.0 + dy_dx ** 2) ** (3.0 / 2.0)
    )
    assert np.allclose(
        road.road_curvature(x),
        expected_curvature
    )


def test_finite_numerical_outputs() -> None:

    road = Road(
        length=300.0,
        amplitude=2.0,
        frequency=0.02
    )

    x_straight, y_straight = road.generate_straight_road()
    x_curved, y_curved = road.generate_curved_road()

    y_ref = road.reference_lateral_position(x_curved)
    yaw_ref = road.reference_heading(x_curved)
    curvature = road.road_curvature(x_curved)

    assert np.all(np.isfinite(x_straight))
    assert np.all(np.isfinite(y_straight))
    assert np.all(np.isfinite(x_curved))
    assert np.all(np.isfinite(y_curved))
    assert np.all(np.isfinite(y_ref))
    assert np.all(np.isfinite(yaw_ref))
    assert np.all(np.isfinite(curvature))
