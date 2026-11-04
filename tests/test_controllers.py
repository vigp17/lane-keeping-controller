import sys
from pathlib import Path

import numpy as np
import pytest


# Add src directory to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare_controllers import (
    run_simulation,
    MAX_STEERING,
    MAX_STEERING_RATE,
)


@pytest.mark.parametrize(
    "controller_type",
    ["Baseline", "Augmented LQR"]
)
def test_steering_angle_limit(controller_type):

    results = run_simulation(controller_type)

    steering = results["steering"]

    assert np.all(
        np.abs(steering) <= MAX_STEERING + 1e-9
    )


@pytest.mark.parametrize(
    "controller_type",
    ["Baseline", "Augmented LQR"]
)
def test_steering_rate_limit(controller_type):

    results = run_simulation(controller_type)

    steering_rate = results["steering_rate"]

    assert np.all(
        np.abs(steering_rate) <= MAX_STEERING_RATE + 1e-9
    )


@pytest.mark.parametrize(
    "controller_type",
    ["Baseline", "Augmented LQR"]
)
def test_final_lateral_error(controller_type):

    results = run_simulation(controller_type)

    final_error = abs(results["y"][-1])

    assert final_error < 0.05


@pytest.mark.parametrize(
    "controller_type",
    ["Baseline", "Augmented LQR"]
)
def test_numerical_stability(controller_type):

    results = run_simulation(controller_type)

    for values in results.values():

        assert np.all(np.isfinite(values))