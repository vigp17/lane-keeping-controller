from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src directory to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from carla_integration.unified_benchmark import (
    CONTROLLER_SPECS,
    RESULTS_DIR,
    SUMMARY_CSV_PATH,
    calculate_benchmark_metrics_from_csv,
    evaluate_existing_telemetry_benchmark,
    resolve_curvature_column,
    resolve_steering_rate_column,
    run_sequential_live_benchmark,
    validate_controller_module_configs,
    validate_telemetry_consistency,
    verify_and_recover_carla_state,
)


# ==================================================
# 1. MODULE CONFIGURATION VALIDATION TESTS
# ==================================================

def test_validate_controller_module_configs() -> None:
    """
    Verify all five CARLA controller modules share identical core route,
    vehicle, timestep, speed, constraint, longitudinal PID, and stopping
    parameters, and that any mismatch is explicitly reported.
    """
    report = validate_controller_module_configs()
    assert report.all_core_parameters_match is True
    assert report.parameter_mismatches == []
    assert len(report.documented_methodological_differences) >= 4

    # Simulate a mismatched target speed in one module and verify detection
    fake_mod = MagicMock()
    fake_mod.VEHICLE_BLUEPRINT_ID = "vehicle.tesla.model3"
    fake_mod.PREFERRED_SPAWN_POINT_INDEX = 61
    fake_mod.ROUTE_DISTANCE_M = 138.0
    fake_mod.WAYPOINT_SPACING_M = 0.5
    fake_mod.TARGET_SPEED_MPS = 12.0  # mismatch!
    fake_mod.FIXED_DELTA_SECONDS = 0.05
    fake_mod.MAX_STEERING_DEG = 30.0
    fake_mod.MAX_STEERING_RATE_DEG_S = 20.0
    fake_mod.ENDPOINT_TOLERANCE_M = 1.5
    fake_mod.MAX_ALLOWED_CTE_M = 2.5
    fake_mod.MAX_SIMULATION_DURATION_S = 35.0
    fake_mod.ConservativeLongitudinalController = (
        CONTROLLER_SPECS[0].module.ConservativeLongitudinalController
    )

    mismatched_spec = replace(CONTROLLER_SPECS[0], name="MismatchedCtrl", module=fake_mod)
    mismatch_report = validate_controller_module_configs([mismatched_spec])
    assert mismatch_report.all_core_parameters_match is False
    assert any("TARGET_SPEED_MPS" in msg for msg in mismatch_report.parameter_mismatches)


# ==================================================
# 2. STEERING-RATE & CURVATURE COLUMN RESOLUTION TESTS
# ==================================================

def test_schema_column_resolution() -> None:
    """
    Verify both steering-rate column naming conventions ('steering_rate_deg_s'
    and 'applied_steering_rate_deg_s') and both curvature column conventions
    ('road_curvature_1pm' and 'reference_curvature_1pm') are resolved properly.
    """
    stanley_headers = ["step", "sim_time_s", "steering_rate_deg_s", "road_curvature_1pm"]
    mpc_headers = ["step", "sim_time_s", "applied_steering_rate_deg_s", "reference_curvature_1pm"]

    assert resolve_steering_rate_column(stanley_headers) == "steering_rate_deg_s"
    assert resolve_steering_rate_column(mpc_headers) == "applied_steering_rate_deg_s"

    assert resolve_curvature_column(stanley_headers) == "road_curvature_1pm"
    assert resolve_curvature_column(mpc_headers) == "reference_curvature_1pm"

    with pytest.raises(KeyError, match="steering-rate"):
        resolve_steering_rate_column(["step", "sim_time_s"])

    with pytest.raises(KeyError, match="curvature"):
        resolve_curvature_column(["step", "sim_time_s"])


# ==================================================
# 3. METRIC CALCULATION & SCHEMA VARIATION TESTS
# ==================================================

def test_metric_calculation_across_schema_variants(tmp_path: Path) -> None:
    """
    Verify calculate_benchmark_metrics_from_csv computes exact RMSE, peak
    error, steering RMS, steering-rate RMS, and max steering rate across both
    'steering_rate_deg_s' and 'applied_steering_rate_deg_s' CSV schemas.
    """
    rear_errors = [0.10, -0.20, 0.30, -0.40]
    front_errors = [0.20, -0.20, 0.40, -0.40]
    steers_deg = [2.0, -4.0, 6.0, -8.0]
    rates_deg_s = [5.0, -10.0, 15.0, -12.0]
    curvatures = [0.0, 0.0, -0.02655, -0.02655]
    progress_m = [10.0, 30.0, 80.0, 134.5]
    progress_pct = [7.4, 22.2, 59.2, 99.5]

    for rate_col, curv_col in [
        ("steering_rate_deg_s", "road_curvature_1pm"),
        ("applied_steering_rate_deg_s", "reference_curvature_1pm"),
    ]:
        csv_file = tmp_path / f"test_{rate_col}.csv"
        fieldnames = [
            "step",
            "sim_time_s",
            "speed_mps",
            "rear_x_m",
            "rear_y_m",
            "front_x_m",
            "front_y_m",
            "yaw_rad",
            "ref_x_m",
            "ref_y_m",
            "ref_yaw_rad",
            "rear_axle_tracking_error_m",
            "front_cross_track_error_m",
            "heading_error_deg",
            "applied_steering_deg",
            rate_col,
            curv_col,
            "route_progress_m",
            "route_progress_pct",
        ]
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(4):
                writer.writerow(
                    {
                        "step": idx,
                        "sim_time_s": idx * 0.05,
                        "speed_mps": 7.85,
                        "rear_x_m": 6.024709,
                        "rear_y_m": 60.899434,
                        "front_x_m": 6.024709 + 3.004645,
                        "front_y_m": 60.899434,
                        "yaw_rad": 0.0,
                        "ref_x_m": 6.024709,
                        "ref_y_m": 60.899434,
                        "ref_yaw_rad": 0.0,
                        "rear_axle_tracking_error_m": rear_errors[idx],
                        "front_cross_track_error_m": front_errors[idx],
                        "heading_error_deg": 1.5,
                        "applied_steering_deg": steers_deg[idx],
                        rate_col: rates_deg_s[idx],
                        curv_col: curvatures[idx],
                        "route_progress_m": progress_m[idx],
                        "route_progress_pct": progress_pct[idx],
                    }
                )

        row = calculate_benchmark_metrics_from_csv(
            controller_name="Synthetic",
            csv_path=csv_file
        )
        expected_rear_rmse = float(np.sqrt(np.mean(np.array(rear_errors) ** 2)))
        expected_front_rmse = float(np.sqrt(np.mean(np.array(front_errors) ** 2)))
        expected_steer_rms = float(np.sqrt(np.mean(np.array(steers_deg) ** 2)))
        expected_rate_rms = float(np.sqrt(np.mean(np.array(rates_deg_s) ** 2)))

        assert row.route_completed is True
        assert row.steering_rate_column_used == rate_col
        assert row.curvature_column_used == curv_col
        assert np.isclose(row.rear_axle_rmse_m, expected_rear_rmse, atol=1e-12)
        assert np.isclose(row.front_cte_rmse_m, expected_front_rmse, atol=1e-12)
        assert np.isclose(row.max_rear_axle_error_m, 0.40, atol=1e-12)
        assert np.isclose(row.max_front_cte_m, 0.40, atol=1e-12)
        assert np.isclose(row.steering_rms_deg, expected_steer_rms, atol=1e-12)
        assert np.isclose(row.steering_rate_rms_deg_s, expected_rate_rms, atol=1e-12)
        assert np.isclose(row.max_steering_rate_deg_s, 15.0, atol=1e-12)
        assert row.angle_constraint_violations == 0
        assert row.rate_constraint_violations == 0
        assert row.non_finite_values == 0


# ==================================================
# 4. MEASURED TELEMETRY CONSISTENCY & SUMMARY EXPORT TESTS
# ==================================================

def test_measured_telemetry_consistency_and_summary_export(tmp_path: Path) -> None:
    """
    Verify that the 5 existing measured CARLA telemetry CSVs pass initial-state,
    wheelbase, and timestep consistency checks, and that
    evaluate_existing_telemetry_benchmark writes unified_benchmark_summary.csv
    without modifying any existing telemetry files.
    """
    before_mtimes = {
        spec.name: (RESULTS_DIR / spec.default_csv_path.name).stat().st_mtime_ns
        for spec in CONTROLLER_SPECS
    }

    temp_summary = tmp_path / "unified_benchmark_summary.csv"
    config_report, consistency_report, rows = evaluate_existing_telemetry_benchmark(
        summary_csv_path=temp_summary
    )

    after_mtimes = {
        spec.name: (RESULTS_DIR / spec.default_csv_path.name).stat().st_mtime_ns
        for spec in CONTROLLER_SPECS
    }
    assert before_mtimes == after_mtimes, "Existing telemetry CSV files must not be modified!"
    assert config_report.all_core_parameters_match is True
    assert consistency_report.all_initial_conditions_match is True
    assert np.isclose(consistency_report.reference_wheelbase_m, 3.004645, atol=1e-4)
    assert len(rows) == 5
    assert temp_summary.exists()
    assert SUMMARY_CSV_PATH.exists()

    for r in rows:
        assert r.route_completed is True
        assert r.collision_count == 0
        assert r.angle_constraint_violations == 0
        assert r.rate_constraint_violations == 0
        assert r.non_finite_values == 0


def test_telemetry_consistency_detects_initial_pose_mismatch(tmp_path: Path) -> None:
    """Verify validate_telemetry_consistency flags initial condition differences."""
    base_csv = RESULTS_DIR / CONTROLLER_SPECS[0].default_csv_path.name
    with base_csv.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Perturb rear_x_m in row 0 of a copy
    header = lines[0].strip().split(",")
    row0 = lines[1].strip().split(",")
    rx_idx = header.index("rear_x_m")
    row0[rx_idx] = "99.999"

    perturbed_csv = tmp_path / "perturbed.csv"
    with perturbed_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(lines[0])
        f.write(",".join(row0) + "\n")
        f.writelines(lines[2:10])

    report = validate_telemetry_consistency(
        {
            "Baseline": base_csv,
            "Perturbed": perturbed_csv,
        }
    )
    assert report.all_initial_conditions_match is False
    assert any("Perturbed" in msg for msg in report.consistency_mismatches)


# ==================================================
# 5. SEQUENTIAL EXECUTION & FAILURE RECOVERY TESTS
# ==================================================

def test_sequential_execution_and_failure_recovery(tmp_path: Path) -> None:
    """
    Verify run_sequential_live_benchmark:
        1. Runs controllers strictly sequentially.
        2. Records a failed controller and continues to the next controller
           when verify_and_recover_carla_state confirms CARLA is clean.
        3. Halts further execution if CARLA state recovery fails.
    """
    call_order: list[str] = []
    valid_csv = RESULTS_DIR / CONTROLLER_SPECS[0].default_csv_path.name

    def runner_ok(**_kwargs: Any) -> Any:
        call_order.append("ok_runner")
        mock_summary = MagicMock()
        mock_summary.csv_path = valid_csv
        mock_summary.collision_count = 0
        mock_summary.route_completed = True
        mock_summary.termination_reason = "route_endpoint_reached"
        return mock_summary

    def runner_fail(**_kwargs: Any) -> Any:
        call_order.append("fail_runner")
        raise RuntimeError("Simulated controller failure")

    specs = [
        replace(CONTROLLER_SPECS[0], name="Ctrl1_Fail", runner=runner_fail),
        replace(CONTROLLER_SPECS[1], name="Ctrl2_OK", runner=runner_ok),
    ]

    with patch(
        "carla_integration.unified_benchmark.verify_and_recover_carla_state",
        return_value=(True, "CARLA world state verified and clean.")
    ):
        summary_out = tmp_path / "summary_seq.csv"
        rows = run_sequential_live_benchmark(
            specs=specs,
            results_dir=tmp_path,
            summary_csv_path=summary_out
        )

    assert call_order == ["fail_runner", "ok_runner"]
    assert len(rows) == 2
    assert rows[0].controller == "Ctrl1_Fail"
    assert rows[0].status == "failed"
    assert rows[0].route_completed is False
    assert rows[1].controller == "Ctrl2_OK"
    assert rows[1].status == "completed"
    assert rows[1].route_completed is True
    assert summary_out.exists()

    # Now test that if post-failure CARLA recovery returns False, execution stops
    call_order_abort: list[str] = []

    def runner_fail_2(**_kwargs: Any) -> Any:
        call_order_abort.append("fail_runner_2")
        raise RuntimeError("Fatal CARLA disconnect")

    def runner_should_not_run(**_kwargs: Any) -> Any:
        call_order_abort.append("should_not_run")
        return MagicMock()

    specs_abort = [
        replace(CONTROLLER_SPECS[0], name="Ctrl1_Disconnect", runner=runner_fail_2),
        replace(CONTROLLER_SPECS[1], name="Ctrl2_Skipped", runner=runner_should_not_run),
    ]

    with patch(
        "carla_integration.unified_benchmark.verify_and_recover_carla_state",
        side_effect=[
            (True, "Initial state OK"),
            (False, "CARLA server unreachable after crash"),
        ]
    ):
        rows_abort = run_sequential_live_benchmark(
            specs=specs_abort,
            results_dir=tmp_path,
            summary_csv_path=tmp_path / "summary_abort.csv"
        )

    assert call_order_abort == ["fail_runner_2"]
    assert len(rows_abort) == 1
    assert rows_abort[0].status == "failed"


def test_verify_and_recover_carla_state_cleans_leftover_actors() -> None:
    """
    Verify verify_and_recover_carla_state destroys leftover vehicle/sensor
    actors and resets synchronous mode.
    """
    mock_sensor = MagicMock()
    mock_sensor.is_alive = True
    mock_vehicle = MagicMock()
    mock_vehicle.is_alive = True

    first_actors = MagicMock()
    first_actors.filter.side_effect = lambda pattern: (
        [mock_sensor] if pattern == "sensor.*" else [mock_vehicle]
    )
    empty_actors = MagicMock()
    empty_actors.filter.return_value = []

    mock_settings = MagicMock()
    mock_settings.synchronous_mode = True

    mock_world = MagicMock()
    mock_world.get_map.return_value.name = "Carla/Maps/Town10HD_Opt"
    mock_world.get_actors.side_effect = [first_actors, empty_actors]
    mock_world.get_settings.return_value = mock_settings

    mock_client = MagicMock()
    mock_client.get_world.return_value = mock_world

    with patch("carla_integration.unified_benchmark.carla") as mock_carla_mod:
        mock_carla_mod.Client.return_value = mock_client
        ok, msg = verify_and_recover_carla_state()

    assert ok is True
    assert "verified" in msg
    mock_sensor.stop.assert_called_once()
    mock_sensor.destroy.assert_called_once()
    mock_vehicle.destroy.assert_called_once()
    mock_world.apply_settings.assert_called_once_with(mock_settings)
    assert mock_settings.synchronous_mode is False

