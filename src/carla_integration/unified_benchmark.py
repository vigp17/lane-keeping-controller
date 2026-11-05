"""
Lightweight Unified CARLA 0.9.16 Closed-Loop Controller Benchmark Runner.

Evaluates and compares all five CARLA lane-keeping controller integrations:
    1. Baseline (PID + Heading Feedback + Curvature Feedforward)
    2. Stanley Geometric Controller
    3. Pure Pursuit Lookahead Controller
    4. Augmented LQR (Steering-Rate Optimal Feedback + Curvature Feedforward)
    5. Constrained MPC (Receding-Horizon QP with Curvature Preview)

Capabilities:
    - Validates configuration parity (map, spawn point #61, route length ~135.24 m,
      target speed 8.0 m/s, dt = 0.05 s, +/-30 deg angle limit, +/-20 deg/s rate
      limit, longitudinal PID controller, and stopping criteria) and explicitly
      reports any structural or schema differences across the five scripts.
    - Validates initial conditions and physics-derived geometry across all
      collected telemetry CSVs without silently assuming they match.
    - Normalizes schema differences (e.g., 'steering_rate_deg_s' vs.
      'applied_steering_rate_deg_s' and 'road_curvature_1pm' vs.
      'reference_curvature_1pm') to compute unified overall and segment-level
      performance metrics.
    - Supports sequential (never simultaneous) live CARLA execution with
      inter-run world-state verification and cleanup recovery, while preserving
      all existing telemetry files.
    - Saves unified benchmark summary metrics to:
          results/unified_benchmark_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

# Ensure src directory is on sys.path when executed directly as a script
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import carla  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    carla = None  # type: ignore[assignment]

import carla_integration.augmented_lqr_carla as augmented_lqr_carla
import carla_integration.baseline_carla as baseline_carla
import carla_integration.mpc_carla as mpc_carla
import carla_integration.pure_pursuit_carla as pure_pursuit_carla
import carla_integration.stanley_carla as stanley_carla


# ==================================================
# 1. BENCHMARK CONSTANTS & CONTROLLER REGISTRY
# ==================================================

EXPECTED_MAP_NAME: str = stanley_carla.EXPECTED_MAP_NAME
EXPECTED_SPAWN_POINT_INDEX: int = stanley_carla.PREFERRED_SPAWN_POINT_INDEX
EXPECTED_ROUTE_DISTANCE_M: float = stanley_carla.ROUTE_DISTANCE_M
EXPECTED_MEASURED_ROUTE_LENGTH_M: float = 135.24
EXPECTED_WAYPOINT_SPACING_M: float = stanley_carla.WAYPOINT_SPACING_M
EXPECTED_TARGET_SPEED_MPS: float = stanley_carla.TARGET_SPEED_MPS
EXPECTED_FIXED_DT_S: float = stanley_carla.FIXED_DELTA_SECONDS
EXPECTED_MAX_STEERING_DEG: float = stanley_carla.MAX_STEERING_DEG
EXPECTED_MAX_STEERING_RATE_DEG_S: float = stanley_carla.MAX_STEERING_RATE_DEG_S
EXPECTED_ENDPOINT_TOLERANCE_M: float = stanley_carla.ENDPOINT_TOLERANCE_M
EXPECTED_MAX_ALLOWED_CTE_M: float = stanley_carla.MAX_ALLOWED_CTE_M
EXPECTED_MAX_DURATION_S: float = stanley_carla.MAX_SIMULATION_DURATION_S
EXPECTED_TESLA_WHEELBASE_M: float = 3.004645

CURVATURE_THRESHOLD_1PM: float = stanley_carla.CURVATURE_THRESHOLD_1PM
STRAIGHT_APPROACH_MAX_S_M: float = 57.50

RESULTS_DIR: Path = stanley_carla.RESULTS_DIR
SUMMARY_CSV_PATH: Path = RESULTS_DIR / "unified_benchmark_summary.csv"


@dataclass(frozen=True)
class ControllerBenchmarkSpec:
    """Specification binding a controller name to its module, runner, and telemetry CSV."""

    name: str
    slug: str
    module: Any
    runner: Callable[..., Any]
    default_csv_path: Path
    live_benchmark_prefix: str


CONTROLLER_SPECS: tuple[ControllerBenchmarkSpec, ...] = (
    ControllerBenchmarkSpec(
        name="Baseline",
        slug="baseline",
        module=baseline_carla,
        runner=baseline_carla.run_baseline_carla_test,
        default_csv_path=baseline_carla.DEFAULT_BASELINE_CSV_PATH,
        live_benchmark_prefix="benchmark_baseline_carla_8ms"
    ),
    ControllerBenchmarkSpec(
        name="Stanley",
        slug="stanley",
        module=stanley_carla,
        runner=stanley_carla.run_stanley_carla_test,
        default_csv_path=stanley_carla.DEFAULT_CSV_PATH,
        live_benchmark_prefix="benchmark_stanley_carla_extended_8ms"
    ),
    ControllerBenchmarkSpec(
        name="Pure Pursuit",
        slug="pure_pursuit",
        module=pure_pursuit_carla,
        runner=pure_pursuit_carla.run_pure_pursuit_carla_test,
        default_csv_path=pure_pursuit_carla.DEFAULT_PP_CSV_PATH,
        live_benchmark_prefix="benchmark_pure_pursuit_carla_8ms"
    ),
    ControllerBenchmarkSpec(
        name="Augmented LQR",
        slug="augmented_lqr",
        module=augmented_lqr_carla,
        runner=augmented_lqr_carla.run_augmented_lqr_carla_test,
        default_csv_path=augmented_lqr_carla.DEFAULT_LQR_CSV_PATH,
        live_benchmark_prefix="benchmark_augmented_lqr_carla_8ms"
    ),
    ControllerBenchmarkSpec(
        name="Constrained MPC",
        slug="mpc",
        module=mpc_carla,
        runner=mpc_carla.run_mpc_carla_test,
        default_csv_path=mpc_carla.DEFAULT_MPC_CSV_PATH,
        live_benchmark_prefix="benchmark_mpc_carla_8ms"
    ),
)


# ==================================================
# 2. DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class ConfigurationValidationReport:
    """Report of module-level configuration parity and documented differences."""

    all_core_parameters_match: bool
    shared_parameters: dict[str, Any]
    parameter_mismatches: list[str]
    documented_methodological_differences: list[str]


@dataclass(frozen=True)
class TelemetryConsistencyReport:
    """Report of initial-condition, timestep, route, and schema consistency across CSVs."""

    all_initial_conditions_match: bool
    reference_wheelbase_m: float
    initial_rear_x_m: float
    initial_rear_y_m: float
    initial_front_x_m: float
    initial_front_y_m: float
    initial_yaw_rad: float
    initial_speed_mps: float
    consistency_mismatches: list[str]
    schema_and_convention_differences: list[str]


@dataclass(frozen=True)
class UnifiedBenchmarkRow:
    """Unified summary row saved to results/unified_benchmark_summary.csv."""

    controller: str
    status: str
    termination_reason: str
    route_completed: bool
    collision_count: int
    total_steps: int
    simulation_duration_s: float
    route_progress_m: float
    wheelbase_m: float
    max_speed_mps: float
    mean_speed_mps: float
    final_speed_mps: float
    rear_axle_rmse_m: float
    front_cte_rmse_m: float
    max_rear_axle_error_m: float
    max_front_cte_m: float
    final_rear_axle_error_m: float
    final_front_cte_m: float
    max_heading_error_deg: float
    steering_rms_deg: float
    max_steering_deg: float
    steering_rate_rms_deg_s: float
    max_steering_rate_deg_s: float
    straight_approach_rear_rmse_m: float
    sustained_curve_rear_rmse_m: float
    straight_exit_rear_rmse_m: float
    sustained_curve_front_rmse_m: float
    angle_constraint_violations: int
    rate_constraint_violations: int
    non_finite_values: int
    steering_rate_column_used: str
    curvature_column_used: str
    telemetry_csv: str


# ==================================================
# 3. CONFIGURATION & TELEMETRY CONSISTENCY VALIDATION
# ==================================================

def validate_controller_module_configs(
    specs: Sequence[ControllerBenchmarkSpec] = CONTROLLER_SPECS
) -> ConfigurationValidationReport:
    """
    Inspect the five CARLA controller modules to verify that they share the same
    route parameters, vehicle blueprint, target speed, fixed timestep, steering
    constraints, longitudinal controller class/gains, and stopping thresholds,
    and report any differences.
    """
    mismatches: list[str] = []

    lon_default = stanley_carla.ConservativeLongitudinalController()
    expected_lon_params = (
        lon_default.target_speed_mps,
        lon_default.pid.kp,
        lon_default.pid.ki,
        lon_default.pid.kd,
        lon_default.max_throttle,
        lon_default.max_brake,
    )

    for spec in specs:
        mod = spec.module
        if getattr(mod, "VEHICLE_BLUEPRINT_ID", None) != stanley_carla.VEHICLE_BLUEPRINT_ID:
            mismatches.append(f"{spec.name}: VEHICLE_BLUEPRINT_ID mismatch.")
        if getattr(mod, "PREFERRED_SPAWN_POINT_INDEX", None) != EXPECTED_SPAWN_POINT_INDEX:
            mismatches.append(f"{spec.name}: PREFERRED_SPAWN_POINT_INDEX mismatch.")
        if not math.isclose(
            float(getattr(mod, "ROUTE_DISTANCE_M", -1.0)),
            EXPECTED_ROUTE_DISTANCE_M,
             abs_tol=1e-6
        ):
            mismatches.append(f"{spec.name}: ROUTE_DISTANCE_M mismatch.")
        if not math.isclose(
            float(getattr(mod, "WAYPOINT_SPACING_M", -1.0)),
            EXPECTED_WAYPOINT_SPACING_M,
            abs_tol=1e-6
        ):
            mismatches.append(f"{spec.name}: WAYPOINT_SPACING_M mismatch.")
        if not math.isclose(
            float(getattr(mod, "TARGET_SPEED_MPS", -1.0)),
            EXPECTED_TARGET_SPEED_MPS,
            abs_tol=1e-6
        ):
            mismatches.append(f"{spec.name}: TARGET_SPEED_MPS mismatch.")
        if not math.isclose(
            float(getattr(mod, "FIXED_DELTA_SECONDS", -1.0)),
            EXPECTED_FIXED_DT_S,
            abs_tol=1e-9
        ):
            mismatches.append(f"{spec.name}: FIXED_DELTA_SECONDS mismatch.")
        if not math.isclose(
            float(getattr(mod, "MAX_STEERING_DEG", -1.0)),
            EXPECTED_MAX_STEERING_DEG,
            abs_tol=1e-6
        ):
            mismatches.append(f"{spec.name}: MAX_STEERING_DEG mismatch.")
        if not math.isclose(
            float(getattr(mod, "MAX_STEERING_RATE_DEG_S", -1.0)),
            EXPECTED_MAX_STEERING_RATE_DEG_S,
            abs_tol=1e-6
        ):
            mismatches.append(f"{spec.name}: MAX_STEERING_RATE_DEG_S mismatch.")
        if not math.isclose(
            float(getattr(mod, "ENDPOINT_TOLERANCE_M", -1.0)),
            EXPECTED_ENDPOINT_TOLERANCE_M,
            abs_tol=1e-6
        ):
            mismatches.append(f"{spec.name}: ENDPOINT_TOLERANCE_M mismatch.")
        if not math.isclose(
            float(getattr(mod, "MAX_ALLOWED_CTE_M", -1.0)),
            EXPECTED_MAX_ALLOWED_CTE_M,
            abs_tol=1e-6
        ):
            mismatches.append(f"{spec.name}: MAX_ALLOWED_CTE_M mismatch.")
        if not math.isclose(
            float(getattr(mod, "MAX_SIMULATION_DURATION_S", -1.0)),
            EXPECTED_MAX_DURATION_S,
            abs_tol=1e-6
        ):
            mismatches.append(f"{spec.name}: MAX_SIMULATION_DURATION_S mismatch.")

        lon_cls = getattr(mod, "ConservativeLongitudinalController", None)
        if lon_cls is not stanley_carla.ConservativeLongitudinalController:
            mismatches.append(
                f"{spec.name}: Uses a different ConservativeLongitudinalController class."
            )
        else:
            inst = lon_cls()
            inst_params = (
                inst.target_speed_mps,
                inst.pid.kp,
                inst.pid.ki,
                inst.pid.kd,
                inst.max_throttle,
                inst.max_brake,
            )
            if inst_params != expected_lon_params:
                mismatches.append(f"{spec.name}: Longitudinal PID parameters differ.")

    documented_differences = [
        (
            "Steering-rate telemetry column name: Stanley and Pure Pursuit record "
            "'steering_rate_deg_s' / 'steering_rate_rad_s', whereas Baseline, "
            "Augmented LQR, and Constrained MPC record 'applied_steering_rate_deg_s' / "
            "'applied_steering_rate_rad_s'."
        ),
        (
            "Curvature telemetry column name & estimator: Stanley and Pure Pursuit "
            "record discrete finite-difference curvature in 'road_curvature_1pm', "
            "whereas Baseline, Augmented LQR, and Constrained MPC record Gaussian-"
            "smoothed arc-length curvature in 'reference_curvature_1pm'."
        ),
        (
            "Reference lookup at step 0: Stanley logs the front-axle nearest waypoint "
            "in (ref_x_m, ref_y_m, nearest_index), whereas Baseline, Pure Pursuit, "
            "Augmented LQR, and Constrained MPC log the rear-axle reference point in "
            "(ref_x_m, ref_y_m, nearest_index) and track 'front_nearest_index' separately."
        ),
        (
            "Error sign convention: Stanley and Pure Pursuit define cross-track error "
            "as (p_ref - p_actual) . n_left (negative when left of centerline), whereas "
            "Baseline, Augmented LQR, and Constrained MPC define lateral error e_y as "
            "(p_actual - p_ref) . n_left (positive when left of centerline). Unsigned "
            "magnitudes and RMSE are invariant."
        ),
        (
            "Stopping criteria detail: All five controllers enforce ENDPOINT_TOLERANCE_M = 1.5 m, "
            "MAX_ALLOWED_CTE_M = 2.5 m, MAX_SIMULATION_DURATION_S = 35.0 s, and collision_count > 0; "
            "Constrained MPC additionally halts on repeated QP solver failures "
            "(MAX_CONSECUTIVE_SOLVER_FAILURES = 5, MAX_TOTAL_SOLVER_FAILURES = 10)."
        ),
    ]

    shared = {
        "map_name": EXPECTED_MAP_NAME,
        "vehicle_blueprint": stanley_carla.VEHICLE_BLUEPRINT_ID,
        "spawn_point_index": EXPECTED_SPAWN_POINT_INDEX,
        "nominal_route_distance_m": EXPECTED_ROUTE_DISTANCE_M,
        "waypoint_spacing_m": EXPECTED_WAYPOINT_SPACING_M,
        "target_speed_mps": EXPECTED_TARGET_SPEED_MPS,
        "fixed_delta_seconds": EXPECTED_FIXED_DT_S,
        "max_steering_deg": EXPECTED_MAX_STEERING_DEG,
        "max_steering_rate_deg_s": EXPECTED_MAX_STEERING_RATE_DEG_S,
        "endpoint_tolerance_m": EXPECTED_ENDPOINT_TOLERANCE_M,
        "max_allowed_cte_m": EXPECTED_MAX_ALLOWED_CTE_M,
        "max_simulation_duration_s": EXPECTED_MAX_DURATION_S,
        "settle_ticks": stanley_carla.SETTLE_TICKS,
        "longitudinal_pid_gains": {
            "kp": lon_default.pid.kp,
            "ki": lon_default.pid.ki,
            "kd": lon_default.pid.kd,
            "max_throttle": lon_default.max_throttle,
            "max_brake": lon_default.max_brake,
        },
    }

    return ConfigurationValidationReport(
        all_core_parameters_match=(len(mismatches) == 0),
        shared_parameters=shared,
        parameter_mismatches=mismatches,
        documented_methodological_differences=documented_differences
    )


# ==================================================
# 4. TELEMETRY LOADING, SCHEMA RESOLUTION & METRICS
# ==================================================

def resolve_steering_rate_column(fieldnames: Sequence[str]) -> str:
    """
    Resolve the applied steering-rate column name (in deg/s) across different
    controller telemetry schemas.
    """
    for candidate in ("applied_steering_rate_deg_s", "steering_rate_deg_s"):
        if candidate in fieldnames:
            return candidate
    raise KeyError(
        f"No recognized steering-rate column found in CSV headers: {list(fieldnames)}"
    )


def resolve_curvature_column(fieldnames: Sequence[str]) -> str:
    """
    Resolve the road curvature column name (in m^-1) across different
    controller telemetry schemas.
    """
    for candidate in ("reference_curvature_1pm", "road_curvature_1pm"):
        if candidate in fieldnames:
            return candidate
    raise KeyError(
        f"No recognized curvature column found in CSV headers: {list(fieldnames)}"
    )


def load_controller_telemetry_csv(
    csv_path: Path
) -> tuple[list[dict[str, str]], list[str]]:
    """Load a controller telemetry CSV file and return (rows, fieldnames)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Telemetry CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise ValueError(f"Telemetry CSV file is empty: {csv_path}")

    return rows, fieldnames


def validate_telemetry_consistency(
    csv_paths_by_controller: dict[str, Path]
) -> TelemetryConsistencyReport:
    """
    Inspect the loaded telemetry CSV files for all controllers to verify that
    initial vehicle pose (rear_x_m, rear_y_m, front_x_m, front_y_m, yaw_rad),
    initial speed, physics-derived wheelbase, fixed simulation timestep, and
    route length match, while reporting schema/convention differences.
    """
    if not csv_paths_by_controller:
        raise ValueError("csv_paths_by_controller must not be empty.")

    mismatches: list[str] = []
    schema_diffs: list[str] = []

    ref_controller: str | None = None
    ref_rx = 0.0
    ref_ry = 0.0
    ref_fx = 0.0
    ref_fy = 0.0
    ref_yaw = 0.0
    ref_spd = 0.0
    ref_wheelbase = 0.0

    for ctrl_name, csv_path in csv_paths_by_controller.items():
        rows, fieldnames = load_controller_telemetry_csv(csv_path)
        first = rows[0]

        rx0 = float(first["rear_x_m"])
        ry0 = float(first["rear_y_m"])
        fx0 = float(first["front_x_m"])
        fy0 = float(first["front_y_m"])
        yaw0 = float(first["yaw_rad"])
        spd0 = float(first["speed_mps"])
        wb0 = math.hypot(fx0 - rx0, fy0 - ry0)

        rate_col = resolve_steering_rate_column(fieldnames)
        curv_col = resolve_curvature_column(fieldnames)
        ref_x0 = float(first["ref_x_m"])

        schema_diffs.append(
            f"{ctrl_name}: rate_col='{rate_col}', curvature_col='{curv_col}', "
            f"step0_ref_x={ref_x0:.4f}m"
        )

        # Check timestep uniformity
        times = np.array([float(r["sim_time_s"]) for r in rows], dtype=np.float64)
        if times.size >= 2:
            dts = np.diff(times)
            if not np.allclose(dts, EXPECTED_FIXED_DT_S, atol=1e-6):
                mismatches.append(
                    f"{ctrl_name}: Timestep dt is not uniform at {EXPECTED_FIXED_DT_S} s."
                )

        if not math.isclose(wb0, EXPECTED_TESLA_WHEELBASE_M, abs_tol=1e-3):
            mismatches.append(
                f"{ctrl_name}: Measured wheelbase {wb0:.6f} m differs from "
                f"expected live Tesla wheelbase {EXPECTED_TESLA_WHEELBASE_M:.6f} m."
            )

        if ref_controller is None:
            ref_controller = ctrl_name
            ref_rx, ref_ry = rx0, ry0
            ref_fx, ref_fy = fx0, fy0
            ref_yaw = yaw0
            ref_spd = spd0
            ref_wheelbase = wb0
        else:
            if not (
                math.isclose(rx0, ref_rx, abs_tol=1e-4)
                and math.isclose(ry0, ref_ry, abs_tol=1e-4)
                and math.isclose(fx0, ref_fx, abs_tol=1e-4)
                and math.isclose(fy0, ref_fy, abs_tol=1e-4)
                and math.isclose(yaw0, ref_yaw, abs_tol=1e-5)
                and math.isclose(spd0, ref_spd, abs_tol=1e-4)
            ):
                mismatches.append(
                    f"{ctrl_name}: Initial vehicle state at step 0 differs from {ref_controller}."
                )

    return TelemetryConsistencyReport(
        all_initial_conditions_match=(len(mismatches) == 0),
        reference_wheelbase_m=ref_wheelbase,
        initial_rear_x_m=ref_rx,
        initial_rear_y_m=ref_ry,
        initial_front_x_m=ref_fx,
        initial_front_y_m=ref_fy,
        initial_yaw_rad=ref_yaw,
        initial_speed_mps=ref_spd,
        consistency_mismatches=mismatches,
        schema_and_convention_differences=schema_diffs
    )


def calculate_benchmark_metrics_from_csv(
    controller_name: str,
    csv_path: Path,
    collision_count: int = 0,
    status: str = "completed",
    termination_reason: str = "route_endpoint_reached",
    route_completed_override: bool | None = None
) -> UnifiedBenchmarkRow:
    """
    Calculate unified tracking, steering, constraint, and segment metrics from
    a controller's telemetry CSV file, handling schema column variations.
    """
    rows, fieldnames = load_controller_telemetry_csv(csv_path)

    rate_col = resolve_steering_rate_column(fieldnames)
    curv_col = resolve_curvature_column(fieldnames)

    sim_time_s = np.array([float(r["sim_time_s"]) for r in rows], dtype=np.float64)
    speed_mps = np.array([float(r["speed_mps"]) for r in rows], dtype=np.float64)
    rear_x = np.array([float(r["rear_x_m"]) for r in rows], dtype=np.float64)
    rear_y = np.array([float(r["rear_y_m"]) for r in rows], dtype=np.float64)
    front_x = np.array([float(r["front_x_m"]) for r in rows], dtype=np.float64)
    front_y = np.array([float(r["front_y_m"]) for r in rows], dtype=np.float64)
    rear_err = np.array(
        [float(r["rear_axle_tracking_error_m"]) for r in rows],
        dtype=np.float64
    )
    front_cte = np.array(
        [float(r["front_cross_track_error_m"]) for r in rows],
        dtype=np.float64
    )
    heading_deg = np.array(
        [float(r["heading_error_deg"]) for r in rows],
        dtype=np.float64
    )
    app_steer_deg = np.array(
        [float(r["applied_steering_deg"]) for r in rows],
        dtype=np.float64
    )
    steer_rate_deg_s = np.array(
        [float(r[rate_col]) for r in rows],
        dtype=np.float64
    )
    curvature_1pm = np.array(
        [float(r[curv_col]) for r in rows],
        dtype=np.float64
    )
    progress_m = (
        np.array([float(r["route_progress_m"]) for r in rows], dtype=np.float64)
        if "route_progress_m" in fieldnames
        else np.zeros(len(rows), dtype=np.float64)
    )
    progress_pct = (
        np.array([float(r["route_progress_pct"]) for r in rows], dtype=np.float64)
        if "route_progress_pct" in fieldnames
        else np.zeros(len(rows), dtype=np.float64)
    )

    wheelbase_m = float(math.hypot(front_x[0] - rear_x[0], front_y[0] - rear_y[0]))

    # Check non-finite values across all numeric arrays
    numeric_arrays = (
        sim_time_s,
        speed_mps,
        rear_x,
        rear_y,
        front_x,
        front_y,
        rear_err,
        front_cte,
        heading_deg,
        app_steer_deg,
        steer_rate_deg_s,
        curvature_1pm,
        progress_m,
    )
    non_finite_count = int(
        sum(int(np.sum(~np.isfinite(arr))) for arr in numeric_arrays)
    )

    angle_violations = int(
        np.sum(np.abs(app_steer_deg) > EXPECTED_MAX_STEERING_DEG + 1e-5)
    )
    rate_violations = int(
        np.sum(np.abs(steer_rate_deg_s) > EXPECTED_MAX_STEERING_RATE_DEG_S + 1e-5)
    )

    # Segment masks matching the 3-segment route breakdown
    approach_mask = (
        (np.abs(curvature_1pm) < CURVATURE_THRESHOLD_1PM)
        & (progress_m <= STRAIGHT_APPROACH_MAX_S_M + 0.5)
    )
    curved_mask = np.abs(curvature_1pm) >= CURVATURE_THRESHOLD_1PM
    exit_mask = (
        (np.abs(curvature_1pm) < CURVATURE_THRESHOLD_1PM)
        & (progress_m > STRAIGHT_APPROACH_MAX_S_M + 0.5)
    )

    def _rmse(arr: np.ndarray, mask: np.ndarray | None = None) -> float:
        sub = arr if mask is None else arr[mask]
        if sub.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(sub ** 2)))

    inferred_completed = bool(
        collision_count == 0
        and non_finite_count == 0
        and (
            float(progress_pct[-1]) >= 97.5
            or float(progress_m[-1]) >= EXPECTED_MEASURED_ROUTE_LENGTH_M - 2.5
        )
    )
    route_completed = (
        inferred_completed
        if route_completed_override is None
        else bool(route_completed_override)
    )

    rel_csv = (
        str(csv_path.relative_to(RESULTS_DIR.parent)).replace("\\", "/")
        if csv_path.is_relative_to(RESULTS_DIR.parent)
        else str(csv_path)
    )

    return UnifiedBenchmarkRow(
        controller=controller_name,
        status=status,
        termination_reason=termination_reason,
        route_completed=route_completed,
        collision_count=int(collision_count),
        total_steps=len(rows),
        simulation_duration_s=float(sim_time_s[-1]),
        route_progress_m=float(progress_m[-1]),
        wheelbase_m=wheelbase_m,
        max_speed_mps=float(np.max(speed_mps)),
        mean_speed_mps=float(np.mean(speed_mps)),
        final_speed_mps=float(speed_mps[-1]),
        rear_axle_rmse_m=_rmse(rear_err),
        front_cte_rmse_m=_rmse(front_cte),
        max_rear_axle_error_m=float(np.max(np.abs(rear_err))),
        max_front_cte_m=float(np.max(np.abs(front_cte))),
        final_rear_axle_error_m=float(abs(rear_err[-1])),
        final_front_cte_m=float(abs(front_cte[-1])),
        max_heading_error_deg=float(np.max(np.abs(heading_deg))),
        steering_rms_deg=_rmse(app_steer_deg),
        max_steering_deg=float(np.max(np.abs(app_steer_deg))),
        steering_rate_rms_deg_s=_rmse(steer_rate_deg_s),
        max_steering_rate_deg_s=float(np.max(np.abs(steer_rate_deg_s))),
        straight_approach_rear_rmse_m=_rmse(rear_err, approach_mask),
        sustained_curve_rear_rmse_m=_rmse(rear_err, curved_mask),
        straight_exit_rear_rmse_m=_rmse(rear_err, exit_mask),
        sustained_curve_front_rmse_m=_rmse(front_cte, curved_mask),
        angle_constraint_violations=angle_violations,
        rate_constraint_violations=rate_violations,
        non_finite_values=non_finite_count,
        steering_rate_column_used=rate_col,
        curvature_column_used=curv_col,
        telemetry_csv=rel_csv
    )


def create_failed_benchmark_row(
    controller_name: str,
    error_message: str,
    csv_path: Path
) -> UnifiedBenchmarkRow:
    """Create a failure record when a controller run raises an exception."""
    rel_csv = (
        str(csv_path.relative_to(RESULTS_DIR.parent)).replace("\\", "/")
        if csv_path.is_relative_to(RESULTS_DIR.parent)
        else str(csv_path)
    )
    return UnifiedBenchmarkRow(
        controller=controller_name,
        status="failed",
        termination_reason=f"error: {error_message}",
        route_completed=False,
        collision_count=0,
        total_steps=0,
        simulation_duration_s=0.0,
        route_progress_m=0.0,
        wheelbase_m=0.0,
        max_speed_mps=0.0,
        mean_speed_mps=0.0,
        final_speed_mps=0.0,
        rear_axle_rmse_m=float("nan"),
        front_cte_rmse_m=float("nan"),
        max_rear_axle_error_m=float("nan"),
        max_front_cte_m=float("nan"),
        final_rear_axle_error_m=float("nan"),
        final_front_cte_m=float("nan"),
        max_heading_error_deg=float("nan"),
        steering_rms_deg=float("nan"),
        max_steering_deg=float("nan"),
        steering_rate_rms_deg_s=float("nan"),
        max_steering_rate_deg_s=float("nan"),
        straight_approach_rear_rmse_m=float("nan"),
        sustained_curve_rear_rmse_m=float("nan"),
        straight_exit_rear_rmse_m=float("nan"),
        sustained_curve_front_rmse_m=float("nan"),
        angle_constraint_violations=0,
        rate_constraint_violations=0,
        non_finite_values=0,
        steering_rate_column_used="N/A",
        curvature_column_used="N/A",
        telemetry_csv=rel_csv
    )


def save_unified_benchmark_summary(
    rows: Sequence[UnifiedBenchmarkRow],
    summary_csv_path: Path = SUMMARY_CSV_PATH
) -> Path:
    """Save the unified benchmark summary table to results/unified_benchmark_summary.csv."""
    if not rows:
        raise ValueError("Cannot save an empty benchmark summary.")

    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())

    with summary_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    return summary_csv_path


# ==================================================
# 5. SEQUENTIAL LIVE CARLA BENCHMARK & STATE RECOVERY
# ==================================================

def verify_and_recover_carla_state(
    host: str = stanley_carla.CARLA_HOST,
    port: int = stanley_carla.CARLA_PORT,
    timeout_s: float = 10.0
) -> tuple[bool, str]:
    """
    Verify that the CARLA server is responsive, expected map is loaded,
    any leftover vehicles/collision sensors are cleaned up, and synchronous
    mode is reset before starting the next sequential controller run.
    """
    if carla is None:
        return False, "CARLA Python API (carla) is not installed."

    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout_s)
        world = client.get_world()
        map_name = str(world.get_map().name)

        if EXPECTED_MAP_NAME not in map_name:
            return False, f"Unexpected CARLA map '{map_name}' (expected '{EXPECTED_MAP_NAME}')."

        actors = world.get_actors()
        leftover_sensors = list(actors.filter("sensor.*"))
        leftover_vehicles = list(actors.filter("vehicle.*"))

        for sensor in leftover_sensors:
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except Exception:
                pass

        for vehicle in leftover_vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except Exception:
                pass

        settings = world.get_settings()
        if settings.synchronous_mode:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        # Verify no remaining vehicles block the spawn point
        remaining_vehicles = list(world.get_actors().filter("vehicle.*"))
        if remaining_vehicles:
            return False, f"Failed to clean up {len(remaining_vehicles)} leftover vehicle actor(s)."

        return True, "CARLA world state verified and clean."

    except Exception as exc:
        return False, f"CARLA state verification failed: {exc}"


def run_sequential_live_benchmark(
    specs: Sequence[ControllerBenchmarkSpec] = CONTROLLER_SPECS,
    host: str = stanley_carla.CARLA_HOST,
    port: int = stanley_carla.CARLA_PORT,
    results_dir: Path = RESULTS_DIR,
    summary_csv_path: Path = SUMMARY_CSV_PATH
) -> list[UnifiedBenchmarkRow]:
    """
    Run all five CARLA controllers sequentially (never simultaneously).
    Preserves existing telemetry files by writing live benchmark runs to
    dedicated 'benchmark_<controller>_8ms' prefixes unless a file does not
    yet exist. If a controller fails, records the failure and continues
    only if verify_and_recover_carla_state() confirms CARLA is clean.
    """
    rows: list[UnifiedBenchmarkRow] = []

    for spec in specs:
        carla_ok, state_msg = verify_and_recover_carla_state(host=host, port=port)
        if not carla_ok:
            rows.append(
                create_failed_benchmark_row(
                    controller_name=spec.name,
                    error_message=f"Aborted before run ({state_msg})",
                    csv_path=spec.default_csv_path
                )
            )
            break

        # Never overwrite an existing telemetry CSV file
        target_default_csv = results_dir / spec.default_csv_path.name
        file_prefix = (
            spec.live_benchmark_prefix
            if target_default_csv.exists()
            else spec.default_csv_path.stem.replace("_telemetry", "")
        )

        try:
            summary = spec.runner(
                host=host,
                port=port,
                target_speed_mps=EXPECTED_TARGET_SPEED_MPS,
                fixed_delta_seconds=EXPECTED_FIXED_DT_S,
                route_distance_m=EXPECTED_ROUTE_DISTANCE_M,
                waypoint_spacing_m=EXPECTED_WAYPOINT_SPACING_M,
                results_dir=results_dir,
                file_prefix=file_prefix
            )
            row = calculate_benchmark_metrics_from_csv(
                controller_name=spec.name,
                csv_path=summary.csv_path,
                collision_count=summary.collision_count,
                status="completed" if summary.route_completed else "incomplete",
                termination_reason=summary.termination_reason,
                route_completed_override=summary.route_completed
            )
            rows.append(row)
        except Exception as exc:
            expected_csv = results_dir / f"{file_prefix}_telemetry.csv"
            rows.append(
                create_failed_benchmark_row(
                    controller_name=spec.name,
                    error_message=str(exc),
                    csv_path=expected_csv
                )
            )
            # Verify CARLA cleanup completed before allowing the next controller to run
            post_ok, _ = verify_and_recover_carla_state(host=host, port=port)
            if not post_ok:
                break

    if rows:
        save_unified_benchmark_summary(rows=rows, summary_csv_path=summary_csv_path)

    return rows


# ==================================================
# 6. UNIFIED BENCHMARK EVALUATION & REPORTING
# ==================================================

def evaluate_existing_telemetry_benchmark(
    specs: Sequence[ControllerBenchmarkSpec] = CONTROLLER_SPECS,
    results_dir: Path = RESULTS_DIR,
    summary_csv_path: Path = SUMMARY_CSV_PATH
) -> tuple[
    ConfigurationValidationReport,
    TelemetryConsistencyReport,
    list[UnifiedBenchmarkRow]
]:
    """
    Validate all five controller configurations and their existing measured
    CARLA telemetry CSV files, calculate unified performance metrics, and
    save results/unified_benchmark_summary.csv without overwriting any
    existing telemetry files.
    """
    config_report = validate_controller_module_configs(specs=specs)

    csv_paths_by_controller: dict[str, Path] = {}
    for spec in specs:
        candidate_path = results_dir / spec.default_csv_path.name
        csv_paths_by_controller[spec.name] = candidate_path

    consistency_report = validate_telemetry_consistency(csv_paths_by_controller)

    rows: list[UnifiedBenchmarkRow] = []
    for spec in specs:
        csv_path = csv_paths_by_controller[spec.name]
        row = calculate_benchmark_metrics_from_csv(
            controller_name=spec.name,
            csv_path=csv_path,
            collision_count=0,
            status="completed",
            termination_reason="route_endpoint_reached"
        )
        rows.append(row)

    save_unified_benchmark_summary(rows=rows, summary_csv_path=summary_csv_path)
    return config_report, consistency_report, rows


def print_benchmark_report(
    config_report: ConfigurationValidationReport,
    consistency_report: TelemetryConsistencyReport | None,
    rows: Sequence[UnifiedBenchmarkRow],
    summary_csv_path: Path
) -> None:
    """Print a structured engineering report of the unified CARLA benchmark."""
    print("=" * 112)
    print("UNIFIED CARLA 0.9.16 CLOSED-LOOP LANE-KEEPING BENCHMARK REPORT (v = 8.0 m/s, Town10HD_Opt)")
    print("=" * 112)

    print("\n1. CONFIGURATION & INITIAL-CONDITION VALIDATION")
    print("-" * 112)
    print(f"   Core Module Parameters Match : {config_report.all_core_parameters_match}")
    if config_report.parameter_mismatches:
        for mismatch in config_report.parameter_mismatches:
            print(f"   [MISMATCH] {mismatch}")

    if consistency_report is not None:
        print(
            f"   Telemetry Initial Pose Match : {consistency_report.all_initial_conditions_match} "
            f"(Rear = ({consistency_report.initial_rear_x_m:.4f}, {consistency_report.initial_rear_y_m:.4f}) m, "
            f"Front = ({consistency_report.initial_front_x_m:.4f}, {consistency_report.initial_front_y_m:.4f}) m, "
            f"Wheelbase = {consistency_report.reference_wheelbase_m:.4f} m)"
        )
        if consistency_report.consistency_mismatches:
            for mismatch in consistency_report.consistency_mismatches:
                print(f"   [TELEMETRY MISMATCH] {mismatch}")

    print("\n2. DOCUMENTED SCHEMA & METHODOLOGICAL DIFFERENCES ACROSS CONTROLLERS")
    print("-" * 112)
    for idx, diff in enumerate(config_report.documented_methodological_differences, start=1):
        print(f"   {idx}. {diff}")

    print("\n3. UNIFIED PERFORMANCE METRICS (OVERALL 135.24 m EXTENDED ROUTE)")
    print("-" * 112)
    print(
        f"{'Controller':<16} | {'Done':<5} | {'Coll':>4} | {'Rear RMSE':>9} | "
        f"{'Front RMSE':>10} | {'Max |Rear|':>10} | {'Max |Front|':>11} | "
        f"{'Steer RMS':>9} | {'Rate RMS':>8} | {'Max Rate':>8}"
    )
    print("-" * 112)
    for r in rows:
        print(
            f"{r.controller:<16} | {str(r.route_completed):<5} | {r.collision_count:4d} | "
            f"{r.rear_axle_rmse_m:9.4f} | {r.front_cte_rmse_m:10.4f} | "
            f"{r.max_rear_axle_error_m:10.4f} | {r.max_front_cte_m:11.4f} | "
            f"{r.steering_rms_deg:9.4f} | {r.steering_rate_rms_deg_s:8.4f} | "
            f"{r.max_steering_rate_deg_s:8.4f}"
        )

    print("\n4. SEGMENT-BY-SEGMENT REAR-AXLE TRACKING RMSE (m)")
    print("-" * 112)
    print(
        f"{'Controller':<16} | {'Straight Approach (57.5m)':>25} | "
        f"{'Sustained Curve (60.2m)':>23} | {'Straight Exit (17.5m)':>21} | "
        f"{'Overall Route (135.2m)':>22}"
    )
    print("-" * 112)
    for r in rows:
        print(
            f"{r.controller:<16} | {r.straight_approach_rear_rmse_m:25.4f} | "
            f"{r.sustained_curve_rear_rmse_m:23.4f} | "
            f"{r.straight_exit_rear_rmse_m:21.4f} | "
            f"{r.rear_axle_rmse_m:22.4f}"
        )
    print("-" * 112)
    print(f"Unified Summary CSV Saved: {summary_csv_path}")
    print("=" * 112)


def main(argv: Sequence[str] | None = None) -> None:
    """
    CLI entry point. By default evaluates existing measured telemetry CSVs
    without launching CARLA or overwriting existing telemetry files; pass
    --live to execute all five controllers sequentially in CARLA.
    """
    parser = argparse.ArgumentParser(
        description="Unified CARLA 0.9.16 Closed-Loop Lane-Keeping Benchmark Runner."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run all 5 controllers sequentially in a live CARLA server."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.live:
        config_report = validate_controller_module_configs()
        rows = run_sequential_live_benchmark()
        print_benchmark_report(
            config_report=config_report,
            consistency_report=None,
            rows=rows,
            summary_csv_path=SUMMARY_CSV_PATH
        )
    else:
        config_report, consistency_report, rows = (
            evaluate_existing_telemetry_benchmark()
        )
        print_benchmark_report(
            config_report=config_report,
            consistency_report=consistency_report,
            rows=rows,
            summary_csv_path=SUMMARY_CSV_PATH
        )


if __name__ == "__main__":
    main()
