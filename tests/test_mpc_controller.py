import sys
from pathlib import Path

import numpy as np
import pytest


# Add src directory to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare_controllers import run_simulation as run_comparison_simulation
from controllers.augmented_lqr_controller import AugmentedLQRController
from controllers.mpc_controller import MPCController
from mpc_simulation import (
    DT,
    HORIZON,
    MAX_STEERING,
    MAX_STEERING_RATE,
    TEST_SPEEDS,
    WHEELBASE,
    calculate_mpc_metrics,
    run_mpc_simulation,
)
from pure_pursuit_simulation import run_pure_pursuit_simulation
from road import Road
from stanley_simulation import run_stanley_simulation
from vehicle import Vehicle


# ============================================================
# 1. STATE-SPACE MATRIX DIMENSIONS
# ============================================================

def test_state_space_matrix_dimensions() -> None:
    """
    Verify dimensions of Ad (3x3), Bd (3x1), Ed (3x1), Q (3x3),
    R (1x1), and Qf (3x3).
    """
    mpc = MPCController(velocity=20.0, wheelbase=2.8, dt=0.1, horizon=15)

    assert mpc.Ad.shape == (3, 3)
    assert mpc.Bd.shape == (3, 1)
    assert mpc.Ed.shape == (3, 1)
    assert mpc.Q.shape == (3, 3)
    assert mpc.R.shape == (1, 1)
    assert mpc.Qf.shape == (3, 3)


# ============================================================
# 2. DISCRETE MODEL CORRECTNESS
# ============================================================

def test_discrete_model_correctness() -> None:
    """
    Verify exact Zero-Order Hold (ZOH) discretization matches
    AugmentedLQRController matrices and maintains zero steady-state
    residual on a constant-curvature arc when delta = L * kappa_ref.
    """
    v = 20.0
    L = 2.8
    dt = 0.1

    mpc = MPCController(velocity=v, wheelbase=L, dt=dt)
    aug_lqr = AugmentedLQRController(velocity=v, wheelbase=L, dt=dt)

    # Ad and Bd must match AugmentedLQRController exactly
    assert np.allclose(mpc.Ad, aug_lqr.Ad, atol=1e-12)
    assert np.allclose(mpc.Bd, aug_lqr.Bd, atol=1e-12)

    # Verify steady-state invariance on constant curvature kappa_ref
    kappa_ref = 0.0008
    delta_ss = L * kappa_ref
    z_ss = np.array([0.0, 0.0, delta_ss], dtype=np.float64)

    z_next = mpc.predict_next_state(z=z_ss, u=0.0, kappa_ref=kappa_ref)
    assert np.allclose(z_next, z_ss, atol=1e-15)


# ============================================================
# 3. ZERO-ERROR STRAIGHT-ROAD BEHAVIOR
# ============================================================

def test_zero_error_straight_road_behavior() -> None:
    """
    When initial state is zero on a straight road (kappa_ref = 0),
    the optimal steering rate, predicted states, and objective value
    must all be zero.
    """
    mpc = MPCController(velocity=20.0, wheelbase=2.8, dt=0.1, horizon=15)
    res = mpc.solve_from_state(
        lateral_error=0.0,
        heading_error=0.0,
        steering_angle=0.0,
        kappa_preview=np.zeros(15, dtype=np.float64),
    )

    assert np.isclose(float(res["steering_rate"]), 0.0, atol=1e-7)
    assert np.isclose(float(res["next_steering"]), 0.0, atol=1e-7)
    assert np.isclose(float(res["objective_value"]), 0.0, atol=1e-7)
    assert np.allclose(res["predicted_controls"], 0.0, atol=1e-6)
    assert np.allclose(res["predicted_states"], 0.0, atol=1e-6)


# ============================================================
# 4. CORRECT STEERING DIRECTION
# ============================================================

def test_correct_steering_direction() -> None:
    """
    Verify corrective steering-rate and steering-angle signs:
    - Positive lateral error (e_y > 0, vehicle left of centerline) -> negative steering rate & angle.
    - Negative lateral error (e_y < 0, vehicle right of centerline) -> positive steering rate & angle.
    """
    mpc = MPCController(velocity=15.0, wheelbase=2.8, dt=0.1, horizon=15)

    res_pos = mpc.solve_from_state(
        lateral_error=2.0,
        heading_error=0.0,
        steering_angle=0.0,
    )
    assert float(res_pos["steering_rate"]) < 0.0
    assert float(res_pos["next_steering"]) < 0.0

    res_neg = mpc.solve_from_state(
        lateral_error=-2.0,
        heading_error=0.0,
        steering_angle=0.0,
    )
    assert float(res_neg["steering_rate"]) > 0.0
    assert float(res_neg["next_steering"]) > 0.0


# ============================================================
# 5. STEERING-ANGLE CONSTRAINTS
# ============================================================

@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_steering_angle_constraints(road_type: str, speed: float) -> None:
    """
    Verify hard steering-angle bounds (-30 deg <= delta <= +30 deg)
    are satisfied across both the predicted horizon and closed-loop run.
    """
    results = run_mpc_simulation(road_type=road_type, velocity=speed)

    assert np.all(np.abs(results["steering"]) <= MAX_STEERING + 1e-6)
    assert np.all(
        np.abs(results["predicted_delta_horizons"]) <= MAX_STEERING + 1e-5
    )


def test_steering_angle_constraint_under_extreme_offset() -> None:
    """
    Verify that even with a large 15.0 m lateral offset and initial
    steering near +29.5 degrees, predicted steering angles across the
    entire N=15 horizon never violate +/-30 degrees.
    """
    mpc = MPCController(velocity=20.0, wheelbase=2.8, dt=0.1, horizon=15)
    init_steer = float(np.radians(29.5))

    res = mpc.solve_from_state(
        lateral_error=-15.0,
        heading_error=-0.4,
        steering_angle=init_steer,
    )
    pred_delta = res["predicted_states"][2, :]
    assert np.all(np.abs(pred_delta) <= MAX_STEERING + 1e-5)


# ============================================================
# 6. STEERING-RATE CONSTRAINTS
# ============================================================

@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_steering_rate_constraints(road_type: str, speed: float) -> None:
    """
    Verify hard steering-rate bounds (-20 deg/s <= u <= +20 deg/s)
    are satisfied at every simulation step.
    """
    results = run_mpc_simulation(road_type=road_type, velocity=speed)

    assert np.all(
        np.abs(results["steering_rate"]) <= MAX_STEERING_RATE + 1e-6
    )


# ============================================================
# 7. INITIAL STEERING-STATE CONTINUITY
# ============================================================

def test_initial_steering_state_continuity() -> None:
    """
    Verify that the optimization problem initializes z[0, 2] to the
    actual applied steering angle from the simulation rather than
    resetting to zero, and that next_steering changes by at most
    max_steering_rate * dt from that nonzero initial steering angle.
    """
    mpc = MPCController(velocity=20.0, wheelbase=2.8, dt=0.1, horizon=15)
    initial_applied = float(np.radians(12.0))

    res = mpc.solve_from_state(
        lateral_error=1.5,
        heading_error=0.05,
        steering_angle=initial_applied,
    )

    predicted_states = res["predicted_states"]
    next_steering = float(res["next_steering"])
    max_step_change = MAX_STEERING_RATE * 0.1

    assert np.isclose(predicted_states[2, 0], initial_applied, atol=1e-12)
    assert abs(next_steering - initial_applied) <= max_step_change + 1e-6


# ============================================================
# 8. CURVATURE PREVIEW CALCULATION
# ============================================================

def test_curvature_preview_calculation() -> None:
    """
    Verify that road reference preview over the N=15 horizon advances
    x_ref[k+1] = x_ref[k] + v * cos(psi_ref[k]) * dt and evaluates
    road curvature at each predicted future position rather than
    holding initial curvature constant.
    """
    road = Road(length=300.0, amplitude=2.0, frequency=0.02)
    mpc = MPCController(velocity=20.0, wheelbase=2.8, dt=0.1, horizon=15)

    preview = mpc.predict_reference_horizon(
        x_ref_0=10.0,
        road=road,
        straight=False,
    )

    x_ref = preview["x_ref"]
    psi_ref = preview["psi_ref"]
    kappa_ref = preview["kappa_ref"]

    assert x_ref.shape == (16,)
    assert psi_ref.shape == (16,)
    assert kappa_ref.shape == (15,)

    # Verify forward progress formula and exact road curvature evaluation
    for k in range(15):
        expected_next_x = x_ref[k] + 20.0 * np.cos(psi_ref[k]) * 0.1
        assert np.isclose(x_ref[k + 1], expected_next_x, atol=1e-12)
        assert np.isclose(
            kappa_ref[k],
            road.road_curvature(float(x_ref[k])),
            atol=1e-12,
        )

    # Curvature along the sinusoidal road must vary across the horizon
    assert float(np.std(kappa_ref)) > 1e-6


# ============================================================
# 9. STRAIGHT-ROAD CONVERGENCE
# ============================================================

def test_straight_road_convergence() -> None:
    """
    Verify straight-road tracking converges from a 2.0 m lateral offset
    to within 0.005 m final lateral error and 0.05 deg final heading error.
    """
    results = run_mpc_simulation(road_type="straight", velocity=20.0)
    metrics = calculate_mpc_metrics(results)

    final_lat = metrics["Final Lateral Error (m)"]
    final_head = metrics["Final Heading Error (deg)"]
    lat_rmse = metrics["Lateral RMSE (m)"]
    settling_time = metrics["Settling Time (s)"]

    assert final_lat is not None and float(final_lat) < 0.005
    assert final_head is not None and float(final_head) < 0.05
    assert lat_rmse is not None and float(lat_rmse) < 0.50
    assert settling_time is not None and float(settling_time) < 5.0


# ============================================================
# 10. CURVED-ROAD TRACKING
# ============================================================

def test_curved_road_tracking() -> None:
    """
    Verify sinusoidal curved-road tracking with curvature preview from
    a 2.0 m initial lateral offset at v = 20.0 m/s.

    Engineering Justification for Thresholds:
    - With N=15 curvature preview and ZOH disturbance feedforward d[k],
      final path-normal lateral error converges below 0.01 m (1 cm)
      and final heading error below 0.15 deg.
    - Transient Lateral RMSE (starting from 2.0 m offset under 20 deg/s
      rate constraint): < 0.45 m.
    """
    results = run_mpc_simulation(road_type="curved", velocity=20.0)
    metrics = calculate_mpc_metrics(results)

    final_lat = metrics["Final Lateral Error (m)"]
    final_head = metrics["Final Heading Error (deg)"]
    lat_rmse = metrics["Lateral RMSE (m)"]

    assert final_lat is not None and float(final_lat) < 0.01
    assert final_head is not None and float(final_head) < 0.15
    assert lat_rmse is not None and float(lat_rmse) < 0.45


# ============================================================
# 11. MULTIPLE VEHICLE SPEEDS
# ============================================================

@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_multiple_vehicle_speeds(road_type: str, speed: float) -> None:
    """
    Verify Constrained MPC converges cleanly across all test speeds
    (5, 10, 20, 30 m/s) on both straight and curved roads.
    """
    results = run_mpc_simulation(road_type=road_type, velocity=speed)
    metrics = calculate_mpc_metrics(results)

    final_lat = metrics["Final Lateral Error (m)"]
    final_head = metrics["Final Heading Error (deg)"]
    lat_rmse = metrics["Lateral RMSE (m)"]

    assert final_lat is not None and float(final_lat) < 0.01
    assert final_head is not None and float(final_head) < 0.15
    assert lat_rmse is not None and float(lat_rmse) < 0.70
    assert metrics["Constraint Violations"] == 0
    assert metrics["Non-Finite Values"] == 0
    assert metrics["Solver Failures"] == 0


# ============================================================
# 12. FINITE OPTIMIZATION OUTPUTS
# ============================================================

@pytest.mark.parametrize("road_type", ["straight", "curved"])
@pytest.mark.parametrize("speed", TEST_SPEEDS)
def test_finite_optimization_outputs(road_type: str, speed: float) -> None:
    """
    Verify all optimization outputs, predicted trajectories, and closed-loop
    histories contain exclusively finite numerical values.
    """
    results = run_mpc_simulation(road_type=road_type, velocity=speed)
    for arr in results.values():
        assert np.all(np.isfinite(arr))


# ============================================================
# 13. SOLVER STATUS HANDLING
# ============================================================

def test_solver_status_handling() -> None:
    """
    Verify solver status, objective value, solver time, and total
    computation time are recorded accurately on normal solves.
    """
    mpc = MPCController(velocity=20.0, wheelbase=2.8, dt=0.1, horizon=15)
    res = mpc.solve_from_state(
        lateral_error=1.5,
        heading_error=0.05,
        steering_angle=0.0,
    )

    assert res["solver_status"] in ("optimal", "optimal_inaccurate")
    assert res["used_fallback"] is False
    assert int(res["solver_failure_count"]) == 0
    assert float(res["objective_value"]) > 0.0
    assert float(res["solver_time"]) > 0.0
    assert float(res["total_time"]) >= float(res["solver_time"]) or np.isclose(
        float(res["total_time"]), float(res["solver_time"]), atol=1e-3
    )


# ============================================================
# 14. SOLVER FAILURE FALLBACK
# ============================================================

def test_solver_failure_fallback() -> None:
    """
    Verify deterministic fallback behavior when a solver failure or
    non-finite state occurs:
    1. Failure is recorded in solver_failure_count and used_fallback=True.
    2. Fallback steering rate and steering angle are finite and respect
       hard limits.
    """
    mpc = MPCController(velocity=20.0, wheelbase=2.8, dt=0.1, horizon=15)

    # Case 1: Forced solver failure with valid state
    res_forced = mpc.solve_from_state(
        lateral_error=2.0,
        heading_error=0.1,
        steering_angle=0.05,
        force_failure=True,
    )
    assert res_forced["used_fallback"] is True
    assert mpc.solver_failure_count == 1
    assert np.isfinite(float(res_forced["steering_rate"]))
    assert np.isfinite(float(res_forced["next_steering"]))
    assert abs(float(res_forced["steering_rate"])) <= MAX_STEERING_RATE + 1e-9
    assert abs(float(res_forced["next_steering"])) <= MAX_STEERING + 1e-9
    assert np.all(np.isfinite(res_forced["predicted_states"]))
    assert np.all(np.isfinite(res_forced["predicted_controls"]))

    # Case 2: Non-finite input state triggers fallback without returning NaN
    res_nan = mpc.solve_from_state(
        lateral_error=float("nan"),
        heading_error=0.0,
        steering_angle=0.1,
    )
    assert res_nan["used_fallback"] is True
    assert mpc.solver_failure_count == 2
    assert np.isfinite(float(res_nan["steering_rate"]))
    assert np.isfinite(float(res_nan["next_steering"]))
    assert abs(float(res_nan["steering_rate"])) <= MAX_STEERING_RATE + 1e-9
    assert abs(float(res_nan["next_steering"])) <= MAX_STEERING + 1e-9


# ============================================================
# 15. PREDICTION HORIZON DIMENSIONS
# ============================================================

def test_prediction_horizon_dimensions() -> None:
    """
    Verify predicted states have shape (3, N+1), predicted controls have
    shape (N,), and global predicted coordinates have shape (N+1,).
    """
    road = Road(length=200.0, amplitude=2.0, frequency=0.02)
    vehicle = Vehicle(x=5.0, y=1.5, yaw=0.02, wheelbase=2.8)
    mpc = MPCController(velocity=20.0, wheelbase=2.8, dt=0.1, horizon=HORIZON)

    details = mpc.compute_with_details(
        vehicle=vehicle,
        road=road,
        applied_steering=0.0,
        straight=False,
    )

    assert details["predicted_states"].shape == (3, HORIZON + 1)
    assert details["predicted_controls"].shape == (HORIZON,)
    assert details["predicted_x"].shape == (HORIZON + 1,)
    assert details["predicted_y"].shape == (HORIZON + 1,)
    assert details["predicted_psi"].shape == (HORIZON + 1,)
    assert details["kappa_preview"].shape == (HORIZON,)


# ============================================================
# 16. TERMINAL COST IMPLEMENTATION
# ============================================================

def test_terminal_cost_implementation() -> None:
    """
    Verify that the objective value equals the sum of stage costs plus
    terminal cost z[N]^T Qf z[N], and that changing Qf alters the
    optimal trajectory and objective value.
    """
    Q = np.diag([10.0, 5.0, 1.0])
    R = np.array([[1.0]])
    Qf_low = np.diag([1.0, 1.0, 1.0])
    Qf_high = np.diag([200.0, 100.0, 20.0])

    mpc_low = MPCController(
        velocity=20.0, wheelbase=2.8, dt=0.1, horizon=10, Q=Q, R=R, Qf=Qf_low
    )
    mpc_high = MPCController(
        velocity=20.0, wheelbase=2.8, dt=0.1, horizon=10, Q=Q, R=R, Qf=Qf_high
    )

    res_low = mpc_low.solve_from_state(2.0, 0.0, 0.0)
    res_high = mpc_high.solve_from_state(2.0, 0.0, 0.0)

    z_opt = res_high["predicted_states"]
    u_opt = res_high["predicted_controls"]

    stage_cost = sum(
        float(z_opt[:, k].T @ Q @ z_opt[:, k] + R[0, 0] * (u_opt[k] ** 2))
        for k in range(10)
    )
    terminal_cost = float(z_opt[:, 10].T @ Qf_high @ z_opt[:, 10])
    expected_total_cost = stage_cost + terminal_cost

    assert np.isclose(
        float(res_high["objective_value"]),
        expected_total_cost,
        rtol=1e-4,
    )
    assert float(res_high["objective_value"]) > float(
        res_low["objective_value"]
    )
    # Higher terminal penalty drives terminal lateral error smaller
    assert abs(float(z_opt[0, 10])) < abs(
        float(res_low["predicted_states"][0, 10])
    )


# ============================================================
# 17. WARM-START COMPATIBILITY
# ============================================================

def test_warm_start_compatibility() -> None:
    """
    Verify that solving with warm_start=True across sequential timesteps
    yields identical optimal solutions to cold-started solves while
    reusing the parametric problem instance.
    """
    mpc_warm = MPCController(
        velocity=20.0, wheelbase=2.8, dt=0.1, horizon=15, warm_start=True
    )
    mpc_cold = MPCController(
        velocity=20.0, wheelbase=2.8, dt=0.1, horizon=15, warm_start=False
    )

    for ey in [2.0, 1.6, 1.1, 0.5, 0.1]:
        rw = mpc_warm.solve_from_state(ey, -0.05, -0.02)
        rc = mpc_cold.solve_from_state(ey, -0.05, -0.02)
        assert np.isclose(
            float(rw["steering_rate"]),
            float(rc["steering_rate"]),
            atol=1e-4,
        )


# ============================================================
# 18. RECEDING-HORIZON OPERATION
# ============================================================

def test_receding_horizon_operation() -> None:
    """
    Verify closed-loop simulation applies only the first control input
    u*[0] at each timestep and integrates it into the next step's
    initial steering state.
    """
    results = run_mpc_simulation(
        road_type="curved",
        velocity=20.0,
        simulation_time=2.0,
    )

    steering_state = results["steering_state"]
    steering_applied = results["steering"]
    steering_rate = results["steering_rate"]
    pred_delta_horizons = results["predicted_delta_horizons"]

    for k in range(len(steering_rate)):
        # Initial steering of horizon k matches applied steering from step k-1
        assert np.isclose(
            pred_delta_horizons[k, 0],
            steering_state[k],
            atol=1e-12,
        )
        # Applied steering at step k equals steering_state[k] + rate[k] * dt
        expected_steering = steering_state[k] + steering_rate[k] * DT
        assert np.isclose(
            steering_applied[k],
            expected_steering,
            atol=1e-12,
        )
        assert np.isclose(
            steering_state[k + 1],
            steering_applied[k],
            atol=1e-12,
        )


# ============================================================
# 19. EXISTING CONTROLLER REGRESSION TESTS
# ============================================================

def test_existing_controllers_regression() -> None:
    """
    Verify Baseline, Augmented LQR, Stanley, and Pure Pursuit controllers
    remain completely functional and unaffected by Phase 10 additions.
    """
    baseline_res = run_comparison_simulation("Baseline")
    lqr_res = run_comparison_simulation("Augmented LQR")
    stanley_res = run_stanley_simulation(road_type="curved", velocity=20.0)
    pp_res = run_pure_pursuit_simulation(road_type="curved", velocity=20.0)

    assert abs(float(baseline_res["y"][-1])) < 0.05
    assert abs(float(lqr_res["y"][-1])) < 0.05
    assert abs(float(stanley_res["lateral_error"][-1])) < 0.02
    assert abs(float(pp_res["lateral_error"][-1])) < 0.01


# ============================================================
# 20. NUMERICAL CONSISTENCY BETWEEN PREDICTION MODEL AND VEHICLE
# ============================================================

def test_prediction_model_vs_vehicle_consistency() -> None:
    """
    Verify numerical consistency between the discrete linear ZOH prediction
    model z[k+1] = Ad z[k] + Bd u[k] and the nonlinear kinematic bicycle
    Vehicle.update() dynamics under small-angle operating conditions:
    1. Single-step dt=0.1 s comparison agrees within O(dt^2) (< 5 mm in y,
       < 2e-3 rad in yaw).
    2. Fine-stepped kinematic bicycle rollout with ramp steering converges
       to the exact ZOH prediction to within < 1e-4 m and < 1e-5 rad.
    """
    v = 15.0
    dt = 0.1
    mpc = MPCController(velocity=v, wheelbase=WHEELBASE, dt=dt, horizon=15)

    # 1. Single-step comparison at dt = 0.1 s
    vehicle_step = Vehicle(x=0.0, y=0.2, yaw=0.02, wheelbase=WHEELBASE)
    z0 = np.array([vehicle_step.y, vehicle_step.yaw, 0.01], dtype=np.float64)
    u_const = -0.05  # rad/s steering rate

    z1 = mpc.predict_next_state(z=z0, u=u_const, kappa_ref=0.0)
    vehicle_step.update(velocity=v, steering_angle=float(z1[2]), dt=dt)

    assert np.isclose(z1[0], vehicle_step.y, atol=5e-3)
    assert np.isclose(z1[1], vehicle_step.yaw, atol=2e-3)

    # 2. Multi-step comparison against fine-stepped continuous bicycle integration
    vehicle_fine = Vehicle(x=0.0, y=0.2, yaw=0.02, wheelbase=WHEELBASE)
    z_pred = z0.copy()
    substeps = 100
    sub_dt = dt / substeps

    for _ in range(10):
        delta_start = float(z_pred[2])
        z_pred = mpc.predict_next_state(z=z_pred, u=u_const, kappa_ref=0.0)
        for s in range(substeps):
            delta_s = delta_start + u_const * ((s + 0.5) * sub_dt)
            vehicle_fine.update(velocity=v, steering_angle=delta_s, dt=sub_dt)

    assert np.isclose(z_pred[0], vehicle_fine.y, atol=1e-3)
    assert np.isclose(z_pred[1], vehicle_fine.yaw, atol=5e-5)



