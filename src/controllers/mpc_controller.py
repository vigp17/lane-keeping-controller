import time
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from road import Road
from vehicle import Vehicle


class MPCController:
    """
    Constrained Linear-Quadratic Model Predictive Controller (MPC) for
    autonomous vehicle lateral path tracking.

    Augmented State Vector:
        z[k] = [e_y[k], e_psi[k], delta_applied[k]]^T  (3x1)
        where:
            e_y: Signed lateral error relative to the path's local left normal (m)
            e_psi: Heading tracking error, psi - psi_ref, wrapped to [-pi, pi] (rad)
            delta_applied: Actual applied front-wheel steering angle (rad)

    Control Input:
        u[k] = delta_rate[k]  (1x1, rad/s)

    Continuous-Time Linearized Model (small-angle, constant forward speed v):
        e_y_dot = v * e_psi
        e_psi_dot = (v / L) * delta_applied - v * kappa_ref
        delta_applied_dot = u

    Exact Zero-Order Hold (ZOH) Discrete-Time Prediction Model:
        z[k+1] = Ad * z[k] + Bd * u[k] + d[k]
        where d[k] = Ed * kappa_ref[k].
    """

    def __init__(
        self,
        velocity: float,
        wheelbase: float = 2.8,
        dt: float = 0.1,
        horizon: int = 15,
        max_steering: float = float(np.radians(30.0)),
        max_steering_rate: float = float(np.radians(20.0)),
        Q: NDArray[np.float64] | None = None,
        R: NDArray[np.float64] | float | None = None,
        Qf: NDArray[np.float64] | None = None,
        warm_start: bool = True
    ) -> None:
        """
        Initialize the Constrained MPC Controller and pre-build the
        parametric CVXPY quadratic program.
        """
        if not np.isfinite(velocity) or velocity <= 0.0:
            raise ValueError("velocity must be a finite positive number.")
        if not np.isfinite(wheelbase) or wheelbase <= 0.0:
            raise ValueError("wheelbase must be a finite positive number.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite positive number.")
        if int(horizon) < 1:
            raise ValueError("horizon (N) must be an integer >= 1.")
        if not np.isfinite(max_steering) or max_steering <= 0.0:
            raise ValueError("max_steering must be a finite positive number.")
        if not np.isfinite(max_steering_rate) or max_steering_rate <= 0.0:
            raise ValueError(
                "max_steering_rate must be a finite positive number."
            )

        self.velocity: float = float(velocity)
        self.wheelbase: float = float(wheelbase)
        self.L: float = float(wheelbase)
        self.dt: float = float(dt)
        self.horizon: int = int(horizon)
        self.N: int = int(horizon)
        self.max_steering: float = float(max_steering)
        self.max_steering_rate: float = float(max_steering_rate)
        self.warm_start: bool = bool(warm_start)

        v = self.velocity
        L = self.wheelbase
        T = self.dt

        # ==========================================================
        # 1. Exact Zero-Order Hold (ZOH) Discrete-Time State Matrices
        # ==========================================================

        self.Ad: NDArray[np.float64] = np.array(
            [
                [1.0, v * T, (v ** 2 * T ** 2) / (2.0 * L)],
                [0.0, 1.0, (v * T) / L],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.Bd: NDArray[np.float64] = np.array(
            [
                [(v ** 2 * T ** 3) / (6.0 * L)],
                [(v * T ** 2) / (2.0 * L)],
                [T],
            ],
            dtype=np.float64,
        )

        self.Ed: NDArray[np.float64] = np.array(
            [
                [-0.5 * (v ** 2) * (T ** 2)],
                [-v * T],
                [0.0],
            ],
            dtype=np.float64,
        )

        # ==========================================================
        # 2. Cost Weight Matrices (Q, R, Qf)
        # ==========================================================

        if Q is None:
            self.Q: NDArray[np.float64] = np.diag([10.0, 5.0, 1.0]).astype(
                np.float64
            )
        else:
            q_arr = np.asarray(Q, dtype=np.float64)
            if q_arr.shape != (3, 3):
                raise ValueError("Q must have shape (3, 3).")
            self.Q = q_arr

        if R is None:
            self.R: NDArray[np.float64] = np.array([[1.0]], dtype=np.float64)
        elif np.ndim(R) == 0:
            self.R = np.array([[float(R)]], dtype=np.float64)
        else:
            r_arr = np.asarray(R, dtype=np.float64)
            if r_arr.shape != (1, 1):
                raise ValueError("R must be a scalar or have shape (1, 1).")
            self.R = r_arr

        if float(self.R[0, 0]) <= 0.0:
            raise ValueError("R weight must be strictly positive.")

        if Qf is None:
            self.Qf: NDArray[np.float64] = np.diag([20.0, 10.0, 2.0]).astype(
                np.float64
            )
        else:
            qf_arr = np.asarray(Qf, dtype=np.float64)
            if qf_arr.shape != (3, 3):
                raise ValueError("Qf must have shape (3, 3).")
            self.Qf = qf_arr

        # ==========================================================
        # 3. Build Parametric CVXPY Optimization Problem
        # ==========================================================

        self._z_var = cp.Variable((3, self.N + 1), name="z")
        self._u_var = cp.Variable((1, self.N), name="u")

        self._z0_param = cp.Parameter(3, name="z0")
        self._kappa_param = cp.Parameter(self.N, name="kappa_ref")

        cost_terms: list[cp.Expression] = []
        constraints: list[cp.Constraint] = [
            self._z_var[:, 0] == self._z0_param
        ]

        r_scalar = float(self.R[0, 0])

        for k in range(self.N):
            d_k = cp.hstack(
                [
                    self.Ed[0, 0] * self._kappa_param[k],
                    self.Ed[1, 0] * self._kappa_param[k],
                    0.0,
                ]
            )

            # Linearized discrete dynamics constraint
            constraints.append(
                self._z_var[:, k + 1]
                == self.Ad @ self._z_var[:, k]
                + self.Bd[:, 0] * self._u_var[0, k]
                + d_k
            )

            # Hard steering-angle constraints on predicted state
            constraints.append(self._z_var[2, k + 1] <= self.max_steering)
            constraints.append(self._z_var[2, k + 1] >= -self.max_steering)

            # Hard steering-rate constraints on control input
            constraints.append(self._u_var[0, k] <= self.max_steering_rate)
            constraints.append(self._u_var[0, k] >= -self.max_steering_rate)

            # Stage cost: state penalty + steering-rate penalty
            cost_terms.append(cp.quad_form(self._z_var[:, k], self.Q))
            cost_terms.append(r_scalar * cp.square(self._u_var[0, k]))

        # Terminal cost at step N
        cost_terms.append(cp.quad_form(self._z_var[:, self.N], self.Qf))

        self._problem = cp.Problem(
            cp.Minimize(cp.sum(cost_terms)),
            constraints
        )

        # Pre-canonicalize problem with zero parameters so online steps
        # do not incur one-time CVXPY compilation overhead
        self._z0_param.value = np.zeros(3, dtype=np.float64)
        self._kappa_param.value = np.zeros(self.N, dtype=np.float64)
        self._problem.solve(
            solver=cp.OSQP,
            warm_start=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            verbose=False,
        )

        # Solver diagnostic counters
        self.solver_failure_count: int = 0
        self.last_solver_status: str = str(self._problem.status)
        self.last_objective_value: float = 0.0
        self.last_solver_time: float = 0.0
        self.last_total_time: float = 0.0
        self.last_used_fallback: bool = False

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize an angle in radians to the principal interval [-pi, pi].
        """
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def compute_disturbance_vector(
        self,
        kappa_ref: float
    ) -> NDArray[np.float64]:
        """
        Compute the 3x1 discrete curvature disturbance vector d[k] = Ed * kappa_ref.
        """
        return (self.Ed[:, 0] * float(kappa_ref)).astype(np.float64)

    def predict_next_state(
        self,
        z: NDArray[np.float64],
        u: float,
        kappa_ref: float = 0.0
    ) -> NDArray[np.float64]:
        """
        Evaluate one step of the discrete-time prediction model:
            z[k+1] = Ad * z[k] + Bd * u[k] + d[k]
        """
        z_vec = np.asarray(z, dtype=np.float64).reshape(3)
        d_vec = self.compute_disturbance_vector(kappa_ref)
        next_z = self.Ad @ z_vec + self.Bd[:, 0] * float(u) + d_vec
        return np.asarray(next_z, dtype=np.float64)

    @staticmethod
    def find_nearest_reference_point(
        x: float,
        y: float,
        road: Road,
        straight: bool = False
    ) -> tuple[float, float, float]:
        """
        Find the orthogonal projection reference point (x_ref, y_ref, psi_ref)
        on the road centerline corresponding to global rear-axle position (x, y).
        """
        if straight or road.A == 0.0 or road.k == 0.0:
            return float(x), 0.0, 0.0

        # Refine closest longitudinal coordinate xi on y_ref(xi) = A * sin(k * xi)
        # using 4 steps of damped Newton-Raphson on squared Euclidean distance
        xi = float(x)
        A = road.A
        k = road.k

        for _ in range(4):
            y_r = A * np.sin(k * xi)
            dy_r = A * k * np.cos(k * xi)
            d2y_r = -A * (k ** 2) * np.sin(k * xi)

            grad = (xi - x) + (y_r - y) * dy_r
            hess = 1.0 + (dy_r ** 2) + (y_r - y) * d2y_r
            if abs(hess) > 1e-9:
                xi = float(xi - grad / hess)

        y_ref = float(road.reference_lateral_position(xi))
        psi_ref = float(road.reference_heading(xi))

        return float(xi), y_ref, psi_ref

    def compute_tracking_error(
        self,
        vehicle: Vehicle,
        road: Road,
        straight: bool = False
    ) -> tuple[float, float, float, float, float]:
        """
        Compute signed lateral tracking error e_y relative to the path's local
        left normal and heading error e_psi = psi - psi_ref.

        For path tangent t = [cos(psi_ref), sin(psi_ref)]^T, the left normal is:
            n = [-sin(psi_ref), cos(psi_ref)]^T

        Signed lateral error:
            e_y = dot([x - x_ref, y - y_ref], n)
                = -(x - x_ref) * sin(psi_ref) + (y - y_ref) * cos(psi_ref)

        Returns:
            Tuple of (e_y, e_psi, x_ref, y_ref, psi_ref).
        """
        x_ref, y_ref, psi_ref = self.find_nearest_reference_point(
            x=vehicle.x,
            y=vehicle.y,
            road=road,
            straight=straight
        )

        e_y = float(
            -(vehicle.x - x_ref) * np.sin(psi_ref)
            + (vehicle.y - y_ref) * np.cos(psi_ref)
        )
        e_psi = self.normalize_angle(vehicle.yaw - psi_ref)

        return e_y, e_psi, x_ref, y_ref, psi_ref

    def predict_reference_horizon(
        self,
        x_ref_0: float,
        road: Road,
        straight: bool = False
    ) -> dict[str, NDArray[np.float64]]:
        """
        Predict future reference path coordinates, headings, and curvatures
        over the N-step optimization horizon using the forward progress model:
            x_ref[k+1] = x_ref[k] + v * cos(psi_ref[k]) * dt
        """
        x_ref_arr = np.zeros(self.N + 1, dtype=np.float64)
        y_ref_arr = np.zeros(self.N + 1, dtype=np.float64)
        psi_ref_arr = np.zeros(self.N + 1, dtype=np.float64)
        kappa_ref_arr = np.zeros(self.N, dtype=np.float64)

        x_curr = float(x_ref_0)

        for k in range(self.N + 1):
            x_ref_arr[k] = x_curr

            if straight or road.A == 0.0 or road.k == 0.0:
                y_k = 0.0
                psi_k = 0.0
                kappa_k = 0.0
            else:
                y_k = float(road.reference_lateral_position(x_curr))
                psi_k = float(road.reference_heading(x_curr))
                kappa_k = float(road.road_curvature(x_curr))

            y_ref_arr[k] = y_k
            psi_ref_arr[k] = psi_k

            if k < self.N:
                kappa_ref_arr[k] = kappa_k
                x_curr = float(
                    x_curr + self.velocity * np.cos(psi_k) * self.dt
                )

        return {
            "x_ref": x_ref_arr,
            "y_ref": y_ref_arr,
            "psi_ref": psi_ref_arr,
            "kappa_ref": kappa_ref_arr,
        }

    def compute_fallback_control(
        self,
        lateral_error: float,
        heading_error: float,
        steering_angle: float
    ) -> tuple[float, float]:
        """
        Deterministic safe fallback steering-rate command if the QP solver
        fails to converge or encounters invalid numerical inputs.

        Applies a bounded proportional-heading stabilizing rate that strictly
        respects both [-max_steering_rate, +max_steering_rate] and
        [-max_steering, +max_steering].
        """
        safe_delta = (
            float(np.clip(steering_angle, -self.max_steering, self.max_steering))
            if np.isfinite(steering_angle)
            else 0.0
        )

        if np.isfinite(lateral_error) and np.isfinite(heading_error):
            target_delta = float(
                np.clip(
                    -0.05 * lateral_error - 0.8 * heading_error,
                    -self.max_steering,
                    self.max_steering,
                )
            )
            desired_rate = (target_delta - safe_delta) / self.dt
        else:
            desired_rate = -safe_delta / self.dt

        rate_limited = float(
            np.clip(
                desired_rate,
                -self.max_steering_rate,
                self.max_steering_rate,
            )
        )
        next_delta = float(
            np.clip(
                safe_delta + rate_limited * self.dt,
                -self.max_steering,
                self.max_steering,
            )
        )
        actual_rate = float((next_delta - safe_delta) / self.dt)

        return actual_rate, next_delta

    def solve_from_state(
        self,
        lateral_error: float,
        heading_error: float,
        steering_angle: float,
        kappa_preview: NDArray[np.float64] | None = None,
        force_failure: bool = False
    ) -> dict[str, Any]:
        """
        Solve the constrained finite-horizon MPC problem from initial state
        z0 = [lateral_error, heading_error, steering_angle]^T.
        """
        t_start = time.perf_counter()

        if kappa_preview is None:
            kappa_arr = np.zeros(self.N, dtype=np.float64)
        else:
            kappa_arr = np.asarray(kappa_preview, dtype=np.float64).reshape(-1)
            if kappa_arr.size != self.N:
                raise ValueError(
                    f"kappa_preview must have length {self.N}, got {kappa_arr.size}."
                )

        clamped_init_steering = (
            float(np.clip(steering_angle, -self.max_steering, self.max_steering))
            if np.isfinite(steering_angle)
            else 0.0
        )
        z0 = np.array(
            [
                float(lateral_error),
                self.normalize_angle(float(heading_error))
                if np.isfinite(heading_error)
                else float("nan"),
                clamped_init_steering,
            ],
            dtype=np.float64,
        )

        inputs_valid = bool(
            np.all(np.isfinite(z0)) and np.all(np.isfinite(kappa_arr))
        )

        solver_time = 0.0
        solver_status = "invalid_input"
        solver_succeeded = False

        if inputs_valid and not force_failure:
            self._z0_param.value = z0
            self._kappa_param.value = kappa_arr

            t_solve_start = time.perf_counter()
            try:
                self._problem.solve(
                    solver=cp.OSQP,
                    warm_start=self.warm_start,
                    eps_abs=1e-6,
                    eps_rel=1e-6,
                    verbose=False,
                )
                t_solve_end = time.perf_counter()
                stats_time = getattr(
                    self._problem.solver_stats, "solve_time", None
                )
                if stats_time is not None and np.isfinite(stats_time) and stats_time > 0.0:
                    solver_time = float(stats_time)
                else:
                    solver_time = float(t_solve_end - t_solve_start)

                solver_status = str(self._problem.status)
                if (
                    self._problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                    and self._u_var.value is not None
                    and self._z_var.value is not None
                    and np.all(np.isfinite(self._u_var.value))
                    and np.all(np.isfinite(self._z_var.value))
                ):
                    solver_succeeded = True
            except Exception:
                t_solve_end = time.perf_counter()
                solver_time = float(t_solve_end - t_solve_start)
                solver_status = "solver_exception"
                solver_succeeded = False
        elif force_failure:
            solver_status = "forced_failure"

        if solver_succeeded:
            assert self._u_var.value is not None
            assert self._z_var.value is not None
            predicted_controls = np.asarray(
                self._u_var.value, dtype=np.float64
            ).reshape(self.N)
            predicted_states = np.asarray(
                self._z_var.value, dtype=np.float64
            ).reshape(3, self.N + 1)
            predicted_states[:, 0] = z0

            raw_rate = float(predicted_controls[0])
            applied_rate = float(
                np.clip(
                    raw_rate,
                    -self.max_steering_rate,
                    self.max_steering_rate,
                )
            )
            next_steering = float(
                np.clip(
                    clamped_init_steering + applied_rate * self.dt,
                    -self.max_steering,
                    self.max_steering,
                )
            )
            applied_rate = float(
                (next_steering - clamped_init_steering) / self.dt
            )
            objective_value = float(
                np.asarray(self._problem.value, dtype=np.float64).item()
            )
            used_fallback = False
        else:
            self.solver_failure_count += 1
            used_fallback = True
            applied_rate, next_steering = self.compute_fallback_control(
                lateral_error=lateral_error,
                heading_error=heading_error,
                steering_angle=clamped_init_steering,
            )

            # Construct safe finite rollout over horizon for diagnostics
            safe_z0 = np.where(np.isfinite(z0), z0, 0.0)
            safe_kappa = np.where(np.isfinite(kappa_arr), kappa_arr, 0.0)
            predicted_states = np.zeros((3, self.N + 1), dtype=np.float64)
            predicted_controls = np.zeros(self.N, dtype=np.float64)
            predicted_states[:, 0] = safe_z0

            for k in range(self.N):
                u_fb, _ = self.compute_fallback_control(
                    lateral_error=float(predicted_states[0, k]),
                    heading_error=float(predicted_states[1, k]),
                    steering_angle=float(predicted_states[2, k]),
                )
                predicted_controls[k] = u_fb
                predicted_states[:, k + 1] = self.predict_next_state(
                    predicted_states[:, k],
                    u_fb,
                    float(safe_kappa[k]),
                )

            objective_value = float(
                sum(
                    float(
                        predicted_states[:, k].T
                        @ self.Q
                        @ predicted_states[:, k]
                        + self.R[0, 0] * (predicted_controls[k] ** 2)
                    )
                    for k in range(self.N)
                )
                + float(
                    predicted_states[:, self.N].T
                    @ self.Qf
                    @ predicted_states[:, self.N]
                )
            )

        total_time = float(time.perf_counter() - t_start)

        self.last_solver_status = solver_status
        self.last_objective_value = objective_value
        self.last_solver_time = solver_time
        self.last_total_time = total_time
        self.last_used_fallback = used_fallback

        return {
            "steering_rate": applied_rate,
            "next_steering": next_steering,
            "predicted_states": predicted_states,
            "predicted_controls": predicted_controls,
            "solver_status": solver_status,
            "objective_value": objective_value,
            "solver_time": solver_time,
            "total_time": total_time,
            "used_fallback": used_fallback,
            "solver_failure_count": self.solver_failure_count,
        }

    def compute_with_details(
        self,
        vehicle: Vehicle,
        road: Road,
        applied_steering: float,
        straight: bool = False,
        force_failure: bool = False
    ) -> dict[str, Any]:
        """
        Compute the receding-horizon MPC steering-rate and steering-angle
        commands along with full predicted trajectories and solver timings.
        """
        t_total_start = time.perf_counter()

        e_y, e_psi, x_ref_0, y_ref_0, psi_ref_0 = self.compute_tracking_error(
            vehicle=vehicle,
            road=road,
            straight=straight,
        )

        ref_horizon = self.predict_reference_horizon(
            x_ref_0=x_ref_0,
            road=road,
            straight=straight,
        )

        solve_result = self.solve_from_state(
            lateral_error=e_y,
            heading_error=e_psi,
            steering_angle=applied_steering,
            kappa_preview=ref_horizon["kappa_ref"],
            force_failure=force_failure,
        )

        predicted_states = solve_result["predicted_states"]
        x_ref_arr = ref_horizon["x_ref"]
        y_ref_arr = ref_horizon["y_ref"]
        psi_ref_arr = ref_horizon["psi_ref"]

        # Reconstruct predicted global (X, Y, psi) coordinates along the horizon
        predicted_x = (
            x_ref_arr - predicted_states[0, :] * np.sin(psi_ref_arr)
        ).astype(np.float64)
        predicted_y = (
            y_ref_arr + predicted_states[0, :] * np.cos(psi_ref_arr)
        ).astype(np.float64)
        predicted_psi = (psi_ref_arr + predicted_states[1, :]).astype(
            np.float64
        )

        total_time = float(time.perf_counter() - t_total_start)
        self.last_total_time = total_time
        solve_result["total_time"] = total_time
        solve_result["lateral_error"] = e_y
        solve_result["heading_error"] = e_psi
        solve_result["x_ref"] = x_ref_0
        solve_result["y_ref"] = y_ref_0
        solve_result["psi_ref"] = psi_ref_0
        solve_result["kappa_preview"] = ref_horizon["kappa_ref"]
        solve_result["predicted_x"] = predicted_x
        solve_result["predicted_y"] = predicted_y
        solve_result["predicted_psi"] = predicted_psi

        return solve_result

    def compute(
        self,
        lateral_error: float,
        heading_error: float,
        steering_angle: float,
        kappa_preview: NDArray[np.float64] | None = None
    ) -> float:
        """
        Compute the optimal first-step steering-rate command u[0] (rad/s).
        """
        res = self.solve_from_state(
            lateral_error=lateral_error,
            heading_error=heading_error,
            steering_angle=steering_angle,
            kappa_preview=kappa_preview,
        )
        return float(res["steering_rate"])
