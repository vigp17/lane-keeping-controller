# Autonomous Vehicle Lane-Keeping Controller

A modular Python simulation and control benchmark for autonomous vehicle lateral path tracking using a kinematic bicycle model, realistic actuator constraints, and automated verification.

---

## Implemented Phases

1. **Phase 1–2**: Kinematic bicycle vehicle model and open-loop trajectory simulation (`src/vehicle.py`, `src/simulation.py`)
2. **Phase 3–4**: Baseline lateral PID + heading feedback controller with steering angle and steering rate constraints (`src/controllers/pid_controller.py`, `src/closed_loop_simulation.py`)
3. **Phase 5**: Discrete-time Linear Quadratic Regulator (`src/controllers/lqr_controller.py`, `src/lqr_simulation.py`)
4. **Phase 6**: Augmented state-space LQR with steering-rate control input and controller comparison framework (`src/controllers/augmented_lqr_controller.py`, `src/compare_controllers.py`)
5. **Phase 7**: Sinusoidal curved-road reference generation with analytical heading and curvature profiles (`src/road.py`, `src/curved_road_simulation.py`)
6. **Phase 8**: Stanley geometric path-tracking controller for straight and curved roads (`src/controllers/stanley_controller.py`, `src/stanley_simulation.py`)
7. **Phase 9**: Pure Pursuit lookahead geometric path-tracking controller with cumulative arc-length target selection (`src/controllers/pure_pursuit_controller.py`, `src/pure_pursuit_simulation.py`)
8. **Phase 10**: Constrained Linear-Quadratic Model Predictive Controller (MPC) with road curvature preview and hard actuator bounds (`src/controllers/mpc_controller.py`, `src/mpc_simulation.py`)

---

## Coordinate and Sign Conventions

- **Global $X$ axis**: Points forward along the initial road direction.
- **Global $Y$ axis**: Points left.
- **Yaw angle ($\psi$)**: Measured counterclockwise from the global $X$ axis in radians.
- **Steering angle ($\delta$)**: Positive steering angle produces positive (counterclockwise) yaw rate.
- **Vehicle reference point $(x, y)$**: Rear-axle center, with wheelbase $L = 2.8\text{ m}$.

---

## Phase 8: Stanley Geometric Path-Tracking Controller

### Overview

The Stanley controller (`src/controllers/stanley_controller.py`) is a nonlinear geometric path-tracking law originally developed for the Stanford DARPA Grand Challenge vehicle *Stanley*. Unlike rear-axle lateral error regulators, Stanley computes cross-track error from the **front axle** (the steered wheel center) to the nearest point on the reference path and combines heading alignment with a velocity-scaled cross-track correction.

### Front-Axle Geometry

Given rear-axle position $(x, y)$, vehicle yaw $\psi$, and wheelbase $L = 2.8\text{ m}$, the front-axle position $(x_{\text{fa}}, y_{\text{fa}})$ is:

$$
x_{\text{fa}} = x + L \cos(\psi), \qquad y_{\text{fa}} = y + L \sin(\psi)
$$

### Nearest Reference Waypoint and Cross-Track Error

Given discrete reference path waypoints $(x_{\text{ref}, i}, y_{\text{ref}, i})$ with tangent heading $\psi_{\text{ref}, i}$, the nearest waypoint index $i^*$ to the front axle is found via Euclidean distance:

$$
i^* = \arg\min_i \left( (x_{\text{ref}, i} - x_{\text{fa}})^2 + (y_{\text{ref}, i} - y_{\text{fa}})^2 \right)
$$

For the path unit tangent vector $\mathbf{t} = [\cos(\psi_{\text{ref}}), \sin(\psi_{\text{ref}})]^\top$, the local left unit normal vector is:

$$
\mathbf{n} = [-\sin(\psi_{\text{ref}}), \cos(\psi_{\text{ref}})]^\top
$$

The signed front-axle cross-track error $e_{\text{fa}}$ is defined as:

$$
e_{\text{fa}} = (\mathbf{p}_{\text{ref}} - \mathbf{p}_{\text{fa}}) \cdot \mathbf{n}
$$

Under this sign convention:
- On a straight road ($y_{\text{ref}} = 0, \psi_{\text{ref}} = 0$), if $y > 0$ (vehicle left of centerline), $e_{\text{fa}} < 0$, commanding a negative (rightward) steering angle.
- If $y < 0$ (vehicle right of centerline), $e_{\text{fa}} > 0$, commanding a positive (leftward) steering angle.

### Heading Error Definition

The heading error $\psi_e$ aligns the vehicle orientation with the local path tangent and is wrapped to $[-\pi, \pi]$:

$$
\psi_e = \operatorname{atan2}\!\left(\sin(\psi_{\text{ref}} - \psi),\ \cos(\psi_{\text{ref}} - \psi)\right)
$$

### Stanley Control Law

The unconstrained steering command $\delta_{\text{cmd}}$ is:

$$
\delta_{\text{cmd}} = \psi_e + \operatorname{atan2}\!\left(\frac{k \, e_{\text{fa}}}{v + \epsilon}\right)
$$

where:
- $k = 1.5\text{ s}^{-1}$ is the Stanley cross-track proportional gain (`stanley_gain`)
- $v \ge 0\text{ m/s}$ is the forward vehicle speed
- $\epsilon = 10^{-3}\text{ m/s}$ is a low-speed regularization constant preventing division by zero at near-standstill speeds

### Steering Constraints

All closed-loop simulations enforce identical physical actuator bounds at timestep $\Delta t = 0.1\text{ s}$:
- **Maximum steering angle**: $|\delta| \le 30^\circ$ ($0.5236\text{ rad}$)
- **Maximum steering rate**: $|\dot{\delta}| \le 20^\circ/\text{s}$ ($0.3491\text{ rad/s}$, or $2^\circ$ per $0.1\text{ s}$ timestep)

---

## Phase 9: Pure Pursuit Path-Tracking Controller

### Overview

The Pure Pursuit controller (`src/controllers/pure_pursuit_controller.py`) is a geometric lookahead path-tracking law that computes the constant-curvature circular arc connecting the vehicle's **rear-axle reference point** $(x, y)$ to a target waypoint $(x_{\text{target}}, y_{\text{target}})$ located a lookahead distance $L_d$ ahead along the reference path.

### Mathematical Derivation

Given the rear-axle position $(x, y)$, vehicle yaw $\psi$, wheelbase $L = 2.8\text{ m}$, and selected target waypoint $(x_{\text{target}}, y_{\text{target}})$:

1. **Heading-to-Target Error ($\alpha$)**:
   $$
   \alpha = \operatorname{atan2}\!\left(\sin(\tilde{\alpha}),\ \cos(\tilde{\alpha})\right), \qquad \tilde{\alpha} = \operatorname{atan2}(y_{\text{target}} - y,\ x_{\text{target}} - x) - \psi
   $$

2. **Actual Euclidean Lookahead Distance ($L_d$)**:
   $$
   L_d = \sqrt{(x_{\text{target}} - x)^2 + (y_{\text{target}} - y)^2}
   $$

3. **Pure Pursuit Steering Command ($\delta_{\text{cmd}}$)**:
   By the law of sines on the isosceles triangle formed by the turning center, rear axle, and target point ($R = \dfrac{L_d}{2\sin(\alpha)}$) combined with kinematic bicycle curvature $\tan(\delta) = L / R$:
   $$
   \delta_{\text{cmd}} = \operatorname{atan2}\!\left(2 L \sin(\alpha),\ L_d\right)
   $$

### Cumulative Arc-Length Lookahead Target Selection

To handle curved paths accurately without distorting lookahead distance along curved segments:
1. Cumulative path arc length $s_i$ is computed across discrete waypoints:
   $$
   s_0 = 0, \qquad s_i = \sum_{j=1}^{i} \sqrt{(x_{\text{ref}, j} - x_{\text{ref}, j-1})^2 + (y_{\text{ref}, j} - y_{\text{ref}, j-1})^2}
   $$
2. The nearest waypoint index $i_{\text{near}}$ to the rear axle $(x, y)$ is identified while enforcing monotonic forward progression along the path.
3. The target index $i_{\text{target}} \ge i_{\text{near}}$ is selected as the first waypoint ahead of the vehicle satisfying:
   $$
   s_{i_{\text{target}}} \ge s_{i_{\text{near}}} + L_{d,\text{desired}}
   $$
   while verifying positive forward projection in the vehicle's heading frame and clamping safely at path endpoints.

### Fixed vs. Speed-Dependent Lookahead

`PurePursuitController` supports two lookahead modes:
- **Fixed Lookahead**: $L_{d,\text{desired}} = L_{d,\text{fixed}} = 10.0\text{ m}$
- **Speed-Dependent Lookahead**:
  $$
  L_{d,\text{desired}}(v) = L_{\min} + K_v \, v
  $$
  with initial tuning parameters $L_{\min} = 3.0\text{ m}$ and $K_v = 0.5\text{ s}$.
  - At low speeds ($v = 5\text{ m/s} \implies L_{d,\text{desired}} = 5.5\text{ m}$), a shorter lookahead reduces corner cutting and accelerates lateral error convergence compared to a $10.0\text{ m}$ fixed lookahead.
  - At high speeds ($v = 30\text{ m/s} \implies L_{d,\text{desired}} = 18.0\text{ m}$), a longer lookahead prevents high-speed oscillatory instability under steering-rate constraints, at the trade-off of minor corner-cutting on curves.

---

## Phase 10: Constrained Model Predictive Control (MPC)

### Overview

The Constrained MPC controller (`src/controllers/mpc_controller.py`) formulates lateral path tracking as a receding-horizon convex quadratic program (QP) solved at each control step using **CVXPY** and **OSQP**. Unlike memoryless feedback laws that clip steering commands after computation, the MPC controller incorporates the **applied steering angle as an internal state**, optimizes over a finite **steering-rate sequence**, previews upcoming **road curvature** over a $1.5\text{ s}$ horizon, and enforces hard physical bounds on both steering angle and steering rate directly inside the optimization problem.

### State Vector, Control Input, and Continuous Error Model

The augmented state vector $\mathbf{z} \in \mathbb{R}^3$ and scalar control input $u \in \mathbb{R}$ are:

$$
\mathbf{z} = \begin{bmatrix} e_y \\ e_\psi \\ \delta_{\text{applied}} \end{bmatrix}, \qquad u = \dot{\delta}_{\text{applied}}
$$

where:
- $e_y = (\mathbf{p} - \mathbf{p}_{\text{ref}}) \cdot \mathbf{n} = -(x - x_{\text{ref}})\sin(\psi_{\text{ref}}) + (y - y_{\text{ref}})\cos(\psi_{\text{ref}})$ is the signed lateral error of the rear axle relative to the path's local left normal $\mathbf{n} = [-\sin(\psi_{\text{ref}}), \cos(\psi_{\text{ref}})]^\top$
- $e_\psi = \operatorname{atan2}(\sin(\psi - \psi_{\text{ref}}),\ \cos(\psi - \psi_{\text{ref}}))$ is the heading error relative to the local path tangent
- $\delta_{\text{applied}}$ is the physical front-wheel steering angle

Under small-angle and constant-speed ($v$) assumptions, the continuous-time linearized path-tracking dynamics with road curvature $\kappa_{\text{ref}}$ are:

$$
\begin{aligned}
\dot{e}_y &= v \, e_\psi \\
\dot{e}_\psi &= \frac{v}{L} \delta_{\text{applied}} - v \, \kappa_{\text{ref}} \\
\dot{\delta}_{\text{applied}} &= u
\end{aligned}
$$

### Discrete-Time Prediction Model (Exact Zero-Order Hold)

Discretizing over timestep $\Delta t = 0.1\text{ s}$ via exact Zero-Order Hold (ZOH) matrix integration yields:

$$
\mathbf{z}[k+1] = A_d \mathbf{z}[k] + B_d u[k] + \mathbf{d}[k]
$$

with:

$$
A_d = \begin{bmatrix} 1 & v \Delta t & \dfrac{v^2 \Delta t^2}{2 L} \\[6pt] 0 & 1 & \dfrac{v \Delta t}{L} \\[6pt] 0 & 0 & 1 \end{bmatrix}, \qquad
B_d = \begin{bmatrix} \dfrac{v^2 \Delta t^3}{6 L} \\[6pt] \dfrac{v \Delta t^2}{2 L} \\[6pt] \Delta t \end{bmatrix}, \qquad
\mathbf{d}[k] = E_d \, \kappa_{\text{ref}}[k] = \begin{bmatrix} -\dfrac{1}{2} v^2 \Delta t^2 \, \kappa_{\text{ref}}[k] \\[6pt] -v \Delta t \, \kappa_{\text{ref}}[k] \\[6pt] 0 \end{bmatrix}
$$

### Road Curvature Preview Over Horizon

Over the $N = 15$ step horizon ($1.5\text{ s}$ lookahead duration at $\Delta t = 0.1\text{ s}$), future reference positions along the road are predicted via:

$$
x_{\text{ref}}[k+1] = x_{\text{ref}}[k] + v \cos(\psi_{\text{ref}}[k]) \Delta t, \qquad k = 0, \dots, N-1
$$

and the analytical road curvature $\kappa_{\text{ref}}[k] = \kappa_{\text{road}}(x_{\text{ref}}[k])$ is evaluated at each predicted future coordinate.

### Finite-Horizon Convex Quadratic Program

At each control step, the controller solves:

$$
\min_{u[0 \dots N-1],\ \mathbf{z}[0 \dots N]} \quad \sum_{k=0}^{N-1} \left( \mathbf{z}[k]^\top Q \mathbf{z}[k] + u[k]^\top R u[k] \right) + \mathbf{z}[N]^\top Q_f \mathbf{z}[N]
$$

subject to:

$$
\begin{aligned}
\mathbf{z}[0] &= \mathbf{z}_{\text{init}} = [e_y(t),\ e_\psi(t),\ \delta_{\text{applied}}(t)]^\top \\
\mathbf{z}[k+1] &= A_d \mathbf{z}[k] + B_d u[k] + \mathbf{d}[k], && k = 0, \dots, N-1 \\
-\delta_{\max} &\le \delta_{\text{applied}}[k] \le +\delta_{\max}, && k = 1, \dots, N \quad (\delta_{\max} = 30^\circ) \\
-u_{\max} &\le u[k] \le +u_{\max}, && k = 0, \dots, N-1 \quad (u_{\max} = 20^\circ/\text{s})
\end{aligned}
$$

Initial experimental cost weights:
- Stage state weight: $Q = \operatorname{diag}([10.0,\ 5.0,\ 1.0])$
- Stage steering-rate weight: $R = \begin{bmatrix} 1.0 \end{bmatrix}$
- Terminal state weight: $Q_f = \operatorname{diag}([20.0,\ 10.0,\ 2.0])$

Only the first optimal steering-rate command $u^*[0]$ is applied at each step and integrated to update $\delta_{\text{applied}}(t + \Delta t) = \delta_{\text{applied}}(t) + u^*[0]\Delta t$.

### Solver Architecture and Deterministic Failure Handling

- **Parametric Pre-Compilation & Warm Starting**: The CVXPY problem is constructed once with `cp.Parameter` objects (`z0` and `kappa_ref`) and pre-canonicalized at initialization so online steps update parameter values and warm-start **OSQP** directly (~$0.16\text{–}0.23\text{ ms}$ mean OSQP solve time; ~$2.0\text{–}2.3\text{ ms}$ mean total controller time, well within the $100\text{ ms}$ control period).
- **Deterministic Fallback**: If OSQP reports infeasibility, fails to converge, raises an exception, or receives non-finite inputs, the failure is logged (`solver_failure_count`), `used_fallback` is flagged, and a bounded proportional-heading stabilizing rate (`compute_fallback_control()`) strictly respecting $|\delta| \le 30^\circ$ and $|\dot{\delta}| \le 20^\circ/\text{s}$ is applied.

---

## Comparison: Baseline PID vs. Augmented LQR vs. Stanley vs. Pure Pursuit vs. MPC

| Characteristic | Baseline (PID + Heading) | Augmented LQR | Stanley Controller | Pure Pursuit Controller | Constrained MPC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reference Point** | Rear axle $(x, y)$ | Rear axle $(x, y)$ | Front axle $(x_{\text{fa}}, y_{\text{fa}})$ | Rear axle $(x, y)$ | Rear axle $(x, y)$ |
| **Control Law Type** | Linear error feedback | Infinite-horizon optimal state feedback | Nonlinear local geometric feedback | Geometric lookahead chord arc fitting | Finite-horizon constrained QP optimization |
| **Preview Capability** | None | None | None | Single target point at $L_d$ | Multi-step curvature preview ($N = 15$) |
| **Actuator Constraints** | External post-clipping | Penalized in $Q, R$; clipped externally | External post-clipping | External post-clipping | Hard inequality constraints on $\delta[k]$ and $u[k]$ inside QP |
| **Computational Cost** | $\mathcal{O}(1)$ ($\sim\mu\text{s}$) | $\mathcal{O}(1)$ online after offline DARE ($\sim\mu\text{s}$) | $\mathcal{O}(M)$ waypoint search | $\mathcal{O}(M)$ waypoint search | Iterative QP solve ($\sim 2\text{ ms}$ mean total time) |

Each controller presents engineering trade-offs:
- **Constrained MPC** anticipates upcoming road curvature changes across the $1.5\text{ s}$ horizon and accounts for hard steering-angle ($\pm 30^\circ$) and steering-rate ($\pm 20^\circ/\text{s}$) saturation during trajectory planning, achieving the lowest curved-road RMSE ($0.3537\text{ m}$ at $30\text{ m/s}$) and sub-$4\text{ mm}$ final error, at the cost of higher computational complexity and dependence on an online numerical solver.
- **Augmented LQR** provides optimal state feedback with negligible online compute once the Discrete Algebraic Riccati Equation is solved offline, but cannot enforce hard state/input inequality constraints or preview multi-step curvature changes without a preview-state extension.
- **Stanley** and **Pure Pursuit** require no matrix optimization or numerical QP solvers and operate directly on geometric paths, making them lightweight and reliable, though less capable of co-optimizing actuator rate limits and multi-step curvature preview.

---

## Running Simulations and Tests

### Run Controller Simulations

```bash
# Straight-road Baseline vs. Augmented LQR comparison
python src/compare_controllers.py

# Sinusoidal curved-road reference generation (Phase 7)
python src/curved_road_simulation.py

# Stanley straight & curved road multi-speed simulation (Phase 8)
python src/stanley_simulation.py

# Pure Pursuit straight & curved road multi-speed simulation (Phase 9)
python src/pure_pursuit_simulation.py

# Constrained MPC straight & curved road multi-speed simulation (Phase 10)
python src/mpc_simulation.py
```

### Run Automated Pytest Suite

```bash
python -m pytest tests -v
```

---

## Performance Metrics Reported (`src/stanley_simulation.py`, `src/pure_pursuit_simulation.py`, & `src/mpc_simulation.py`)

1. **Lateral RMSE (m)**: Root-mean-square lateral error relative to the reference path
2. **Max Lateral Error (m)**: Peak absolute lateral error
3. **Final Lateral Error (m)**: Terminal absolute lateral error at $t = 10\text{ s}$
4. **Settling Time (s) & Overshoot (m / %)**: Time to enter and stay within $\pm 0.05\text{ m}$ and peak centerline overshoot
5. **Front-Axle CTE RMSE / Max Cross-Track Error (m)**: Reported in Stanley simulation for front-axle geometric tracking
6. **Lookahead Distance Statistics (m)**: Desired, mean, minimum, and maximum actual Euclidean lookahead distance $L_d$ reported in Pure Pursuit simulation
7. **Max Steering (deg) & Steering RMS (deg)**: Peak and root-mean-square applied steering angle
8. **Max Steering Rate (deg/s)**: Peak applied steering rate (including step 0 from $0^\circ$)
9. **Final Heading Error (deg)**: Terminal heading alignment error $|\psi - \psi_{\text{ref}}|$
10. **Solver & Controller Timing (ms)**: Mean and maximum OSQP solver time and total controller computation time per step (reported in MPC simulation)
11. **Solver Failures / Constraint Violations / Non-Finite Values**: Verification counters for solver reliability, actuator bounds, and numerical stability

---

## Known Limitations

1. **Linearized Small-Angle Prediction Model**: The MPC internal prediction model linearizes $\sin(e_\psi) \approx e_\psi$ and $\tan(\delta) \approx \delta$ around the reference path and assumes constant longitudinal speed $v$ across the $1.5\text{ s}$ horizon.
2. **Computational Overhead vs. Algebraic Controllers**: Solving a constrained QP via CVXPY + OSQP requires $\sim 2.0\text{–}2.3\text{ ms}$ mean total controller time per step (vs. microseconds for LQR, Stanley, or Pure Pursuit), though it remains well below the $100\text{ ms}$ ($\Delta t = 0.1\text{ s}$) control period.
3. **Discrete Waypoint Quantization (Stanley & Pure Pursuit)**: Nearest-waypoint and arc-length target selection operate on discrete waypoints rather than continuous spline interpolation.
4. **Pure Pursuit Corner Cutting**: Large lookahead distances at high speeds cut across the inside of sinusoidal curves.
5. **Kinematic Bicycle Assumption**: Tire slip angles, lateral force saturation, and roll dynamics are neglected across all current phases, limiting physical fidelity at high lateral accelerations.
