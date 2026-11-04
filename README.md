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

## Comparison: Baseline PID vs. Augmented LQR vs. Stanley vs. Pure Pursuit

| Characteristic | Baseline (PID + Heading) | Augmented LQR | Stanley Controller | Pure Pursuit Controller |
| :--- | :--- | :--- | :--- | :--- |
| **Reference Point** | Rear axle $(x, y)$ | Rear axle $(x, y)$ | Front axle $(x_{\text{fa}}, y_{\text{fa}})$ | Rear axle $(x, y)$ |
| **Control Law Type** | Linear error feedback | Optimal state feedback on $[\,e_y, e_\psi, \delta\,]^\top$ | Nonlinear local cross-track + heading | Geometric lookahead arc fitting |
| **Preview / Lookahead** | None (instantaneous error) | None (instantaneous state) | None (nearest front-axle point) | Explicit lookahead distance $L_d$ along path |
| **Speed Adaptation** | Fixed gains | Velocity-parameterized $A_d(v), B_d(v)$ | Inverse velocity in $\arctan(k e_{\text{fa}} / (v + \epsilon))$ | Speed-dependent lookahead $L_{\min} + K_v v$ |
| **Actuator Rate Awareness** | External post-clipping | Internalized in quadratic cost $R$ | External post-clipping | External post-clipping |
| **Curved-Path Behavior** | Requires path frame transformation | Requires curvature feedforward | High accuracy at low/mid speed; sensitive to rate limits at high gain | Smooth steering and strong damping; cuts corners at large $L_d$ |

No single controller is universally superior across all driving regimes:
- **Pure Pursuit** provides smooth, inherently damped steering commands due to its lookahead preview and is highly resilient to local path noise, but exhibits corner-cutting on tight curves when $L_d$ is large.
- **Stanley** directly eliminates local front-axle cross-track and heading error without corner-cutting bias from lookahead chords, but reacts more aggressively to local path perturbations.
- **Augmented LQR** explicitly accounts for steering-rate effort in its optimal feedback gain, making it well-suited for high-speed highway lane keeping when paired with curvature feedforward.

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
```

### Run Automated Pytest Suite

```bash
python -m pytest tests -v
```

---

## Performance Metrics Reported (`src/stanley_simulation.py` & `src/pure_pursuit_simulation.py`)

1. **Lateral RMSE (m)**: Root-mean-square rear-axle lateral error $y - y_{\text{ref}}(x)$
2. **Max Lateral Error (m)**: Peak absolute rear-axle lateral error
3. **Final Lateral Error (m)**: Terminal absolute rear-axle lateral error at $t = 10\text{ s}$
4. **Front-Axle CTE RMSE / Max Cross-Track Error (m)**: Reported in Stanley simulation for front-axle geometric tracking
5. **Lookahead Distance Statistics (m)**: Desired, mean, minimum, and maximum actual Euclidean lookahead distance $L_d$ reported in Pure Pursuit simulation
6. **Max Steering (deg)**: Peak applied steering angle
7. **Steering RMS (deg)**: Root-mean-square applied steering effort
8. **Max Steering Rate (deg/s)**: Peak applied steering rate (including step 0 from $0^\circ$)
9. **Final Heading Error (deg)**: Terminal heading alignment error $|\psi_{\text{ref}} - \psi|$
10. **Constraint Violations / Non-Finite Values**: Verification counters for actuator bounds and numerical stability

---

## Known Limitations

1. **Discrete Waypoint Quantization**: Nearest-waypoint and arc-length target selection operate on discrete waypoints rather than continuous spline interpolation; waypoint spacing causes a minor difference between desired arc-length lookahead and actual Euclidean chord distance $L_d$.
2. **Pure Pursuit Corner Cutting**: Because Pure Pursuit steers toward a single chord point at distance $L_d$ ahead, large lookahead distances at high speeds cut across the inside of sinusoidal curves.
3. **No Curvature Feedforward in Pure Stanley**: On curved roads ($y_{\text{ref}}(x) = A\sin(kx)$), pure Stanley retains a small steady-state geometric offset proportional to $L \kappa(x)$.
4. **Kinematic Bicycle Assumption**: Tire slip angles, lateral force saturation, and roll dynamics are neglected, which limits physical fidelity at high lateral accelerations.
5. **External Rate Limiting**: Because steering rate limits ($\pm 20^\circ/\text{s}$) are enforced downstream of geometric controllers (Stanley and Pure Pursuit), overly short lookahead distances or overly high gains can induce rate-limit phase lag.
