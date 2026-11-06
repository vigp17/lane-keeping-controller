# Autonomous Vehicle Lane-Keeping: Controller Design and CARLA Validation

**Python · CARLA 0.9.16 · Stanley · Pure Pursuit · Augmented LQR · Constrained MPC · pytest**

A simulation-based lateral-control project progressing from a kinematic bicycle model to five closed-loop controller integrations in CARLA. The project compares tracking accuracy, steering smoothness, actuator constraints, speed sensitivity, and an initial-offset recovery experiment.

> **Scope:** This is a simulation and validation portfolio project, not a production-ready or safety-certified autonomous-driving system. Results below are taken from reported CARLA runs and telemetry summaries; they do not establish performance in arbitrary traffic, road, or weather conditions.

## Highlights

- **Five controllers:** Baseline lateral/heading feedback with curvature feedforward, Stanley, Pure Pursuit, augmented LQR, and constrained MPC.
- **15 speed-benchmark runs:** all five controllers completed the same route at target speeds of **5, 8, and 10 m/s**.
- **Additional recovery test:** Stanley at 8 m/s with a **+1 m initial lateral offset**.
- **Environment:** CARLA 0.9.16, Tesla Model 3, `Town10HD_Opt`, physics-derived wheelbase **3.0046 m**.
- **Route:** 135.24 m, including a 60.22 m sustained right-hand curve (minimum radius approximately 37.66 m).
- **Actuator constraints:** applied steering **±30°**, applied steering rate **±20°/s**, synchronous timestep **0.05 s**.
- **Reported outcome:** all 16 runs completed with **zero detected collisions** and **zero reported applied-steering constraint violations**.
- **Tests:** latest reported full suite: **264 passed in 42.39 s**.

![Five-controller comparison at 8 m/s](results/final_controller_comparison.png)

## Benchmark Configuration

| Parameter | Value |
|---|---|
| Map | `Town10HD_Opt` |
| Vehicle | `vehicle.tesla.model3` |
| Spawn point | 61 |
| Route | Lane 1; roads 10 → 0 → 3 |
| Waypoints | 277, approximately 0.5 m spacing |
| Route length | 135.24 m |
| Straight approach / sustained curve / straight exit | 57.50 / 60.22 / 17.51 m |
| Target speeds | 5, 8, 10 m/s |
| Timestep | 0.05 s |
| Steering angle / rate limits | ±30° / ±20°/s |

The controller-specific runs use the same nominal route, vehicle, steering constraints, and longitudinal controller design. Results were gathered from separate executions; their initial conditions and metrics should be audited from raw telemetry before drawing broader statistical conclusions.

## Multi-Speed Tracking Results

### Rear-axle tracking RMSE (m; lower is better)

| Controller | 5 m/s | 8 m/s | 10 m/s |
|---|---:|---:|---:|
| Stanley | 0.0344 | **0.0488** | 0.1089 |
| Constrained MPC | **0.0268** | 0.0664 | **0.0942** |
| Pure Pursuit | 0.0609 | 0.1460 | 0.2132 |
| Augmented LQR | 0.0837 | 0.1873 | 0.2583 |
| Baseline feedback + feedforward | 0.1604 | 0.2847 | 0.3540 |

### Front-axle cross-track RMSE (m; lower is better)

| Controller | 5 m/s | 8 m/s | 10 m/s |
|---|---:|---:|---:|
| Stanley | **0.0321** | **0.0915** | 0.1410 |
| Constrained MPC | 0.0940 | 0.1116 | **0.1283** |
| Pure Pursuit | 0.1222 | 0.1871 | 0.2437 |
| Augmented LQR | 0.1496 | 0.2315 | 0.2913 |
| Baseline feedback + feedforward | 0.2238 | 0.3276 | 0.3861 |

### Steering effort at 8 m/s

| Controller | Steering RMS (°) | Steering-rate RMS (°/s) | Peak steering rate (°/s) |
|---|---:|---:|---:|
| Stanley | 3.3415 | 4.4192 | 14.8446 |
| Constrained MPC | 3.4207 | 2.6930 | 12.5964 |
| Pure Pursuit | **3.3156** | **1.5476** | **6.4024** |
| Augmented LQR | 3.3987 | 2.4933 | 15.2076 |
| Baseline feedback + feedforward | 3.4049 | 2.8547 | 20.0000 |

**Interpretation:** At 5 m/s, MPC had the lowest rear-axle RMSE and Stanley had the lowest front-axle RMSE. At 8 m/s, Stanley had the lowest RMSE on both measures. At 10 m/s, MPC had the lowest RMSE on both measures, while Stanley had smaller peak deviations. Pure Pursuit had the lowest steering-rate RMS in the 8 m/s run. These are **scenario-specific** findings, not universal rankings.

### Sustained-curve rear-axle RMSE at 10 m/s

| Controller | RMSE (m) |
|---|---:|
| Stanley | 0.1587 |
| Constrained MPC | **0.1432** |
| Pure Pursuit | 0.2988 |
| Augmented LQR | 0.3925 |
| Baseline | 0.5270 |

## Stanley +1 m Initial-Offset Recovery

A separate test started the Tesla approximately 1 m left of the reference lane centerline, with nominally aligned initial heading and an 8 m/s target speed. The original Stanley gain was retained.

| Metric | Zero offset | +1 m offset |
|---|---:|---:|
| Rear-axle RMSE | 0.0488 m | 0.4542 m |
| Front-axle CTE RMSE | 0.0915 m | 0.4896 m |
| Maximum absolute rear-axle error | 0.0942 m | 1.1020 m |
| Maximum absolute front-axle CTE | 0.1527 m | 1.3000 m |
| Maximum heading error | Not tabulated | 15.67° |
| Maximum applied steering | 5.8144° | 20.0000° |
| Maximum applied steering rate | 14.8446°/s | 20.0000°/s |
| Route completed | Yes | Yes |
| Detected collisions | 0 | 0 |

The offset run showed **oscillatory overshoot** on the initial straight. At low speed, the Stanley law requested a large steering correction (peak reported requested magnitude **89.57°**). Angle and rate constraints limited applied steering, and delayed steering reversals contributed to repeated centerline crossings. During the sustained curve, rear-axle RMSE was **0.0688 m**, close to the zero-offset curve result of **0.0682 m**. This demonstrates recovery in this particular simulation, **not a general stability guarantee**.

Settling time and recovery distance under a sustained ±0.05 m criterion have **not yet been calculated** from raw telemetry.

![Stanley offset-recovery trajectory](results/phase12b_stanley_offset_1m_8ms_trajectory.png)

![Stanley offset-recovery error](results/phase12b_stanley_offset_1m_8ms_cross_track_error.png)

## MPC Computational Performance

The CARLA MPC uses **15 prediction steps × 0.05 s = 0.75 s**, CVXPY, and OSQP.

| Target speed | Mean OSQP (ms) | Max OSQP (ms) | Mean total controller (ms) | Max total controller (ms) | Solver failures | Observed >50 ms misses |
|---|---:|---:|---:|---:|---:|---:|
| 5 m/s | 0.7611 | 2.2250 | 4.3762 | 8.4010 | 0 | 0 |
| 8 m/s | 0.2091 | 0.6578 | 4.0738 | 11.4903 | 0 | 0 |
| 10 m/s | 0.2147 | 0.5658 | 4.1621 | 7.5572 | 0 | 0 |

Observed desktop timing within a 50 ms control period does **not** imply hard real-time guarantees on embedded hardware.

## Comparison Plots (8 m/s)

![Rear-axle tracking error](results/final_rear_axle_error_comparison.png)

![Front-axle cross-track error](results/final_front_axle_error_comparison.png)

![Applied steering angle](results/final_steering_angle_comparison.png)

![Applied steering rate](results/final_steering_rate_comparison.png)

![Vehicle speed](results/final_speed_comparison.png)

> The `final_*` images represent the **original 8 m/s runs**, not the full multi-speed sweep. Signed cross-track errors use different conventions in some controller integrations; use sign-normalized data or error magnitudes for overlays.

## System Architecture

```text
CARLA map waypoints --> route extraction / curvature estimation
                                      |
CARLA Tesla --> CARLA adapter --> tracking state / reference errors
                                      |
                 +--------------------+---------------------+
                 |          |          |          |          |
              Baseline    Stanley    Pure       Augmented    MPC
                                   Pursuit       LQR
                 +----------+----------+----------+----------+
                                      |
                            Steering angle / rate
                                      |
                         Limits + CARLA conversion
                                      |
                      Vehicle throttle, brake, steering
                                      |
                       Telemetry --> metrics --> plots
```

`CarlaControllerAdapter` uses live physics geometry to identify the front and rear axle, transforms CARLA coordinates to the controller convention, and maps desired steering to normalized CARLA input. The nominal maximum front-wheel angle is approximately **70°**, but effective steering response can depend on speed and vehicle physics settings.

## Control Models and Algorithms

### Kinematic bicycle and reference road

The offline kinematic bicycle model uses rear-axle position $(x,y)$, yaw $\psi$, wheelbase $L=2.8\,\mathrm{m}$, speed $v$, and front-wheel steering $\delta$:

$$
\dot{x}=v\cos\psi,\qquad
\dot{y}=v\sin\psi,\qquad
\dot{\psi}=\frac{v}{L}\tan\delta.
$$

For an offline sinusoidal reference $y_{\mathrm{ref}}(x)=A\sin(kx)$:

$$
\psi_{\mathrm{ref}}=\arctan(Ak\cos(kx)),\qquad
\kappa_{\mathrm{ref}}=\frac{-Ak^2\sin(kx)}{\left[1+(Ak\cos(kx))^2\right]^{3/2}}.
$$

CARLA uses the measured Tesla wheelbase **3.0046 m** instead of the offline **2.8 m** default. Offline simulations use a **0.1 s** timestep, while the CARLA integrations use **0.05 s**.

### Error and sign conventions

In controller coordinates, $X$ points along the initial road direction, $Y$ points left, positive yaw is counterclockwise, and positive steering turns left. The local left normal to a reference heading $\psi_{\mathrm{ref}}$ is:

$$
\mathbf{n}=\begin{bmatrix}-\sin\psi_{\mathrm{ref}}\\\cos\psi_{\mathrm{ref}}\end{bmatrix}.
$$

The rear-axle lateral error used by baseline, augmented LQR, and MPC is:

$$
e_y=(\mathbf{p}_{\mathrm{rear}}-\mathbf{p}_{\mathrm{ref}})\cdot\mathbf{n}.
$$

Stanley's front-axle error uses the opposite displacement order:

$$
e_{\mathrm{fa}}=(\mathbf{p}_{\mathrm{ref}}-\mathbf{p}_{\mathrm{front}})\cdot\mathbf{n}.
$$

Stanley's rear-error telemetry also follows a controller-specific sign convention. RMSE and absolute errors are invariant to sign, but signed traces must be normalized for comparison.

### Baseline feedback with curvature feedforward

The baseline reuses `PIDController` with $K_p=0.05$, $K_i=K_d=0$, and heading gain $K_\psi=0.8$:

$$
\delta_{\mathrm{cmd}}=\arctan(L\kappa_{\mathrm{ref}})-K_pe_y-K_\psi e_\psi.
$$

This is **proportional lateral and heading feedback**, not a fully tuned three-term PID. Steering angle and rate are constrained after the command is computed.

### Stanley

With front-axle position $\mathbf{p}_{\mathrm{front}}=(x+L\cos\psi,\,y+L\sin\psi)$ and wrapped heading correction:

$$
\psi_e=\mathrm{atan2}\!\left(\sin(\psi_{\mathrm{ref}}-\psi),\cos(\psi_{\mathrm{ref}}-\psi)\right),
$$

$$
\delta_{\mathrm{cmd}}=\psi_e+\arctan\!\left(\frac{k_s e_{\mathrm{fa}}}{v+\epsilon}\right).
$$

The tested gain is $k_s=1.5\,\mathrm{s}^{-1}$ and low-speed regularization is $\epsilon=10^{-3}\,\mathrm{m/s}$. The +1 m offset test highlights large initial steering requests and oscillatory recovery under rate constraints.

### Pure Pursuit

Pure Pursuit chooses a target ahead along cumulative path arc length. From the rear axle, let $L_d$ be the Euclidean distance to the selected target and $\alpha$ the wrapped bearing error:

$$
\delta_{\mathrm{cmd}}=\arctan\!\left(\frac{2L\sin\alpha}{L_d}\right).
$$

The implementation supports fixed lookahead (10 m) and dynamic desired lookahead $L_{d,\mathrm{desired}}=3.0+0.5v$ (meters when $v$ is in m/s). Actual Euclidean lookahead can differ due to waypoint spacing and road curvature.

### Augmented LQR

The augmented state is $z=[e_y,e_\psi,\delta]^T$ and the input is steering rate $u=\dot\delta$. The CARLA integration uses steering deviation relative to curvature feedforward:

$$
\delta_{\mathrm{ff}}=\arctan(L\kappa_{\mathrm{ref}}),\qquad
u=u_{\mathrm{feedback}}+\frac{d\delta_{\mathrm{ff}}}{dt}.
$$

The feedback gain is computed for the configured nominal speed. Cost weights were not systematically retuned for the CARLA vehicle's dynamic response.

### Constrained MPC

MPC optimizes steering rate over a finite horizon with state $z=[e_y,e_\psi,\delta]^T$:

$$
\min_{u_0,\ldots,u_{N-1}}
\sum_{k=0}^{N-1}\left(z_k^TQz_k+u_k^TRu_k\right)+z_N^TQ_fz_N
$$

subject to discrete prediction dynamics and hard bounds $|\delta_k|\le30^\circ$ and $|u_k|\le20^\circ/\mathrm{s}$. Weights: $Q=\mathrm{diag}(10,5,1)$, $R=1$, and $Q_f=\mathrm{diag}(20,10,2)$. CVXPY + OSQP provides the QP solve, with warm starting and fallback handling. The offline horizon is **15 × 0.1 s = 1.5 s**; CARLA uses **15 × 0.05 s = 0.75 s** with matrices constructed for the actual timestep and nominal speed.

## Development Milestones

| Phase | Work |
|---|---|
| 1–6 | Kinematic model, baseline, LQR, augmented LQR, offline comparison and tests |
| 7 | Curved-road reference generation |
| 8 | Stanley |
| 9 | Pure Pursuit |
| 10 | Constrained MPC |
| 11 | CARLA adapter and five controller integrations |
| 12A | Unified benchmark metrics and consistency checks |
| 12B | 5, 8, 10 m/s tests and Stanley +1 m offset recovery |
| 12D | Comparison charts and telemetry analysis |
| 12E | Technical documentation |
| 12F | CARLA demonstration recordings |

Further disturbance/noise tests and a fully parameterized multi-scenario runner were proposed but are **not claimed complete**.

## Installation and Reproduction

### Offline simulations

```powershell
python -m pip install -r requirements.txt
python src/compare_controllers.py
python src/curved_road_simulation.py
python src/stanley_simulation.py
python src/pure_pursuit_simulation.py
python src/mpc_simulation.py
python -m pytest tests -q
```

### CARLA 0.9.16 on Windows

Launch CARLA with `Town10HD_Opt` on `127.0.0.1:2000`. Activate the CARLA Python environment, then run one controller at a time:

```powershell
.\.venv-carla\Scripts\Activate.ps1
python src/carla_integration/baseline_carla.py
python src/carla_integration/stanley_carla.py
python src/carla_integration/pure_pursuit_carla.py
python src/carla_integration/augmented_lqr_carla.py
python src/carla_integration/mpc_carla.py
```

Use unique output prefixes to avoid overwriting prior results. For a speed-specific run, invoke the exported runner with an explicit target speed:

```powershell
python -c "import sys; from pathlib import Path; sys.path.insert(0,'src'); from carla_integration import stanley_carla; s=stanley_carla.run_stanley_carla_test(target_speed_mps=10.0, file_prefix='phase12b_stanley_10ms', results_dir=Path('results')); print(s)"
```

The Stanley +1 m test uses `initial_lateral_offset_m=1.0`, with initial-position verification and spawn-clearance checks. The other four runner interfaces were not extended for offset tests in the reported work.

### Benchmark and plots from saved telemetry

```powershell
python src/carla_integration/unified_benchmark.py
python src/final_controller_analysis.py
python src/final_telemetry_comparison.py
```

**Caution:** The existing `unified_benchmark.py --live` cleanup code can remove all vehicles and sensors in the CARLA world, including actors belonging to other clients. Do not run `--live` in a shared simulator until cleanup is restricted to benchmark-owned actors. Its offline evaluation also assumes successful historical termination and zero collisions unless separately verified from run summaries.

## Repository Structure

```text
src/
  vehicle.py
  road.py
  controllers/
    pid_controller.py
    lqr_controller.py
    augmented_lqr_controller.py
    stanley_controller.py
    pure_pursuit_controller.py
    mpc_controller.py
  carla_integration/
    carla_adapter.py
    baseline_carla.py
    stanley_carla.py
    pure_pursuit_carla.py
    augmented_lqr_carla.py
    mpc_carla.py
    unified_benchmark.py
  final_controller_analysis.py
  final_telemetry_comparison.py
tests/
results/
README.md
requirements.txt
```

## Testing

The latest reported complete suite was:

```text
264 passed in 42.39s
```

Tests include mock-based checks for geometry, coordinate conversion, steering constraints, route continuity, stopping logic, controller integration, cleanup, and benchmark metrics. Passing tests does not validate every physical driving condition.

## Limitations

- **One map and route:** Broader validation requires different road geometries, conditions, and vehicle configurations.
- **Model mismatch:** Offline control design uses simplified kinematic dynamics; CARLA simulates more detailed vehicle behavior and speed-dependent steering response.
- **Steering conversion:** Nominal angle-to-normalized-CARLA conversion is not a full actuator dynamics model.
- **Reference points and signs:** Stanley's control error is front-axle-based; other controllers principally regulate rear-axle error. Signed telemetry conventions differ.
- **Discrete waypoints:** Quantization and curvature differentiation can introduce steering ripple; some integrations smooth curvature spatially.
- **Initial tuning:** Gains and cost weights were not jointly optimized across speeds and routes.
- **Recovery transient:** Stanley's +1 m test showed large oscillatory overshoot and reached the 20°/s rate limit; formal settling time and recovery distance are not yet reported.
- **MPC:** Linearized constant-nominal-speed prediction and desktop solver timing do not imply hard real-time embedded performance.
- **Scope:** Radar-based ACC, blind-spot detection, and lane-change assistance are not included.

## Project Status

**The scoped simulation-based portfolio project is complete.** Five controllers were integrated into CARLA and evaluated at three speeds, with an additional Stanley offset-recovery test, automated tests, telemetry plots, and demonstration recordings.

Optional future improvements include raw-telemetry auditing, more disturbance/noise scenarios, a fully shared multi-scenario runner, recovery-settling metrics, and a consolidated demonstration video. These are not prerequisites for describing the current project accurately as a completed simulation portfolio exercise.

**Next separate project:** Radar-based Adaptive Cruise Control, Blind Spot Detection, and Lane Change Assistance in CARLA.
