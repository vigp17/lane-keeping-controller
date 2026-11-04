from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from vehicle import Vehicle
from controllers.pid_controller import PIDController
from controllers.augmented_lqr_controller import AugmentedLQRController


# ============================================================
# 1. SIMULATION CONFIGURATION
# ============================================================

VELOCITY: float = 30.0
DT: float = 0.1
SIMULATION_TIME: float = 10.0
WHEELBASE: float = 2.8

INITIAL_X: float = 0.0
INITIAL_Y: float = 2.0
INITIAL_YAW: float = 0.0

MAX_STEERING: float = float(np.radians(30.0))
MAX_STEERING_RATE: float = float(np.radians(20.0))

KP: float = 0.05
HEADING_GAIN: float = 0.8

SETTLING_TOLERANCE: float = 0.05

STEPS: int = int(SIMULATION_TIME / DT)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. RUN CONTROLLER SIMULATION
# ============================================================

def run_simulation(
    controller_type: str
) -> dict[str, NDArray[np.float64]]:

    if controller_type not in ("Baseline", "Augmented LQR"):
        raise ValueError(
            f"Unknown controller type: {controller_type}"
        )

    # Initialize vehicle
    vehicle = Vehicle(
        x=INITIAL_X,
        y=INITIAL_Y,
        yaw=INITIAL_YAW,
        wheelbase=WHEELBASE
    )

    # Initialize controllers separately
    pid_controller = PIDController(
        kp=KP,
        ki=0.0,
        kd=0.0
    )

    lqr_controller = AugmentedLQRController(
        velocity=VELOCITY,
        wheelbase=WHEELBASE,
        dt=DT
    )

    applied_steering: float = 0.0

    # Store initial vehicle state
    x_history: list[float] = [vehicle.x]
    y_history: list[float] = [vehicle.y]
    yaw_history: list[float] = [vehicle.yaw]

    steering_history: list[float] = []
    steering_rate_history: list[float] = []

    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    for _ in range(STEPS):

        # Reference road is y = 0, yaw = 0
        lateral_error: float = vehicle.y
        heading_error: float = vehicle.yaw

        # ----------------------------------------------------
        # BASELINE CONTROLLER
        # ----------------------------------------------------

        if controller_type == "Baseline":

            # Baseline PID uses reference minus actual
            steering_command = pid_controller.compute(
                error=-lateral_error,
                dt=DT
            )

            # Heading correction
            steering_command += (
                -HEADING_GAIN * heading_error
            )

            # Steering angle constraint
            steering_command = float(
                np.clip(
                    steering_command,
                    -MAX_STEERING,
                    MAX_STEERING
                )
            )

            # Steering rate constraint
            max_change = MAX_STEERING_RATE * DT

            new_steering = float(
                np.clip(
                    steering_command,
                    applied_steering - max_change,
                    applied_steering + max_change
                )
            )

        # ----------------------------------------------------
        # AUGMENTED LQR CONTROLLER
        # ----------------------------------------------------

        else:

            # LQR outputs steering rate, not steering angle
            requested_rate = lqr_controller.compute(
                lateral_error=lateral_error,
                heading_error=heading_error,
                steering_angle=applied_steering
            )

            # Steering rate constraint
            limited_rate = float(
                np.clip(
                    requested_rate,
                    -MAX_STEERING_RATE,
                    MAX_STEERING_RATE
                )
            )

            # Integrate steering rate
            new_steering = (
                applied_steering + limited_rate * DT
            )

        # ----------------------------------------------------
        # COMMON STEERING ANGLE CONSTRAINT
        # ----------------------------------------------------

        new_steering = float(
            np.clip(
                new_steering,
                -MAX_STEERING,
                MAX_STEERING
            )
        )

        # Calculate actual applied steering rate
        actual_steering_rate = (
            new_steering - applied_steering
        ) / DT

        # Update applied steering
        applied_steering = new_steering

        # ----------------------------------------------------
        # UPDATE VEHICLE
        # ----------------------------------------------------

        vehicle.update(
            velocity=VELOCITY,
            steering_angle=applied_steering,
            dt=DT
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        x_history.append(vehicle.x)
        y_history.append(vehicle.y)
        yaw_history.append(vehicle.yaw)

        steering_history.append(applied_steering)
        steering_rate_history.append(actual_steering_rate)

    return {
        "x": np.asarray(x_history, dtype=np.float64),
        "y": np.asarray(y_history, dtype=np.float64),
        "yaw": np.asarray(yaw_history, dtype=np.float64),
        "steering": np.asarray(
            steering_history,
            dtype=np.float64
        ),
        "steering_rate": np.asarray(
            steering_rate_history,
            dtype=np.float64
        )
    }


# ============================================================
# 3. CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    results: dict[str, NDArray[np.float64]]
) -> dict[str, float | None]:

    errors = results["y"]
    steering = results["steering"]
    steering_rate = results["steering_rate"]

    # --------------------------------------------------------
    # LATERAL RMSE
    # --------------------------------------------------------

    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    # --------------------------------------------------------
    # FINAL ABSOLUTE ERROR
    # --------------------------------------------------------

    final_error = float(
        abs(errors[-1])
    )

    # --------------------------------------------------------
    # MAXIMUM ABSOLUTE ERROR
    # --------------------------------------------------------

    max_error = float(
        np.max(np.abs(errors))
    )

    # --------------------------------------------------------
    # SETTLING TIME
    # --------------------------------------------------------

    outside_tolerance = np.where(
        np.abs(errors) > SETTLING_TOLERANCE
    )[0]

    settling_time: float | None

    if len(outside_tolerance) == 0:

        settling_time = 0.0

    elif outside_tolerance[-1] == len(errors) - 1:

        settling_time = None

    else:

        settling_time = float(
            (outside_tolerance[-1] + 1) * DT
        )

    # --------------------------------------------------------
    # OVERSHOOT
    # --------------------------------------------------------

    # Initial position is above the road centerline.
    # Overshoot occurs when y becomes negative.

    minimum_y = float(np.min(errors))

    overshoot = max(
        0.0,
        -minimum_y
    )

    initial_error = abs(INITIAL_Y)

    overshoot_percentage = (
        overshoot / initial_error * 100.0
        if initial_error > 0
        else 0.0
    )

    # --------------------------------------------------------
    # MAXIMUM STEERING ANGLE
    # --------------------------------------------------------

    max_steering = float(
        np.degrees(
            np.max(np.abs(steering))
        )
    )

    # --------------------------------------------------------
    # STEERING RMS
    # --------------------------------------------------------

    steering_rms = float(
        np.degrees(
            np.sqrt(np.mean(steering ** 2))
        )
    )

    # --------------------------------------------------------
    # MAXIMUM STEERING RATE
    # --------------------------------------------------------

    max_steering_rate = float(
        np.degrees(
            np.max(np.abs(steering_rate))
        )
    )

    # --------------------------------------------------------
    # FINAL HEADING ERROR
    # --------------------------------------------------------

    final_heading_error = float(
        np.degrees(abs(results["yaw"][-1]))
    )

    return {
        "Lateral RMSE (m)": rmse,
        "Final Error (m)": final_error,
        "Max Error (m)": max_error,
        "Settling Time (s)": settling_time,
        "Overshoot (m)": overshoot,
        "Overshoot (%)": overshoot_percentage,
        "Max Steering (deg)": max_steering,
        "Steering RMS (deg)": steering_rms,
        "Max Rate (deg/s)": max_steering_rate,
        "Final Heading Error (deg)": final_heading_error
    }


# ============================================================
# 4. PRINT COMPARISON TABLE
# ============================================================

def print_comparison(
    baseline_metrics: dict[str, float | None],
    lqr_metrics: dict[str, float | None]
) -> None:

    print("\nCONTROLLER PERFORMANCE COMPARISON")
    print("=" * 75)

    print(
        f"{'Metric':<30}"
        f"{'Baseline':>20}"
        f"{'Augmented LQR':>20}"
    )

    print("-" * 75)

    for metric in baseline_metrics:

        baseline_value = baseline_metrics[metric]
        lqr_value = lqr_metrics[metric]

        baseline_text = (
            "Not settled"
            if baseline_value is None
            else f"{baseline_value:.4f}"
        )

        lqr_text = (
            "Not settled"
            if lqr_value is None
            else f"{lqr_value:.4f}"
        )

        print(
            f"{metric:<30}"
            f"{baseline_text:>20}"
            f"{lqr_text:>20}"
        )

    print("=" * 75)


# ============================================================
# 5. PLOT CONTROLLER COMPARISON
# ============================================================

def plot_comparison(
    baseline: dict[str, NDArray[np.float64]],
    lqr: dict[str, NDArray[np.float64]]
) -> None:

    time_states = np.arange(STEPS + 1) * DT
    time_controls = np.arange(STEPS) * DT

    # ========================================================
    # PLOT 1: VEHICLE TRAJECTORY
    # ========================================================

    plt.figure(figsize=(12, 6))

    plt.plot(
        baseline["x"],
        baseline["y"],
        linewidth=2,
        label="Baseline Controller"
    )

    plt.plot(
        lqr["x"],
        lqr["y"],
        linewidth=2,
        label="Augmented LQR"
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        label="Reference Road"
    )

    plt.scatter(
        INITIAL_X,
        INITIAL_Y,
        color="green",
        s=80,
        label="Start"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Lane Keeping - Trajectory Comparison")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "comparison_trajectory.png",
        dpi=300
    )

    # ========================================================
    # PLOT 2: LATERAL ERROR VS TIME
    # ========================================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        time_states,
        baseline["y"],
        linewidth=2,
        label="Baseline Controller"
    )

    plt.plot(
        time_states,
        lqr["y"],
        linewidth=2,
        label="Augmented LQR"
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        label="Zero Error"
    )

    plt.axhline(
        y=SETTLING_TOLERANCE,
        color="gray",
        linestyle=":",
        label="Settling Tolerance"
    )

    plt.axhline(
        y=-SETTLING_TOLERANCE,
        color="gray",
        linestyle=":"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Lateral Error (m)")
    plt.title("Lateral Tracking Error Comparison")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "comparison_lateral_error.png",
        dpi=300
    )

    # ========================================================
    # PLOT 3: STEERING ANGLE
    # ========================================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        time_controls,
        np.degrees(baseline["steering"]),
        linewidth=2,
        label="Baseline Controller"
    )

    plt.plot(
        time_controls,
        np.degrees(lqr["steering"]),
        linewidth=2,
        label="Augmented LQR"
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle (degrees)")
    plt.title("Steering Angle Comparison")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "comparison_steering_angle.png",
        dpi=300
    )

    # ========================================================
    # PLOT 4: STEERING RATE
    # ========================================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        time_controls,
        np.degrees(baseline["steering_rate"]),
        linewidth=2,
        label="Baseline Controller"
    )

    plt.plot(
        time_controls,
        np.degrees(lqr["steering_rate"]),
        linewidth=2,
        label="Augmented LQR"
    )

    plt.axhline(
        y=20.0,
        color="red",
        linestyle="--",
        label="Upper Rate Limit"
    )

    plt.axhline(
        y=-20.0,
        color="red",
        linestyle="--",
        label="Lower Rate Limit"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Rate (degrees/s)")
    plt.title("Steering Rate Comparison")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "comparison_steering_rate.png",
        dpi=300
    )

    # Display all four figures
    plt.show()


# ============================================================
# 6. MAIN FUNCTION
# ============================================================

def main() -> None:

    print("Running Baseline Controller...")

    baseline_results = run_simulation(
        controller_type="Baseline"
    )

    print("Running Augmented LQR Controller...")

    lqr_results = run_simulation(
        controller_type="Augmented LQR"
    )

    # Calculate performance metrics
    baseline_metrics = calculate_metrics(
        baseline_results
    )

    lqr_metrics = calculate_metrics(
        lqr_results
    )

    # Print comparison table
    print_comparison(
        baseline_metrics,
        lqr_metrics
    )

    # Generate comparison plots
    plot_comparison(
        baseline_results,
        lqr_results
    )


# ============================================================
# 7. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()