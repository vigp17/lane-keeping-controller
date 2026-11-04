import numpy as np
import matplotlib.pyplot as plt

from vehicle import Vehicle
from controllers.augmented_lqr_controller import AugmentedLQRController


# ==========================================
# 1. Initialize Vehicle
# ==========================================

vehicle = Vehicle(
    x=0.0,
    y=2.0,
    yaw=0.0,
    wheelbase=2.8
)


# ==========================================
# 2. Simulation Parameters
# ==========================================

velocity: float = 30.0
dt: float = 0.1
simulation_time: float = 10.0

steps = int(simulation_time / dt)


# ==========================================
# 3. Steering Constraints
# ==========================================

max_steering = float(np.radians(30.0))
max_steering_rate = float(np.radians(20.0))

# Actual applied steering angle
applied_steering = 0.0


# ==========================================
# 4. Initialize Augmented LQR
# ==========================================

controller = AugmentedLQRController(
    velocity=velocity,
    wheelbase=vehicle.wheelbase,
    dt=dt
)


# ==========================================
# 5. Store Results
# ==========================================

x_history = [vehicle.x]
y_history = [vehicle.y]

error_history = []
steering_history = []
steering_rate_history = []


# ==========================================
# 6. Run Closed-Loop Simulation
# ==========================================

for i in range(steps):

    # Calculate tracking errors
    lateral_error = vehicle.y
    heading_error = vehicle.yaw

    # LQR calculates steering RATE
    requested_rate = controller.compute(
        lateral_error=lateral_error,
        heading_error=heading_error,
        steering_angle=applied_steering
    )

    # Apply steering-rate limit
    applied_rate = float(
        np.clip(
            requested_rate,
            -max_steering_rate,
            max_steering_rate
        )
    )

    # Integrate steering rate
    new_steering = applied_steering + applied_rate * dt

    # Apply steering-angle limit
    new_steering = float(
        np.clip(
            new_steering,
            -max_steering,
            max_steering
        )
    )

    # Calculate actual applied steering rate
    actual_rate = (new_steering - applied_steering) / dt

    # Update applied steering
    applied_steering = new_steering

    # Update vehicle dynamics
    vehicle.update(
        velocity,
        applied_steering,
        dt
    )

    # Store results
    x_history.append(vehicle.x)
    y_history.append(vehicle.y)

    error_history.append(lateral_error)
    steering_history.append(applied_steering)
    steering_rate_history.append(actual_rate)


# ==========================================
# 7. Calculate Performance Metrics
# ==========================================

error_array = np.array(y_history)
steering_array = np.array(steering_history)
rate_array = np.array(steering_rate_history)

rmse = float(np.sqrt(np.mean(error_array**2)))
final_error = abs(vehicle.y)

max_steering_used = float(np.max(np.abs(steering_array)))
max_rate_used = float(np.max(np.abs(rate_array)))

# ==========================================
# Calculate Settling Time
# ==========================================

tolerance = 0.05  # 5 cm

outside_tolerance = np.where(
    np.abs(error_array) > tolerance
)[0]

if len(outside_tolerance) == 0:
    settling_time = 0.0

elif outside_tolerance[-1] == len(error_array) - 1:
    settling_time = None

else:
    settling_time = (
        outside_tolerance[-1] + 1
    ) * dt


# ==========================================
# Calculate Overshoot
# ==========================================

minimum_y = float(np.min(error_array))

overshoot = max(0.0, -minimum_y)

initial_error = abs(y_history[0])

overshoot_percentage = (
    overshoot / initial_error * 100.0
    if initial_error > 0 else 0.0
)


# ==========================================
# Print Additional Metrics
# ==========================================

print("\nAdditional LQR Metrics")

if settling_time is None:
    print("Settling Time: Not settled within simulation")
else:
    print(f"Settling Time: {settling_time:.2f} s")

print(f"Maximum Overshoot: {overshoot:.4f} m")
print(f"Overshoot Percentage: {overshoot_percentage:.2f}%")


# ==========================================
# 8. Print Results
# ==========================================

print("\nAugmented LQR Performance")

print(f"Lateral RMSE: {rmse:.4f} m")
print(f"Final Absolute Error: {final_error:.4f} m")

print(
    f"Maximum Steering Angle: "
    f"{np.degrees(max_steering_used):.4f} deg"
)

print(
    f"Maximum Steering Rate: "
    f"{np.degrees(max_rate_used):.4f} deg/s"
)

print(f"Final X Position: {vehicle.x:.2f} m")
print(f"Final Y Position: {vehicle.y:.4f} m")
print(f"Final Yaw Angle: {vehicle.yaw:.4f} rad")

# ==========================================
# 9. Plot Augmented LQR Results
# ==========================================

from pathlib import Path

# Ensure results directory exists
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Generate time arrays
time_states = np.arange(steps + 1) * dt
time_controls = np.arange(steps) * dt


# ==========================================
# Plot 1: Vehicle Trajectory
# ==========================================

plt.figure(figsize=(10, 6))

# Reference road
plt.axhline(
    y=0,
    color="black",
    linestyle="--",
    label="Reference Road"
)

# Vehicle trajectory
plt.plot(
    x_history,
    y_history,
    color="blue",
    linewidth=2,
    label="Augmented LQR"
)

plt.scatter(
    x_history[0],
    y_history[0],
    color="green",
    s=80,
    label="Start"
)

plt.scatter(
    x_history[-1],
    y_history[-1],
    color="red",
    s=80,
    label="End"
)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Augmented LQR - Vehicle Trajectory")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    results_dir / "augmented_lqr_trajectory.png",
    dpi=300
)


# ==========================================
# Plot 2: Steering Angle
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    time_controls,
    np.degrees(steering_history),
    color="purple",
    linewidth=2,
    label="Steering Angle"
)

plt.axhline(
    y=0,
    color="black",
    linestyle="--"
)

plt.xlabel("Time (s)")
plt.ylabel("Steering Angle (degrees)")
plt.title("Augmented LQR - Steering Angle")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    results_dir / "augmented_lqr_steering.png",
    dpi=300
)


# ==========================================
# Plot 3: Steering Rate
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    time_controls,
    np.degrees(steering_rate_history),
    color="orange",
    linewidth=2,
    label="Steering Rate"
)

plt.axhline(
    y=20,
    color="red",
    linestyle="--",
    label="Upper Limit"
)

plt.axhline(
    y=-20,
    color="red",
    linestyle="--",
    label="Lower Limit"
)

plt.xlabel("Time (s)")
plt.ylabel("Steering Rate (degrees/s)")
plt.title("Augmented LQR - Steering Rate")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    results_dir / "augmented_lqr_steering_rate.png",
    dpi=300
)

plt.show()