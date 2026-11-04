import numpy as np
import matplotlib.pyplot as plt

from vehicle import Vehicle
from road import Road
from controllers.pid_controller import PIDController


# ==========================================
# 1. Initialize Vehicle and Road
# ==========================================

vehicle = Vehicle(
    x=0.0,
    y=2.0,
    yaw=0.0
)

road = Road(length=320.0)

road_x, road_y = road.generate_straight_road()


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

# Maximum steering angle: 30 degrees
max_steering: float = float(np.radians(30.0))

# NEW: Maximum steering rate: 20 degrees/second
max_steering_rate: float = float(np.radians(20.0))

# Initial applied steering angle
previous_steering: float = 0.0


# ==========================================
# 4. Initialize Controller
# ==========================================

pid = PIDController(
    kp=0.05,
    ki=0.0,
    kd=0.0
)

heading_gain: float = 0.8


# ==========================================
# 5. Store Simulation Results
# ==========================================

x_history: list[float] = [vehicle.x]
y_history: list[float] = [vehicle.y]

error_history: list[float] = []
steering_history: list[float] = []


# ==========================================
# 6. Closed-Loop Simulation
# ==========================================

for i in range(steps):

    # --------------------------------------
    # Calculate tracking errors
    # --------------------------------------

    y_reference: float = 0.0

    lateral_error: float = y_reference - vehicle.y

    heading_error: float = -vehicle.yaw


    # --------------------------------------
    # Calculate controller output
    # --------------------------------------

    lateral_correction = pid.compute(
        lateral_error,
        dt
    )

    heading_correction = heading_gain * heading_error

    steering_command = (
        lateral_correction + heading_correction
    )


    # --------------------------------------
    # Apply steering angle constraint
    # --------------------------------------

    steering_command = float(
        np.clip(
            steering_command,
            -max_steering,
            max_steering
        )
    )


    # ======================================
    # NEW: Apply Steering Rate Constraint
    # ======================================

    # Maximum steering change per timestep
    max_delta = max_steering_rate * dt

    # Limit steering change relative to
    # previously applied steering angle
    steering_angle = float(
        np.clip(
            steering_command,
            previous_steering - max_delta,
            previous_steering + max_delta
        )
    )

    # Store applied steering for next timestep
    previous_steering = steering_angle


    # --------------------------------------
    # Update vehicle dynamics
    # --------------------------------------

    vehicle.update(
        velocity,
        steering_angle,
        dt
    )


    # --------------------------------------
    # Store simulation results
    # --------------------------------------

    x_history.append(vehicle.x)
    y_history.append(vehicle.y)

    error_history.append(lateral_error)
    steering_history.append(steering_angle)


# ==========================================
# 7. Calculate Tracking Performance Metrics
# ==========================================

error_array = np.array(error_history)

# Root Mean Square Error
rmse = float(
    np.sqrt(np.mean(error_array ** 2))
)

# Maximum absolute lateral error
max_error = float(
    np.max(np.abs(error_array))
)

# Final absolute lateral error
final_error = abs(vehicle.y)


# ==========================================
# 8. Calculate Settling Time
# ==========================================

tolerance: float = 0.05

# Include the final state when checking settling
error_samples = np.array(
    [0.0 - y for y in y_history]
)

outside_tolerance = np.where(
    np.abs(error_samples) > tolerance
)[0]

if len(outside_tolerance) == 0:

    settling_time = 0.0

elif outside_tolerance[-1] == len(error_samples) - 1:

    settling_time = None

else:

    settling_time = (
        outside_tolerance[-1] + 1
    ) * dt


# ==========================================
# 9. Calculate Overshoot
# ==========================================

y_array = np.array(y_history)

minimum_y = float(np.min(y_array))

# Vehicle starts above the road centerline
overshoot = max(0.0, -minimum_y)

initial_error = abs(y_history[0])

overshoot_percentage = (
    overshoot / initial_error
) * 100.0 if initial_error > 0 else 0.0


# ==========================================
# 10. Steering Performance Metrics
# ==========================================

steering_array = np.array(steering_history)

# Maximum absolute steering angle
max_steering_angle = float(
    np.max(np.abs(steering_array))
)

# RMS steering angle
steering_rms = float(
    np.sqrt(np.mean(steering_array ** 2))
)


# ==========================================
# NEW: Calculate Steering Rate
# ==========================================

# Include initial steering change from zero
steering_with_initial = np.concatenate(
    ([0.0], steering_array)
)

steering_rate = np.diff(
    steering_with_initial
) / dt

max_measured_steering_rate = float(
    np.max(np.abs(steering_rate))
)


# ==========================================
# 11. Print Performance Metrics
# ==========================================

print("\nController Performance Metrics")

print(f"Lateral RMSE: {rmse:.4f} m")
print(f"Maximum Lateral Error: {max_error:.4f} m")
print(f"Final Absolute Error: {final_error:.4f} m")

if settling_time is None:
    print("Settling Time: Not settled within simulation")
else:
    print(f"Settling Time: {settling_time:.2f} s")


print("\nOvershoot Metrics")

print(f"Maximum Overshoot: {overshoot:.4f} m")
print(f"Overshoot Percentage: {overshoot_percentage:.2f}%")


print("\nSteering Performance Metrics")

print(
    f"Maximum Steering Angle: "
    f"{np.degrees(max_steering_angle):.4f} deg"
)

print(
    f"Steering RMS: "
    f"{np.degrees(steering_rms):.4f} deg"
)

print(
    f"Maximum Steering Rate: "
    f"{np.degrees(max_measured_steering_rate):.4f} deg/s"
)


print("\nClosed-Loop Lane Keeping Results")

print(f"Final X Position: {vehicle.x:.2f} m")
print(f"Final Y Position: {vehicle.y:.2f} m")
print(f"Final Yaw Angle: {vehicle.yaw:.4f} rad")
print(f"Final Lateral Error: {-vehicle.y:.4f} m")


# ==========================================
# 12. Plot Vehicle Trajectory
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    road_x,
    road_y,
    "k--",
    linewidth=2,
    label="Reference Road"
)

plt.plot(
    x_history,
    y_history,
    "b-",
    linewidth=2,
    label="Controlled Vehicle"
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
plt.title("Closed-Loop Lane Keeping - Steering Rate Limited")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/closed_loop_trajectory.png",
    dpi=300
)


# ==========================================
# 13. Plot Steering Angle vs Time
# ==========================================

time_history = np.arange(steps) * dt

steering_degrees = np.degrees(steering_array)

plt.figure(figsize=(10, 5))

plt.plot(
    time_history,
    steering_degrees,
    color="purple",
    linewidth=2,
    label="Applied Steering Angle"
)

plt.axhline(
    y=0,
    color="black",
    linestyle="--",
    label="Zero Steering"
)

plt.xlabel("Time (s)")
plt.ylabel("Steering Angle (degrees)")
plt.title("Steering Angle vs Time")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/steering_angle.png",
    dpi=300
)


# ==========================================
# 14. Plot Steering Rate vs Time
# ==========================================

steering_rate_degrees = np.degrees(steering_rate)

plt.figure(figsize=(10, 5))

plt.plot(
    time_history,
    steering_rate_degrees,
    color="orange",
    linewidth=2,
    label="Applied Steering Rate"
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
plt.title("Steering Rate vs Time")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/steering_rate.png",
    dpi=300
)


# ==========================================
# 15. Display All Graphs
# ==========================================

plt.show()