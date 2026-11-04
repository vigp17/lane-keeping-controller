import numpy as np
import matplotlib.pyplot as plt

from road import Road
from vehicle import Vehicle


# ==========================================
# 1. Initialize Road
# ==========================================

road = Road(length=100.0)

road_x, road_y = road.generate_straight_road()


# ==========================================
# 2. Initialize Vehicle
# ==========================================

vehicle = Vehicle()


# ==========================================
# 3. Simulation Parameters
# ==========================================

velocity: float = 10.0

steering_angle: float = float(np.radians(10.0))

dt: float = 0.1

simulation_time: float = 5.0

steps = int(simulation_time / dt)


# ==========================================
# 4. Store Vehicle Trajectory and Error
# ==========================================

x_history = [vehicle.x]
y_history = [vehicle.y]

# NEW: Store lateral tracking errors
lateral_error_history: list[float] = []


# ==========================================
# 5. Run Simulation
# ==========================================

for i in range(steps):

    # Update vehicle position and heading
    vehicle.update(
        velocity,
        steering_angle,
        dt
    )

    # Store vehicle position
    x_history.append(vehicle.x)
    y_history.append(vehicle.y)

    # --------------------------------------
    # NEW: Calculate lateral tracking error
    # --------------------------------------

    # Straight road centerline
    y_reference: float = 0.0

    # Difference between desired and actual Y
    lateral_error: float = y_reference - vehicle.y

    # Store error for future analysis
    lateral_error_history.append(lateral_error)


# ==========================================
# 6. Print Simulation Results
# ==========================================

print("Vehicle Final Position")

print(f"X Position: {vehicle.x:.2f} m")
print(f"Y Position: {vehicle.y:.2f} m")
print(f"Yaw Angle: {vehicle.yaw:.2f} rad")

# NEW: Print final lateral tracking error
print(f"Lateral Tracking Error: {lateral_error_history[-1]:.2f} m")


# ==========================================
# 7. Plot Vehicle Trajectory
# ==========================================

plt.figure(figsize=(10, 6))


# Plot reference road
plt.plot(
    road_x,
    road_y,
    "k--",
    linewidth=2,
    label="Reference Road"
)


# Plot vehicle trajectory
plt.plot(
    x_history,
    y_history,
    color="blue",
    linewidth=2,
    label="Vehicle Trajectory"
)


# Plot starting position
plt.scatter(
    x_history[0],
    y_history[0],
    color="green",
    s=80,
    label="Start"
)


# Plot ending position
plt.scatter(
    x_history[-1],
    y_history[-1],
    color="red",
    s=80,
    label="End"
)


# ==========================================
# 8. Configure Graph
# ==========================================

plt.xlabel("X Position (m)")

plt.ylabel("Y Position (m)")

plt.title("Vehicle Trajectory vs Reference Road")

plt.axis("equal")

plt.grid(True)

plt.legend()

plt.tight_layout()


# ==========================================
# 9. Save and Display Graph
# ==========================================

plt.savefig(
    "results/vehicle_trajectory.png",
    dpi=300
)


# ==========================================
# 10. Plot Lateral Tracking Error
# ==========================================

# Generate time values
time_history = np.arange(1, steps + 1) * dt

# Create figure
plt.figure(figsize=(10, 5))

# Plot lateral error
plt.plot(
    time_history,
    lateral_error_history,
    color="red",
    linewidth=2,
    label="Lateral Tracking Error"
)

# Zero-error reference
plt.axhline(
    y=0,
    color="black",
    linestyle="--",
    label="Zero Error"
)

# Configure graph
plt.xlabel("Time (s)")
plt.ylabel("Lateral Error (m)")
plt.title("Lateral Tracking Error vs Time")

plt.grid(True)
plt.legend()
plt.tight_layout()

# Save graph
plt.savefig(
    "results/lateral_error.png",
    dpi=300
)

# Display graph
plt.show()