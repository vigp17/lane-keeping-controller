import numpy as np
import matplotlib.pyplot as plt

from vehicle import Vehicle
from road import Road
from controllers.lqr_controller import LQRController


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
# 2. Initialize Road
# ==========================================

road = Road(length=320.0)

road_x, road_y = road.generate_straight_road()


# ==========================================
# 3. Simulation Parameters
# ==========================================

velocity: float = 30.0

dt: float = 0.1

simulation_time: float = 10.0

steps = int(simulation_time / dt)


# ==========================================
# 4. Steering Constraints
# ==========================================

max_steering = float(np.radians(30.0))

max_steering_rate = float(np.radians(20.0))

previous_steering = 0.0


# ==========================================
# 5. Initialize LQR Controller
# ==========================================

lqr = LQRController(
    velocity=velocity,
    wheelbase=vehicle.wheelbase,
    dt=dt
)


# ==========================================
# 6. Store Simulation Results
# ==========================================

x_history = [vehicle.x]
y_history = [vehicle.y]

error_history = []
steering_history = []

# ==========================================
# 7. Run LQR Closed-Loop Simulation
# ==========================================

for i in range(steps):

    # --------------------------------------
    # Calculate tracking errors
    # --------------------------------------

    y_reference: float = 0.0
    yaw_reference: float = 0.0

    # LQR uses actual minus reference
    lateral_error: float = vehicle.y - y_reference

    heading_error: float = vehicle.yaw - yaw_reference

    # --------------------------------------
    # Calculate LQR steering command
    # --------------------------------------

    steering_command: float = lqr.compute(
        lateral_error,
        heading_error
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

    # --------------------------------------
    # Apply steering rate constraint
    # --------------------------------------

    max_delta = max_steering_rate * dt

    steering_angle = float(
        np.clip(
            steering_command,
            previous_steering - max_delta,
            previous_steering + max_delta
        )
    )

    # Store applied steering
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
    # Store results
    # --------------------------------------

    x_history.append(vehicle.x)
    y_history.append(vehicle.y)

    error_history.append(lateral_error)
    steering_history.append(steering_angle)

        # ==========================================
    # Debug First 15 Simulation Steps
    # ==========================================

    if i < 15:
        print(
            f"Step {i:02d} | "
            f"Y = {vehicle.y:8.3f} m | "
            f"Yaw = {np.degrees(vehicle.yaw):8.2f} deg | "
            f"LQR Command = {np.degrees(steering_command):8.2f} deg | "
            f"Applied Steering = {np.degrees(steering_angle):8.2f} deg"
        )


# ==========================================
# 8. Print LQR Simulation Results
# ==========================================

print("\nLQR Lane Keeping Results")

print(f"Final X Position: {vehicle.x:.2f} m")

print(f"Final Y Position: {vehicle.y:.4f} m")

print(f"Final Yaw Angle: {vehicle.yaw:.4f} rad")

print(f"Final Lateral Error: {vehicle.y:.4f} m")

print(
    f"Maximum Applied Steering: "
    f"{np.degrees(np.max(np.abs(steering_history))):.4f} deg"
)