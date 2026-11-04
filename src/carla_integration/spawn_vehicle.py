import carla
import math
import random
import time

from typing import cast


# ==================================================
# 1. UPDATE SPECTATOR CAMERA
# ==================================================

def update_spectator_camera(
    world: carla.World,
    vehicle: carla.Vehicle
) -> None:

    """
    Position the spectator camera behind and above
    the vehicle, pointing directly toward it.
    """

    # Get vehicle transform
    transform = vehicle.get_transform()
    location = transform.location

    # Get vehicle forward direction
    forward = transform.get_forward_vector()

    # Position camera behind and above vehicle
    camera_location = carla.Location(
        x=location.x - 8.0 * forward.x,
        y=location.y - 8.0 * forward.y,
        z=location.z + 5.0
    )

    # Calculate direction toward vehicle
    dx = location.x - camera_location.x
    dy = location.y - camera_location.y
    dz = location.z - camera_location.z

    # Calculate camera yaw
    camera_yaw = math.degrees(
        math.atan2(dy, dx)
    )

    # Calculate camera pitch
    horizontal_distance = math.hypot(dx, dy)

    camera_pitch = math.degrees(
        math.atan2(
            dz,
            horizontal_distance
        )
    )

    # Get spectator
    spectator = world.get_spectator()

    # Apply spectator transform
    spectator.set_transform(
        carla.Transform(
            camera_location,
            carla.Rotation(
                pitch=camera_pitch,
                yaw=camera_yaw,
                roll=0.0
            )
        )
    )


# ==================================================
# 2. MAIN FUNCTION
# ==================================================

def main() -> None:

    # ==============================================
    # Connect to CARLA
    # ==============================================

    client = carla.Client(
        "127.0.0.1",
        2000
    )

    client.set_timeout(30.0)

    world = client.get_world()

    print("Connected to CARLA!")
    print(f"Map: {world.get_map().name}")

    # ==============================================
    # Select Tesla Model 3
    # ==============================================

    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.find(
        "vehicle.tesla.model3"
    )

    # ==============================================
    # Get Spawn Points
    # ==============================================

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError(
            "No spawn points available."
        )

    random.shuffle(spawn_points)

    # Explicitly declare vehicle type
    vehicle: carla.Vehicle | None = None

    try:

        # ==========================================
        # Spawn Vehicle
        # ==========================================

        for spawn_point in spawn_points:

            spawned_actor = world.try_spawn_actor(
                vehicle_bp,
                spawn_point
            )

            if spawned_actor is not None:

                vehicle = cast(
                    carla.Vehicle,
                    spawned_actor
                )

                break

        if vehicle is None:
            raise RuntimeError(
                "Unable to spawn Tesla Model 3."
            )

        print("\nVehicle spawned successfully!")

        print(f"Vehicle ID: {vehicle.id}")

        print(f"Vehicle Type: {vehicle.type_id}")

        # ==========================================
        # Disable Autopilot
        # ==========================================

        vehicle.set_autopilot(False)

        # ==========================================
        # Apply Handbrake
        # ==========================================

        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=0.0,
                hand_brake=True
            )
        )

        # ==========================================
        # Print Initial Vehicle State
        # ==========================================

        transform = vehicle.get_transform()
        location = transform.location

        print("\nInitial Vehicle Position:")

        print(f"X: {location.x:.2f} m")
        print(f"Y: {location.y:.2f} m")
        print(f"Z: {location.z:.2f} m")

        print(
            f"Yaw: {transform.rotation.yaw:.2f} degrees"
        )

        # ==========================================
        # Continuously Follow Vehicle
        # ==========================================

        print("\nSpectator camera following vehicle.")

        print("Switch to the CARLA window.")

        print("Press Ctrl+C to stop early.")

        # 600 iterations x 0.1 seconds = 60 seconds

        for i in range(600):

            if not vehicle.is_alive:
                print("Vehicle no longer exists.")
                break

            # Update camera
            update_spectator_camera(
                world,
                vehicle
            )

            # Print position every 5 seconds
            if i % 50 == 0:

                current_transform = (
                    vehicle.get_transform()
                )

                current_location = (
                    current_transform.location
                )

                print(
                    f"Time: {i * 0.1:.1f} s | "
                    f"X: {current_location.x:.2f} m | "
                    f"Y: {current_location.y:.2f} m | "
                    f"Z: {current_location.z:.2f} m"
                )

            time.sleep(0.1)

    except KeyboardInterrupt:

        print("\nSimulation stopped by user.")

    finally:

        # ==========================================
        # Clean Up Vehicle
        # ==========================================

        if vehicle is not None and vehicle.is_alive:

            vehicle.destroy()

            print("\nVehicle destroyed successfully.")


# ==================================================
# 3. PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()