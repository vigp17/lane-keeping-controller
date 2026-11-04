import carla
import math
import time
from typing import cast


def main() -> None:

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)

    world = client.get_world()

    vehicle_bp = world.get_blueprint_library().find(
        "vehicle.tesla.model3"
    )

    spawn_points = world.get_map().get_spawn_points()

    vehicle: carla.Vehicle | None = None

    try:
        for spawn_point in spawn_points:

            actor = world.try_spawn_actor(
                vehicle_bp,
                spawn_point
            )

            if actor is not None:
                vehicle = cast(carla.Vehicle, actor)
                break

        if vehicle is None:
            raise RuntimeError("Unable to spawn vehicle")

        vehicle.set_autopilot(False)

        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=0.0,
                steer=0.0,
                hand_brake=True
            )
        )

        print("Vehicle spawned successfully!")

        for i in range(10):

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()

            x = transform.location.x
            y = transform.location.y
            z = transform.location.z

            yaw = math.radians(transform.rotation.yaw)

            speed = math.sqrt(
                velocity.x**2
                + velocity.y**2
                + velocity.z**2
            )

            print(
                f"Step {i:02d} | "
                f"X={x:.2f} m | "
                f"Y={y:.2f} m | "
                f"Z={z:.2f} m | "
                f"Yaw={yaw:.3f} rad | "
                f"Speed={speed:.2f} m/s"
            )

            time.sleep(0.5)

    finally:
        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()
            print("Vehicle destroyed.")


if __name__ == "__main__":
    main()