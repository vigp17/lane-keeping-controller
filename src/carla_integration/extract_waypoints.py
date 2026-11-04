import carla
import math
import time


# ==================================================
# 1. CONNECT TO CARLA
# ==================================================

def main() -> None:

    client = carla.Client(
        "127.0.0.1",
        2000
    )

    client.set_timeout(30.0)

    world = client.get_world()

    carla_map = world.get_map()

    print("Connected to CARLA!")
    print(f"Map: {carla_map.name}")

    # ==============================================
    # 2. GET SPAWN POINTS
    # ==============================================

    spawn_points = carla_map.get_spawn_points()

    if not spawn_points:
        raise RuntimeError(
            "No spawn points available."
        )

    # Select first spawn point
    start_location = spawn_points[0].location

    # ==============================================
    # 3. FIND STARTING ROAD WAYPOINT
    # ==============================================

    waypoint = carla_map.get_waypoint(
        start_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if waypoint is None:
        raise RuntimeError(
            "No driving waypoint found."
        )

    # ==============================================
    # 4. GENERATE REFERENCE WAYPOINTS
    # ==============================================

    waypoints = [waypoint]

    spacing: float = 2.0

    max_points: int = 100

    for _ in range(max_points - 1):

        next_waypoints = waypoint.next(spacing)

        if not next_waypoints:
            break

        # Select first continuation
        waypoint = next_waypoints[0]

        waypoints.append(waypoint)

    print(
        f"\nSuccessfully extracted "
        f"{len(waypoints)} waypoints."
    )

    # ==============================================
    # 5. PRINT FIRST 10 WAYPOINTS
    # ==============================================

    print("\nFirst 10 Waypoints:")

    for i, wp in enumerate(waypoints[:10]):

        location = wp.transform.location

        yaw = math.radians(
            wp.transform.rotation.yaw
        )

        print(
            f"Waypoint {i:02d} | "
            f"X={location.x:.2f} m | "
            f"Y={location.y:.2f} m | "
            f"Yaw={yaw:.3f} rad"
        )

    # ==============================================
    # 6. DRAW GREEN WAYPOINT MARKERS
    # ==============================================

    print("\nDrawing green waypoint markers...")

    for wp in waypoints:

        location = wp.transform.location

        # Raise markers above the road
        marker_location = carla.Location(
            x=location.x,
            y=location.y,
            z=location.z + 1.0
        )

        world.debug.draw_point(
            marker_location,
            size=0.3,
            color=carla.Color(
                r=0,
                g=255,
                b=0
            ),
            life_time=60.0
        )

    # ==============================================
    # 7. MOVE SPECTATOR CAMERA
    # ==============================================

    spectator = world.get_spectator()

    first_waypoint = waypoints[0]

    first_location = first_waypoint.transform.location

    # Position camera 30 meters above the road
    camera_location = carla.Location(
        x=first_location.x,
        y=first_location.y,
        z=first_location.z + 30.0
    )

    camera_rotation = carla.Rotation(
        pitch=-90.0,
        yaw=0.0,
        roll=0.0
    )

    spectator.set_transform(
        carla.Transform(
            camera_location,
            camera_rotation
        )
    )

    print("\nSpectator camera moved above the road.")

    # ==============================================
    # 8. KEEP MARKERS VISIBLE
    # ==============================================

    print("\nGreen markers will remain for 60 seconds.")
    print("Switch to the CARLA window to view them.")

    time.sleep(60)

    print("\nWaypoint visualization complete.")


# ==================================================
# 9. PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()