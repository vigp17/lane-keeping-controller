from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import cast

import carla


# Ensure src directory is on sys.path so carla_integration can be imported
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from carla_integration.carla_adapter import (
    CarlaControllerAdapter,
    carla_to_controller_position,
    carla_to_controller_yaw,
    estimate_rear_axle_position,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

CARLA_HOST: str = "127.0.0.1"
CARLA_PORT: int = 2000
CLIENT_TIMEOUT: float = 30.0
VEHICLE_BLUEPRINT_ID: str = "vehicle.tesla.model3"

TEST_STEERING_ANGLES_DEG: tuple[float, ...] = (-5.0, 0.0, 5.0)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def spawn_stationary_tesla_at_driving_lane(
    world: carla.World
) -> carla.Vehicle:
    """
    Spawn a Tesla Model 3 at an available driving-lane spawn point,
    disable autopilot, engage the handbrake, and allow physics to settle.
    """
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points available in the active CARLA map.")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(VEHICLE_BLUEPRINT_ID)

    for sp in spawn_points:
        waypoint = carla_map.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            continue

        spawned_actor = world.try_spawn_actor(vehicle_bp, sp)
        if spawned_actor is not None:
            vehicle = cast(carla.Vehicle, spawned_actor)
            vehicle.set_autopilot(False)
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                    hand_brake=True,
                    reverse=False
                )
            )
            # Allow vehicle suspension to settle on the road surface
            time.sleep(0.5)
            return vehicle

    raise RuntimeError(
        "Unable to spawn Tesla Model 3 at any driving-lane spawn point."
    )


def format_steering_curve(
    physics_control: carla.VehiclePhysicsControl
) -> list[tuple[float, float]]:
    """
    Extract (speed_kmh_or_mps, steering_multiplier) control points from
    physics_control.steering_curve.
    """
    curve_points: list[tuple[float, float]] = []
    for pt in physics_control.steering_curve:
        curve_points.append((float(pt.x), float(pt.y)))
    return curve_points


# ==================================================
# 3. MAIN VALIDATION PROCEDURE
# ==================================================

def main() -> None:
    """
    Connect to CARLA 0.9.16, spawn a stationary Tesla Model 3, inspect its
    live physics/wheel geometry, and validate CarlaControllerAdapter.
    """
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CLIENT_TIMEOUT)

    world = client.get_world()
    carla_map = world.get_map()

    print("=" * 85)
    print("CARLA 0.9.16 ADAPTER LIVE VALIDATION")
    print("=" * 85)
    print(f"Connected to CARLA at {CARLA_HOST}:{CARLA_PORT}")
    print(f"Active Map: {carla_map.name}")

    vehicle: carla.Vehicle | None = None

    try:
        # 1. Spawn stationary Tesla Model 3 with handbrake engaged
        vehicle = spawn_stationary_tesla_at_driving_lane(world)

        # Keep vehicle stationary with handbrake enabled
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True
            )
        )

        # 2. Retrieve live transform and physics control
        transform = vehicle.get_transform()
        actor_loc = transform.location
        actor_rot = transform.rotation
        physics_control = vehicle.get_physics_control()

        # 3. Initialize CarlaControllerAdapter
        adapter = CarlaControllerAdapter(vehicle)
        geom = adapter.geometry
        ctrl_state = adapter.get_controller_state()

        # Compute direct front and rear axle world locations from raw CARLA wheels
        # (CARLA WheelPhysicsControl.position is in world centimeters)
        wheels = list(physics_control.wheels)
        fl_w, fr_w, rl_w, rr_w = wheels[0], wheels[1], wheels[2], wheels[3]

        front_axle_carla_x = 0.5 * (fl_w.position.x + fr_w.position.x) / 100.0
        front_axle_carla_y = 0.5 * (fl_w.position.y + fr_w.position.y) / 100.0
        front_axle_carla_z = 0.5 * (fl_w.position.z + fr_w.position.z) / 100.0

        rear_axle_carla_x = 0.5 * (rl_w.position.x + rr_w.position.x) / 100.0
        rear_axle_carla_y = 0.5 * (rl_w.position.y + rr_w.position.y) / 100.0
        rear_axle_carla_z = 0.5 * (rl_w.position.z + rr_w.position.z) / 100.0

        front_axle_ctrl_x, front_axle_ctrl_y = carla_to_controller_position(
            front_axle_carla_x,
            front_axle_carla_y
        )
        direct_rear_ctrl_x, direct_rear_ctrl_y = carla_to_controller_position(
            rear_axle_carla_x,
            rear_axle_carla_y
        )

        # 4. Print Vehicle Actor State and Physics Geometry
        print("\n1. LIVE VEHICLE ACTOR STATE & PHYSICS GEOMETRY")
        print("-" * 85)
        print(f"   Vehicle Blueprint            : {vehicle.type_id} (ID: {vehicle.id})")
        print(
            f"   Actor Location (CARLA)       : "
            f"X = {actor_loc.x:.4f} m, Y = {actor_loc.y:.4f} m, Z = {actor_loc.z:.4f} m"
        )
        print(
            f"   Actor Rotation (CARLA)       : "
            f"Yaw = {actor_rot.yaw:.4f} deg, Pitch = {actor_rot.pitch:.4f} deg, Roll = {actor_rot.roll:.4f} deg"
        )
        print(
            f"   Center of Mass (Local)       : "
            f"X = {physics_control.center_of_mass.x:.4f} m, "
            f"Y = {physics_control.center_of_mass.y:.4f} m, "
            f"Z = {physics_control.center_of_mass.z:.4f} m"
        )

        print("\n   Per-Wheel Physics Inspection (CARLA World cm -> m):")
        wheel_labels = ("Front-Left ", "Front-Right", "Rear-Left  ", "Rear-Right ")
        for idx, w in enumerate(wheels):
            lbl = wheel_labels[idx] if idx < len(wheel_labels) else f"Wheel {idx}    "
            print(
                f"   - {lbl} : "
                f"World=({w.position.x / 100.0:8.4f}, {w.position.y / 100.0:8.4f}, {w.position.z / 100.0:6.4f}) m | "
                f"Radius={w.radius:.2f} cm | MaxSteer={w.max_steer_angle:.2f} deg"
            )

        print("\n   Extracted Axle Geometry:")
        print(
            f"   - Measured Wheelbase (L)     : {geom.wheelbase:.4f} m"
        )
        print(
            f"   - Front-Axle Local Offset X  : {geom.front_axle_offset_x:+.4f} m (ahead of actor origin)"
        )
        print(
            f"   - Rear-Axle Local Offset X   : {geom.rear_axle_offset_x:+.4f} m (behind actor origin)"
        )
        print(
            f"   - Rear-Axle Local Offset Y   : {geom.rear_axle_offset_y_ctrl:+.4f} m (controller +left)"
        )
        print(
            f"   - Front Axle Location (CARLA): "
            f"X = {front_axle_carla_x:.4f} m, Y = {front_axle_carla_y:.4f} m, Z = {front_axle_carla_z:.4f} m"
        )
        print(
            f"   - Front Axle Location (Ctrl) : "
            f"X = {front_axle_ctrl_x:.4f} m, Y = {front_axle_ctrl_y:.4f} m"
        )
        print(
            f"   - Rear Axle Location (CARLA) : "
            f"X = {rear_axle_carla_x:.4f} m, Y = {rear_axle_carla_y:.4f} m, Z = {rear_axle_carla_z:.4f} m"
        )
        print(
            f"   - Rear Axle Location (Ctrl)  : "
            f"X = {direct_rear_ctrl_x:.4f} m, Y = {direct_rear_ctrl_y:.4f} m"
        )
        print(
            f"   - Max Front-Wheel Steering   : "
            f"{geom.max_front_wheel_steer_deg:.4f} deg ({geom.max_front_wheel_steer_rad:.4f} rad)"
        )

        steering_curve_pts = format_steering_curve(physics_control)
        print("   - Steering Curve Points      :")
        for spd, mult in steering_curve_pts:
            print(f"       Speed = {spd:6.2f} -> Steer Multiplier = {mult:.4f}")

        # 5. Coordinate & Rear-Axle Transformation Verification
        actor_ctrl_x, actor_ctrl_y = carla_to_controller_position(
            float(actor_loc.x),
            float(actor_loc.y)
        )
        actor_ctrl_yaw = carla_to_controller_yaw(float(actor_rot.yaw))
        adapter_rear_x, adapter_rear_y, adapter_yaw = adapter.get_rear_axle_pose()

        est_rear_x, est_rear_y = estimate_rear_axle_position(
            carla_x=float(actor_loc.x),
            carla_y=float(actor_loc.y),
            carla_yaw_degrees=float(actor_rot.yaw),
            rear_axle_offset_x=geom.rear_axle_offset_x,
            rear_axle_offset_y_ctrl=geom.rear_axle_offset_y_ctrl
        )

        rear_pos_error_m = math.hypot(
            adapter_rear_x - direct_rear_ctrl_x,
            adapter_rear_y - direct_rear_ctrl_y
        )

        print("\n2. COORDINATE & REAR-AXLE TRANSFORMATION VERIFICATION")
        print("-" * 85)
        print(
            f"   CARLA Actor Pose             : "
            f"X = {actor_loc.x:.4f} m, Y = {actor_loc.y:.4f} m, Yaw = {actor_rot.yaw:.4f} deg"
        )
        print(
            f"   Controller Actor Pose        : "
            f"X = {actor_ctrl_x:.4f} m, Y = {actor_ctrl_y:.4f} m, "
            f"Yaw = {actor_ctrl_yaw:.4f} rad ({math.degrees(actor_ctrl_yaw):.4f} deg)"
        )
        print(
            f"   Adapter Rear-Axle Pose       : "
            f"X = {adapter_rear_x:.4f} m, Y = {adapter_rear_y:.4f} m, "
            f"Yaw = {adapter_yaw:.4f} rad ({math.degrees(adapter_yaw):.4f} deg)"
        )
        print(
            f"   Direct Wheel-Midpoint Rear   : "
            f"X = {direct_rear_ctrl_x:.4f} m, Y = {direct_rear_ctrl_y:.4f} m"
        )
        print(
            f"   Rear-Axle Transformation Err : {rear_pos_error_m:.6e} m"
        )

        assert math.isclose(adapter_rear_x, est_rear_x, abs_tol=1e-9)
        assert math.isclose(adapter_rear_y, est_rear_y, abs_tol=1e-9)
        assert rear_pos_error_m < 1e-3, (
            f"Rear-axle transformation mismatch: {rear_pos_error_m} m"
        )

        # 6. Test Controller Steering Requests (-5 deg, 0 deg, +5 deg)
        print("\n3. STEERING CONVERSION & SIGN REVERSAL VERIFICATION")
        print("-" * 85)
        print(
            f"   {'Ctrl Request (deg)':>20} | "
            f"{'Ctrl Request (rad)':>20} | "
            f"{'CARLA Norm Steer':>18} | "
            f"{'Sign Check'}"
        )
        print("   " + "-" * 78)

        steering_results: dict[float, float] = {}
        for req_deg in TEST_STEERING_ANGLES_DEG:
            req_rad = math.radians(req_deg)
            norm_steer = adapter.to_carla_steer(req_rad)
            steering_results[req_deg] = norm_steer

            if req_deg < 0.0:
                sign_ok = norm_steer > 0.0
                sign_note = "PASS (-ctrl -> +carla right)" if sign_ok else "FAIL"
            elif req_deg > 0.0:
                sign_ok = norm_steer < 0.0
                sign_note = "PASS (+ctrl -> -carla left)" if sign_ok else "FAIL"
            else:
                sign_ok = norm_steer == 0.0
                sign_note = "PASS (0 -> 0)" if sign_ok else "FAIL"

            print(
                f"   {req_deg:+20.2f} | "
                f"{req_rad:+20.6f} | "
                f"{norm_steer:+18.6f} | "
                f"{sign_note}"
            )
            assert sign_ok, f"Steering sign check failed for {req_deg} deg"
            assert math.isfinite(norm_steer), "Non-finite normalized steer"

        # 7. Check all outputs are finite and vehicle remained stationary
        final_speed_mps = adapter.get_speed()
        all_finite = all(
            math.isfinite(val) for val in ctrl_state.values()
        ) and all(
            math.isfinite(val) for val in steering_results.values()
        )

        print("\n4. VALIDATION CHECKS")
        print("-" * 85)
        print(f"   All Adapter Outputs Finite   : {all_finite}")
        print(f"   Rear-Axle Transform Verified : {rear_pos_error_m < 1e-3}")
        print(
            f"   Steering Sign Reversal       : "
            f"{steering_results[-5.0] > 0.0 and steering_results[0.0] == 0.0 and steering_results[5.0] < 0.0}"
        )
        print(
            f"   Vehicle Stationary (Handbrake): "
            f"Speed = {final_speed_mps:.4f} m/s"
        )
        print("=" * 85)

        assert all_finite, "Detected non-finite adapter output values."

    finally:
        if vehicle is not None and vehicle.is_alive:
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                    hand_brake=True
                )
            )
            vehicle.destroy()
            print("\nCleaned up: Tesla Model 3 actor destroyed successfully.")


if __name__ == "__main__":
    main()
