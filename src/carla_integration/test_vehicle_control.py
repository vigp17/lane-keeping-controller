from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import cast

import carla


# ==================================================
# 1. TEST CONFIGURATION (CONSERVATIVE PARAMETERS)
# ==================================================

CARLA_HOST: str = "127.0.0.1"
CARLA_PORT: int = 2000
CLIENT_TIMEOUT: float = 30.0

VEHICLE_BLUEPRINT_ID: str = "vehicle.tesla.model3"

CONTROL_DT: float = 0.1

# Conservative longitudinal control parameters
ACCEL_THROTTLE: float = 0.35
STEER_TEST_THROTTLE: float = 0.30
MAX_TEST_SPEED_MPS: float = 4.5
STOP_SPEED_THRESHOLD_MPS: float = 0.05
FULL_BRAKE: float = 1.0

# Conservative normalized steering commands in [-1.0, 1.0]
POSITIVE_STEER_CMD: float = 0.15
NEGATIVE_STEER_CMD: float = -0.15

# Test phase durations (seconds)
SETTLE_DURATION: float = 0.5
ACCEL_DURATION: float = 2.5
MAX_BRAKE_DURATION: float = 4.0
STEER_RUN_DURATION: float = 2.5


# ==================================================
# 2. DATA STRUCTURES
# ==================================================

@dataclass
class ControlLogEntry:
    """Single telemetry and control log sample."""

    phase: str
    sim_time: float
    x: float
    y: float
    z: float
    yaw_deg: float
    speed_mps: float
    throttle: float
    brake: float
    steer: float


@dataclass
class SteeringRunSummary:
    """Summary metrics for a single constant-steering test run."""

    label: str
    commanded_steer: float
    initial_yaw_deg: float
    final_yaw_deg: float
    delta_yaw_deg: float
    peak_speed_mps: float
    turn_direction: str


# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

def normalize_angle_deg(angle_deg: float) -> float:
    """
    Wrap an angle in degrees to the principal interval (-180.0, 180.0].
    """
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    if wrapped <= -180.0:
        return 180.0
    return float(wrapped)


def get_vehicle_speed_mps(vehicle: carla.Vehicle) -> float:
    """
    Calculate the 3D Euclidean speed of the vehicle in meters/second.
    """
    velocity = vehicle.get_velocity()
    return float(
        math.sqrt(
            velocity.x ** 2
            + velocity.y ** 2
            + velocity.z ** 2
        )
    )


def get_simulation_time_seconds(
    world: carla.World,
    start_elapsed_seconds: float,
    wall_start_time: float
) -> float:
    """
    Retrieve elapsed simulation time in seconds from the CARLA world snapshot,
    falling back to wall-clock elapsed time if snapshot timestamp is zero.
    """
    snapshot = world.get_snapshot()
    sim_elapsed = float(snapshot.timestamp.elapsed_seconds) - start_elapsed_seconds
    if sim_elapsed > 0.0:
        return sim_elapsed
    return float(time.perf_counter() - wall_start_time)


def update_spectator_camera(
    world: carla.World,
    vehicle: carla.Vehicle
) -> None:
    """
    Position the spectator camera 8 m behind and 5 m above the vehicle,
    aimed directly toward the vehicle reference location.
    """
    transform = vehicle.get_transform()
    location = transform.location
    forward = transform.get_forward_vector()

    camera_location = carla.Location(
        x=location.x - 8.0 * forward.x,
        y=location.y - 8.0 * forward.y,
        z=location.z + 5.0
    )

    dx = location.x - camera_location.x
    dy = location.y - camera_location.y
    dz = location.z - camera_location.z

    camera_yaw = math.degrees(math.atan2(dy, dx))
    horizontal_distance = math.hypot(dx, dy)
    camera_pitch = math.degrees(
        math.atan2(dz, horizontal_distance)
    )

    spectator = world.get_spectator()
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


def select_driving_lane_spawn_point(
    world: carla.World,
    vehicle_bp: carla.ActorBlueprint
) -> tuple[carla.Vehicle, carla.Transform]:
    """
    Find an available driving-lane spawn point with a clear forward road
    segment and spawn the Tesla Model 3.
    """
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    if not spawn_points:
        raise RuntimeError("No spawn points available in the current CARLA map.")

    # Rank spawn points: prefer straight driving-lane segments ahead
    scored_spawn_points: list[tuple[float, carla.Transform]] = []

    for sp in spawn_points:
        waypoint = carla_map.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            continue

        next_wps = waypoint.next(25.0)
        if next_wps:
            yaw_diff = abs(
                normalize_angle_deg(
                    next_wps[0].transform.rotation.yaw
                    - waypoint.transform.rotation.yaw
                )
            )
        else:
            yaw_diff = 90.0

        scored_spawn_points.append((yaw_diff, sp))

    # Sort so straightest driving-lane spawn points are tried first
    scored_spawn_points.sort(key=lambda item: item[0])
    candidate_transforms = [sp for _, sp in scored_spawn_points] or list(spawn_points)

    for spawn_transform in candidate_transforms:
        spawned_actor = world.try_spawn_actor(
            vehicle_bp,
            spawn_transform
        )
        if spawned_actor is not None:
            vehicle = cast(carla.Vehicle, spawned_actor)
            return vehicle, spawn_transform

    raise RuntimeError("Unable to spawn Tesla Model 3 at any driving-lane spawn point.")


def log_vehicle_step(
    phase: str,
    world: carla.World,
    vehicle: carla.Vehicle,
    throttle: float,
    brake: float,
    steer: float,
    start_elapsed_seconds: float,
    wall_start_time: float,
    log_history: list[ControlLogEntry]
) -> ControlLogEntry:
    """
    Apply control, update spectator camera, read state, print telemetry,
    and append a ControlLogEntry.
    """
    control = carla.VehicleControl(
        throttle=float(throttle),
        steer=float(steer),
        brake=float(brake),
        hand_brake=False,
        reverse=False
    )
    vehicle.apply_control(control)
    update_spectator_camera(world, vehicle)

    transform = vehicle.get_transform()
    location = transform.location
    yaw_deg = float(transform.rotation.yaw)
    speed_mps = get_vehicle_speed_mps(vehicle)
    sim_time = get_simulation_time_seconds(
        world,
        start_elapsed_seconds,
        wall_start_time
    )

    entry = ControlLogEntry(
        phase=phase,
        sim_time=sim_time,
        x=float(location.x),
        y=float(location.y),
        z=float(location.z),
        yaw_deg=yaw_deg,
        speed_mps=speed_mps,
        throttle=float(throttle),
        brake=float(brake),
        steer=float(steer)
    )
    log_history.append(entry)

    print(
        f"[{entry.phase:<16}] "
        f"t={entry.sim_time:6.2f} s | "
        f"X={entry.x:8.2f} m | "
        f"Y={entry.y:8.2f} m | "
        f"Z={entry.z:6.2f} m | "
        f"Yaw={entry.yaw_deg:8.2f} deg | "
        f"Speed={entry.speed_mps:5.2f} m/s | "
        f"Thr={entry.throttle:4.2f} | "
        f"Brk={entry.brake:4.2f} | "
        f"Steer={entry.steer:+6.2f}"
    )

    return entry


def brake_until_stopped(
    phase_name: str,
    world: carla.World,
    vehicle: carla.Vehicle,
    start_elapsed_seconds: float,
    wall_start_time: float,
    log_history: list[ControlLogEntry],
    max_duration: float = MAX_BRAKE_DURATION,
    dt: float = CONTROL_DT
) -> tuple[float, float, int]:
    """
    Apply full braking with zero throttle and zero steering until the vehicle
    comes to a complete stop (speed <= STOP_SPEED_THRESHOLD_MPS).

    Returns:
        Tuple of (initial_speed_mps, final_speed_mps, steps_taken).
    """
    max_steps = int(round(max_duration / dt))
    initial_speed = get_vehicle_speed_mps(vehicle)
    stopped_Hold_count = 0
    steps_taken = 0

    for _ in range(max_steps):
        steps_taken += 1
        entry = log_vehicle_step(
            phase=phase_name,
            world=world,
            vehicle=vehicle,
            throttle=0.0,
            brake=FULL_BRAKE,
            steer=0.0,
            start_elapsed_seconds=start_elapsed_seconds,
            wall_start_time=wall_start_time,
            log_history=log_history
        )
        time.sleep(dt)

        if entry.speed_mps <= STOP_SPEED_THRESHOLD_MPS:
            stopped_Hold_count += 1
            if stopped_Hold_count >= 2:
                break
        else:
            stopped_Hold_count = 0

    final_speed = get_vehicle_speed_mps(vehicle)
    return initial_speed, final_speed, steps_taken


def reset_vehicle_pose(
    world: carla.World,
    vehicle: carla.Vehicle,
    spawn_transform: carla.Transform,
    dt: float = CONTROL_DT
) -> None:
    """
    Reset the vehicle to its initial driving-lane spawn transform with zero
    linear and angular velocity so separate steering runs start from identical
    initial conditions.
    """
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=FULL_BRAKE,
            hand_brake=True
        )
    )
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_transform(spawn_transform)
    update_spectator_camera(world, vehicle)
    time.sleep(0.3)

    # Release handbrake and settle with normal brake
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=FULL_BRAKE,
            hand_brake=False
        )
    )
    for _ in range(int(round(SETTLE_DURATION / dt))):
        update_spectator_camera(world, vehicle)
        time.sleep(dt)


def run_steering_test(
    run_label: str,
    steer_command: float,
    world: carla.World,
    vehicle: carla.Vehicle,
    spawn_transform: carla.Transform,
    start_elapsed_seconds: float,
    wall_start_time: float,
    log_history: list[ControlLogEntry],
    duration: float = STEER_RUN_DURATION,
    dt: float = CONTROL_DT
) -> SteeringRunSummary:
    """
    Execute a single constant-steering test run starting from the clean
    driving-lane spawn point, then brake to a stop.
    """
    print("\n" + "=" * 95)
    print(
        f"STARTING STEERING RUN: {run_label} "
        f"(Normalized Steer = {steer_command:+.2f})"
    )
    print("=" * 95)

    reset_vehicle_pose(world, vehicle, spawn_transform, dt=dt)

    initial_yaw_deg = float(vehicle.get_transform().rotation.yaw)
    peak_speed_mps = 0.0
    steps = int(round(duration / dt))

    for _ in range(steps):
        current_speed = get_vehicle_speed_mps(vehicle)
        applied_throttle = (
            STEER_TEST_THROTTLE
            if current_speed < MAX_TEST_SPEED_MPS
            else 0.0
        )

        entry = log_vehicle_step(
            phase=run_label,
            world=world,
            vehicle=vehicle,
            throttle=applied_throttle,
            brake=0.0,
            steer=steer_command,
            start_elapsed_seconds=start_elapsed_seconds,
            wall_start_time=wall_start_time,
            log_history=log_history
        )
        peak_speed_mps = max(peak_speed_mps, entry.speed_mps)
        time.sleep(dt)

    final_yaw_deg = float(vehicle.get_transform().rotation.yaw)
    delta_yaw_deg = normalize_angle_deg(final_yaw_deg - initial_yaw_deg)

    # Brake vehicle back to a complete stop after steering run
    brake_until_stopped(
        phase_name=f"{run_label} Brake",
        world=world,
        vehicle=vehicle,
        start_elapsed_seconds=start_elapsed_seconds,
        wall_start_time=wall_start_time,
        log_history=log_history,
        dt=dt
    )

    if delta_yaw_deg > 0.5:
        turn_direction = "Right turn (CARLA +Yaw / Clockwise)"
    elif delta_yaw_deg < -0.5:
        turn_direction = "Left turn (CARLA -Yaw / Counterclockwise)"
    else:
        turn_direction = "Negligible yaw change"

    return SteeringRunSummary(
        label=run_label,
        commanded_steer=steer_command,
        initial_yaw_deg=initial_yaw_deg,
        final_yaw_deg=final_yaw_deg,
        delta_yaw_deg=delta_yaw_deg,
        peak_speed_mps=peak_speed_mps,
        turn_direction=turn_direction
    )


def print_test_summary(
    accel_peak_speed: float,
    brake_initial_speed: float,
    brake_final_speed: float,
    brake_steps: int,
    positive_summary: SteeringRunSummary,
    negative_summary: SteeringRunSummary,
    total_logs: int
) -> None:
    """
    Print a structured summary of longitudinal and steering-direction tests.
    """
    print("\n" + "=" * 95)
    print("CARLA 0.9.16 VEHICLE CONTROL & STEERING SIGN CONVENTION SUMMARY")
    print("=" * 95)

    print("\n1. Longitudinal Control Verification:")
    print(
        f"   - Low-Speed Acceleration Peak Speed : {accel_peak_speed:.2f} m/s"
    )
    print(
        f"   - Braking Initial -> Final Speed    : "
        f"{brake_initial_speed:.2f} m/s -> {brake_final_speed:.2f} m/s "
        f"({brake_steps * CONTROL_DT:.1f} s)"
    )

    print("\n2. Steering-Direction Response Summary:")
    print(
        f"   {'Run':<18} | {'Steer Cmd':>10} | {'Init Yaw (deg)':>14} | "
        f"{'Final Yaw (deg)':>15} | {'Delta Yaw (deg)':>15} | {'Measured Response'}"
    )
    print("   " + "-" * 90)

    for s in (positive_summary, negative_summary):
        print(
            f"   {s.label:<18} | "
            f"{s.commanded_steer:+10.2f} | "
            f"{s.initial_yaw_deg:14.2f} | "
            f"{s.final_yaw_deg:15.2f} | "
            f"{s.delta_yaw_deg:+15.2f} | "
            f"{s.turn_direction}"
        )

    pos_ok = positive_summary.delta_yaw_deg > 0.0
    neg_ok = negative_summary.delta_yaw_deg < 0.0
    convention_verified = pos_ok and neg_ok

    print("\n3. Steering Sign Convention Verification:")
    print(
        f"   - Positive Steer ({positive_summary.commanded_steer:+.2f}) -> "
        f"Delta Yaw = {positive_summary.delta_yaw_deg:+.2f} deg "
        f"({'PASS: Yaw Increased' if pos_ok else 'CHECK'})"
    )
    print(
        f"   - Negative Steer ({negative_summary.commanded_steer:+.2f}) -> "
        f"Delta Yaw = {negative_summary.delta_yaw_deg:+.2f} deg "
        f"({'PASS: Yaw Decreased' if neg_ok else 'CHECK'})"
    )
    print(
        f"   - Sign Convention Verified          : "
        f"{'YES' if convention_verified else 'NO'}"
    )
    print(
        "   - Coordinate Note                   : "
        "CARLA uses a left-handed coordinate system (+X forward, +Y right, +Yaw clockwise).\n"
        "                                         "
        "Positive CARLA steer (+1) turns right (+Yaw), whereas positive steering delta\n"
        "                                         "
        "in our right-handed controller models (+Y left, +Yaw CCW) turns left."
    )
    print(f"   - Total Telemetry Samples Logged    : {total_logs}")
    print("=" * 95)


# ==================================================
# 4. MAIN TEST PROCEDURE
# ==================================================

def main() -> None:
    """
    Connect to CARLA, spawn a Tesla Model 3, and execute low-speed
    acceleration, braking, and positive/negative steering verification tests.
    """
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(CLIENT_TIMEOUT)

    world = client.get_world()
    carla_map = world.get_map()

    print(f"Connected to CARLA at {CARLA_HOST}:{CARLA_PORT}")
    print(f"Active Map: {carla_map.name}")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(VEHICLE_BLUEPRINT_ID)

    vehicle: carla.Vehicle | None = None
    log_history: list[ControlLogEntry] = []

    try:
        # 1. Spawn Tesla Model 3 at a driving-lane spawn point
        vehicle, spawn_transform = select_driving_lane_spawn_point(
            world,
            vehicle_bp
        )

        print(
            f"Spawned {vehicle.type_id} (Actor ID: {vehicle.id}) at "
            f"X={spawn_transform.location.x:.2f} m, "
            f"Y={spawn_transform.location.y:.2f} m, "
            f"Z={spawn_transform.location.z:.2f} m, "
            f"Yaw={spawn_transform.rotation.yaw:.2f} deg"
        )

        # 2. Disable autopilot and allow vehicle suspension to settle
        vehicle.set_autopilot(False)
        update_spectator_camera(world, vehicle)

        initial_snapshot = world.get_snapshot()
        start_elapsed_seconds = float(
            initial_snapshot.timestamp.elapsed_seconds
        )
        wall_start_time = time.perf_counter()

        print("\n" + "=" * 95)
        print("PHASE 1: INITIAL SETTLE")
        print("=" * 95)
        for _ in range(int(round(SETTLE_DURATION / CONTROL_DT))):
            log_vehicle_step(
                phase="Settle",
                world=world,
                vehicle=vehicle,
                throttle=0.0,
                brake=FULL_BRAKE,
                steer=0.0,
                start_elapsed_seconds=start_elapsed_seconds,
                wall_start_time=wall_start_time,
                log_history=log_history
            )
            time.sleep(CONTROL_DT)

        # 3. Test low-speed acceleration using conservative throttle
        print("\n" + "=" * 95)
        print(
            f"PHASE 2: LOW-SPEED ACCELERATION "
            f"(Throttle = {ACCEL_THROTTLE:.2f}, Speed Cap = {MAX_TEST_SPEED_MPS:.1f} m/s)"
        )
        print("=" * 95)

        accel_peak_speed = 0.0
        accel_steps = int(round(ACCEL_DURATION / CONTROL_DT))

        for _ in range(accel_steps):
            current_speed = get_vehicle_speed_mps(vehicle)
            applied_throttle = (
                ACCEL_THROTTLE
                if current_speed < MAX_TEST_SPEED_MPS
                else 0.0
            )
            entry = log_vehicle_step(
                phase="Accelerate",
                world=world,
                vehicle=vehicle,
                throttle=applied_throttle,
                brake=0.0,
                steer=0.0,
                start_elapsed_seconds=start_elapsed_seconds,
                wall_start_time=wall_start_time,
                log_history=log_history
            )
            accel_peak_speed = max(accel_peak_speed, entry.speed_mps)
            time.sleep(CONTROL_DT)

        # 4. Test braking until the vehicle stops
        print("\n" + "=" * 95)
        print("PHASE 3: BRAKING TO COMPLETE STOP")
        print("=" * 95)

        brake_init_speed, brake_final_speed, brake_steps = (
            brake_until_stopped(
                phase_name="Brake-Stop",
                world=world,
                vehicle=vehicle,
                start_elapsed_seconds=start_elapsed_seconds,
                wall_start_time=wall_start_time,
                log_history=log_history,
                dt=CONTROL_DT
            )
        )

        # 5. Test small positive normalized steering command in a separate run
        positive_summary = run_steering_test(
            run_label="Steer-Positive",
            steer_command=POSITIVE_STEER_CMD,
            world=world,
            vehicle=vehicle,
            spawn_transform=spawn_transform,
            start_elapsed_seconds=start_elapsed_seconds,
            wall_start_time=wall_start_time,
            log_history=log_history,
            duration=STEER_RUN_DURATION,
            dt=CONTROL_DT
        )

        # 6. Test small negative normalized steering command in a separate run
        negative_summary = run_steering_test(
            run_label="Steer-Negative",
            steer_command=NEGATIVE_STEER_CMD,
            world=world,
            vehicle=vehicle,
            spawn_transform=spawn_transform,
            start_elapsed_seconds=start_elapsed_seconds,
            wall_start_time=wall_start_time,
            log_history=log_history,
            duration=STEER_RUN_DURATION,
            dt=CONTROL_DT
        )

        # 7. Print summary of measured steering-direction response
        print_test_summary(
            accel_peak_speed=accel_peak_speed,
            brake_initial_speed=brake_init_speed,
            brake_final_speed=brake_final_speed,
            brake_steps=brake_steps,
            positive_summary=positive_summary,
            negative_summary=negative_summary,
            total_logs=len(log_history)
        )

    except KeyboardInterrupt:
        print("\nVehicle control test interrupted by user.")

    finally:
        if vehicle is not None and vehicle.is_alive:
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=FULL_BRAKE,
                    hand_brake=True
                )
            )
            vehicle.destroy()
            print("\nCleaned up: Tesla Model 3 actor destroyed successfully.")


if __name__ == "__main__":
    main()
