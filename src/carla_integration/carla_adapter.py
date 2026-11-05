"""
CARLA-to-Controller Adapter for Autonomous Vehicle Lane Keeping.

Bridges CARLA 0.9.16's left-handed coordinate and control conventions with
the project's right-handed controller conventions:

Coordinate Conventions:
    - CARLA World Frame (Left-Handed):
        +X: Forward
        +Y: Right
        +Z: Up
        +Yaw: Clockwise (degrees)
        +Steer: Right turn, normalized in [-1.0, 1.0]

    - Controller Frame (Right-Handed):
        +X: Forward
        +Y: Left
        +Yaw: Counterclockwise (radians, wrapped to [-pi, pi])
        +Steering Angle (delta): Left turn (radians)

Assumptions and Limitations:
    1. Planar 2D Tracking: Road elevation (Z), roll, and pitch are neglected
       by the 2D kinematic bicycle controllers; X and Y coordinates are
       projected onto the horizontal plane.
    2. Linear Steering Mapping: Normalized CARLA steering is mapped linearly
       via delta_ctrl / delta_max_front. Speed-dependent steering reduction
       curves (physics_control.steering_curve) and tire/suspension compliance
       inside PhysX/Chaos can alter effective wheel angles at higher speeds.
    3. Wheel Position Frame: In CARLA 0.9.16, WheelPhysicsControl.position
       is reported in world coordinates in centimeters at the moment
       get_physics_control() is queried. This adapter transforms those 3D
       wheel positions into the actor's local frame using the actor's current
       transform to determine the exact rear-axle offset and wheelbase.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import carla  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    carla = None  # type: ignore[assignment]


# ==================================================
# 1. STRUCTURAL PROTOCOLS FOR TYPE SAFETY & TESTING
# ==================================================

class Vector3DLike(Protocol):
    """Protocol matching carla.Vector3D and carla.Location."""

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def z(self) -> float: ...


class RotationLike(Protocol):
    """Protocol matching carla.Rotation."""

    @property
    def pitch(self) -> float: ...

    @property
    def yaw(self) -> float: ...

    @property
    def roll(self) -> float: ...


class TransformLike(Protocol):
    """Protocol matching carla.Transform."""

    @property
    def location(self) -> Vector3DLike: ...

    @property
    def rotation(self) -> RotationLike: ...


class WaypointLike(Protocol):
    """Protocol matching carla.Waypoint."""

    @property
    def transform(self) -> TransformLike: ...


class WheelPhysicsLike(Protocol):
    """Protocol matching carla.WheelPhysicsControl."""

    @property
    def position(self) -> Vector3DLike: ...

    @property
    def max_steer_angle(self) -> float: ...


class PhysicsControlLike(Protocol):
    """Protocol matching carla.VehiclePhysicsControl."""

    @property
    def wheels(self) -> Sequence[WheelPhysicsLike]: ...


class VehicleLike(Protocol):
    """Protocol matching carla.Vehicle methods used by the adapter."""

    def get_transform(self) -> TransformLike: ...

    def get_velocity(self) -> Vector3DLike: ...

    def get_physics_control(self) -> PhysicsControlLike: ...


@dataclass(frozen=True)
class VehicleGeometryParameters:
    """
    Vehicle geometry and steering parameters extracted from CARLA physics.

    Attributes:
        rear_axle_offset_x: Longitudinal offset of the rear-axle midpoint
            relative to the actor origin along the actor +X forward axis (m).
            Negative when the rear axle is behind the actor origin.
        rear_axle_offset_y_ctrl: Lateral offset of the rear-axle midpoint
            relative to the actor origin along the controller +Y left axis (m).
        front_axle_offset_x: Longitudinal offset of the front-axle midpoint
            relative to the actor origin along the actor +X forward axis (m).
        wheelbase: Distance between front and rear axle midpoints (m).
        max_front_wheel_steer_deg: Maximum front-wheel steering angle (deg).
        max_front_wheel_steer_rad: Maximum front-wheel steering angle (rad).
    """

    rear_axle_offset_x: float
    rear_axle_offset_y_ctrl: float
    front_axle_offset_x: float
    wheelbase: float
    max_front_wheel_steer_deg: float
    max_front_wheel_steer_rad: float


# ==================================================
# 2. ANGLE & COORDINATE CONVERSIONS
# ==================================================

def normalize_angle(angle_rad: float) -> float:
    """
    Wrap an angle in radians to the principal interval [-pi, pi].
    """
    if not math.isfinite(angle_rad):
        raise ValueError("angle_rad must be a finite number.")

    wrapped = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
    return float(wrapped)


def carla_to_controller_position(
    carla_x: float,
    carla_y: float
) -> tuple[float, float]:
    """
    Convert CARLA global (X, Y) coordinates to controller (X, Y) coordinates.

    CARLA:      +X forward, +Y right
    Controller: +X forward, +Y left

        controller_x = carla_x
        controller_y = -carla_y
    """
    if not (math.isfinite(carla_x) and math.isfinite(carla_y)):
        raise ValueError(
            "carla_x and carla_y must be finite numerical values."
        )

    controller_x = float(carla_x)
    controller_y = float(-carla_y)
    return controller_x, controller_y


def carla_location_to_controller_position(
    location: Vector3DLike
) -> tuple[float, float]:
    """
    Convert a carla.Location (or Vector3D) to controller (X, Y) coordinates.
    """
    return carla_to_controller_position(
        float(location.x),
        float(location.y)
    )


def carla_to_controller_yaw(carla_yaw_degrees: float) -> float:
    """
    Convert CARLA yaw in degrees (clockwise positive) to controller yaw
    in radians (counterclockwise positive), normalized to [-pi, pi].

        controller_yaw = normalize_angle(-radians(carla_yaw_degrees))
    """
    if not math.isfinite(carla_yaw_degrees):
        raise ValueError("carla_yaw_degrees must be a finite number.")

    controller_yaw = -math.radians(float(carla_yaw_degrees))
    return normalize_angle(controller_yaw)


def carla_transform_to_controller_pose(
    transform: TransformLike
) -> tuple[float, float, float]:
    """
    Convert a carla.Transform into (controller_x, controller_y, controller_yaw).
    """
    ctrl_x, ctrl_y = carla_to_controller_position(
        float(transform.location.x),
        float(transform.location.y)
    )
    ctrl_yaw = carla_to_controller_yaw(float(transform.rotation.yaw))
    return ctrl_x, ctrl_y, ctrl_yaw


# ==================================================
# 3. WAYPOINT CONVERSION
# ==================================================

def convert_carla_waypoints(
    waypoints: Sequence[WaypointLike | TransformLike]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert a sequence of CARLA Waypoints (or Transforms) into 1D NumPy
    arrays (road_x, road_y, road_yaw) in the controller coordinate frame.

    Args:
        waypoints: Non-empty sequence of carla.Waypoint or carla.Transform.

    Returns:
        Tuple of (road_x, road_y, road_yaw) float64 arrays where:
            road_x: Longitudinal coordinates (m)
            road_y: Lateral coordinates (m, +left)
            road_yaw: Reference headings in [-pi, pi] (rad, +CCW)
    """
    if len(waypoints) == 0:
        raise ValueError("waypoints sequence must not be empty.")

    n_pts = len(waypoints)
    road_x = np.empty(n_pts, dtype=np.float64)
    road_y = np.empty(n_pts, dtype=np.float64)
    road_yaw = np.empty(n_pts, dtype=np.float64)

    for i, item in enumerate(waypoints):
        tf: TransformLike = (
            getattr(item, "transform", item)  # type: ignore[assignment]
        )
        cx, cy, cyaw = carla_transform_to_controller_pose(tf)
        road_x[i] = cx
        road_y[i] = cy
        road_yaw[i] = cyaw

    return road_x, road_y, road_yaw


# ==================================================
# 4. VEHICLE SPEED EXTRACTION
# ==================================================

def extract_vehicle_speed(
    vehicle_or_velocity: VehicleLike | Vector3DLike,
    planar_only: bool = False
) -> float:
    """
    Extract scalar vehicle speed in meters/second from a CARLA vehicle
    or velocity vector.

    Args:
        vehicle_or_velocity: A carla.Vehicle (with get_velocity()) or
            a carla.Vector3D velocity vector (in m/s).
        planar_only: If True, compute horizontal speed sqrt(vx^2 + vy^2);
            otherwise compute full 3D speed sqrt(vx^2 + vy^2 + vz^2).

    Returns:
        Nonnegative vehicle speed in m/s.
    """
    if hasattr(vehicle_or_velocity, "get_velocity"):
        vel: Vector3DLike = vehicle_or_velocity.get_velocity()  # type: ignore[union-attr]
    else:
        vel = vehicle_or_velocity  # type: ignore[assignment]

    vx = float(vel.x)
    vy = float(vel.y)
    vz = 0.0 if planar_only else float(vel.z)

    if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
        raise ValueError("Velocity components must be finite numbers.")

    return float(math.sqrt(vx * vx + vy * vy + vz * vz))


# ==================================================
# 5. VEHICLE GEOMETRY & REAR-AXLE ESTIMATION
# ==================================================

def _resolve_local_wheel_positions_meters(
    physics_control: PhysicsControlLike,
    actor_transform: TransformLike | None = None
) -> list[tuple[float, float, float]]:
    """
    Convert wheel positions from physics_control.wheels into the actor's
    local frame (x_forward_m, y_right_m, max_steer_angle_deg).

    Handles CARLA 0.9.16's native convention where WheelPhysicsControl.position
    is reported in world coordinates in centimeters, as well as unit-test mocks
    that provide positions in local coordinates or meters.
    """
    wheels = list(physics_control.wheels)
    if len(wheels) < 2:
        raise ValueError(
            f"Expected at least 2 wheels in physics_control, got {len(wheels)}."
        )

    raw_coords: list[tuple[float, float, float, float]] = []
    for w in wheels:
        wx = float(w.position.x)
        wy = float(w.position.y)
        wz = float(w.position.z)
        steer_deg = float(w.max_steer_angle)
        if not (
            math.isfinite(wx)
            and math.isfinite(wy)
            and math.isfinite(wz)
            and math.isfinite(steer_deg)
        ):
            raise ValueError(
                "Wheel position and max_steer_angle must be finite numbers."
            )
        raw_coords.append((wx, wy, wz, steer_deg))

    # Determine whether wheel coordinates are in centimeters (CARLA default)
    # or already in meters by checking the maximum inter-wheel spread.
    max_spread = 0.0
    for i in range(len(raw_coords)):
        for j in range(i + 1, len(raw_coords)):
            dist = math.hypot(
                raw_coords[i][0] - raw_coords[j][0],
                raw_coords[i][1] - raw_coords[j][1]
            )
            if dist > max_spread:
                max_spread = dist

    if max_spread <= 1e-6:
        raise ValueError(
            "Degenerate wheel positions: all wheels are at the same location."
        )

    # A vehicle wheelbase/track spread is > 50 in cm (~280 cm) and < 20 in m (~2.8 m)
    scale_to_meters = 0.01 if max_spread > 25.0 else 1.0

    scaled_coords = [
        (wx * scale_to_meters, wy * scale_to_meters, wz * scale_to_meters, s_deg)
        for wx, wy, wz, s_deg in raw_coords
    ]

    if actor_transform is None:
        return [(wx, wy, s_deg) for wx, wy, _, s_deg in scaled_coords]

    actor_x = float(actor_transform.location.x)
    actor_y = float(actor_transform.location.y)
    actor_yaw_rad = math.radians(float(actor_transform.rotation.yaw))

    mean_wx = sum(c[0] for c in scaled_coords) / len(scaled_coords)
    mean_wy = sum(c[1] for c in scaled_coords) / len(scaled_coords)

    dist_to_actor = math.hypot(mean_wx - actor_x, mean_wy - actor_y)
    dist_to_origin = math.hypot(mean_wx, mean_wy)

    # If the wheel centroid is near the actor's world location (or far from origin
    # when the actor is far from origin), transform from CARLA world frame to
    # actor local frame. Otherwise, coordinates are already in actor local frame.
    is_world_frame = (dist_to_actor <= 5.0) or (dist_to_actor < dist_to_origin)

    if not is_world_frame:
        return [(wx, wy, s_deg) for wx, wy, _, s_deg in scaled_coords]

    cos_yaw = math.cos(actor_yaw_rad)
    sin_yaw = math.sin(actor_yaw_rad)

    local_wheels: list[tuple[float, float, float]] = []
    for wx, wy, _, s_deg in scaled_coords:
        dx = wx - actor_x
        dy = wy - actor_y
        # Project onto CARLA actor forward [cos, sin] and right [-sin, cos]
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw
        local_wheels.append((local_x, local_y, s_deg))

    return local_wheels


def extract_vehicle_geometry_from_physics(
    physics_control: PhysicsControlLike,
    actor_transform: TransformLike | None = None
) -> VehicleGeometryParameters:
    """
    Inspect CARLA vehicle physics and wheel positions to determine:
        - Rear-axle center offset relative to the actor reference frame
        - Front-axle center offset relative to the actor reference frame
        - Effective wheelbase L
        - Maximum front-wheel steering angle (degrees and radians)
    """
    local_wheels = _resolve_local_wheel_positions_meters(
        physics_control=physics_control,
        actor_transform=actor_transform
    )

    mean_x = sum(w[0] for w in local_wheels) / len(local_wheels)

    front_wheels = [w for w in local_wheels if w[0] >= mean_x]
    rear_wheels = [w for w in local_wheels if w[0] < mean_x]

    if not front_wheels or not rear_wheels:
        sorted_by_x = sorted(local_wheels, key=lambda item: item[0])
        half = len(sorted_by_x) // 2
        rear_wheels = sorted_by_x[:half]
        front_wheels = sorted_by_x[half:]

    rear_axle_x = sum(w[0] for w in rear_wheels) / len(rear_wheels)
    rear_axle_y_carla = sum(w[1] for w in rear_wheels) / len(rear_wheels)
    # Convert lateral offset from CARLA (+right) to controller (+left)
    rear_axle_y_ctrl = -rear_axle_y_carla

    front_axle_x = sum(w[0] for w in front_wheels) / len(front_wheels)
    wheelbase = float(front_axle_x - rear_axle_x)

    if wheelbase <= 0.0:
        raise ValueError(
            f"Computed non-positive wheelbase ({wheelbase:.4f} m) from wheels."
        )

    steered_angles = [w[2] for w in front_wheels if w[2] > 0.0]
    if not steered_angles:
        steered_angles = [w[2] for w in local_wheels if w[2] > 0.0]

    if not steered_angles:
        raise ValueError(
            "No positive front-wheel max_steer_angle found in physics_control."
        )

    max_steer_deg = float(max(steered_angles))
    max_steer_rad = float(math.radians(max_steer_deg))

    return VehicleGeometryParameters(
        rear_axle_offset_x=float(rear_axle_x),
        rear_axle_offset_y_ctrl=float(rear_axle_y_ctrl),
        front_axle_offset_x=float(front_axle_x),
        wheelbase=wheelbase,
        max_front_wheel_steer_deg=max_steer_deg,
        max_front_wheel_steer_rad=max_steer_rad
    )


def get_rear_axle_offset_from_physics(
    physics_control: PhysicsControlLike,
    actor_transform: TransformLike | None = None
) -> tuple[float, float]:
    """
    Determine the rear-axle center offset (rear_axle_offset_x, rear_axle_offset_y_ctrl)
    in meters relative to the actor reference frame.

    Note:
        rear_axle_offset_x is negative because the rear axle lies behind
        the actor origin along the +X forward axis.
    """
    geom = extract_vehicle_geometry_from_physics(
        physics_control=physics_control,
        actor_transform=actor_transform
    )
    return geom.rear_axle_offset_x, geom.rear_axle_offset_y_ctrl


def estimate_rear_axle_position(
    carla_x: float,
    carla_y: float,
    carla_yaw_degrees: float,
    rear_axle_offset_x: float,
    rear_axle_offset_y_ctrl: float = 0.0
) -> tuple[float, float]:
    """
    Calculate the controller-frame rear-axle position (x_rear_ctrl, y_rear_ctrl)
    from the CARLA actor pose and the physics-derived rear-axle offset.

    Args:
        carla_x: CARLA actor X coordinate (m).
        carla_y: CARLA actor Y coordinate (m).
        carla_yaw_degrees: CARLA actor yaw angle (degrees, +CW).
        rear_axle_offset_x: Longitudinal offset of the rear axle relative to
            the actor origin along +X forward (m, negative behind origin).
        rear_axle_offset_y_ctrl: Lateral offset of the rear axle relative to
            the actor origin along +Y left (m, 0.0 for symmetric axle).

    Returns:
        Tuple of (rear_x_ctrl, rear_y_ctrl) in controller coordinates (m).
    """
    if not (
        math.isfinite(rear_axle_offset_x)
        and math.isfinite(rear_axle_offset_y_ctrl)
    ):
        raise ValueError("Rear-axle offsets must be finite numbers.")

    actor_x_ctrl, actor_y_ctrl = carla_to_controller_position(
        carla_x,
        carla_y
    )
    yaw_ctrl = carla_to_controller_yaw(carla_yaw_degrees)

    cos_yaw = math.cos(yaw_ctrl)
    sin_yaw = math.sin(yaw_ctrl)

    rear_x_ctrl = (
        actor_x_ctrl
        + rear_axle_offset_x * cos_yaw
        - rear_axle_offset_y_ctrl * sin_yaw
    )
    rear_y_ctrl = (
        actor_y_ctrl
        + rear_axle_offset_x * sin_yaw
        + rear_axle_offset_y_ctrl * cos_yaw
    )

    return float(rear_x_ctrl), float(rear_y_ctrl)


def extract_rear_axle_pose(
    vehicle: VehicleLike,
    rear_axle_offset: tuple[float, float] | None = None
) -> tuple[float, float, float]:
    """
    Extract the vehicle's rear-axle position and heading in controller
    coordinates: (rear_x_ctrl, rear_y_ctrl, controller_yaw_rad).

    If rear_axle_offset is not provided, it is computed directly from
    vehicle.get_physics_control() and vehicle.get_transform().
    """
    transform = vehicle.get_transform()

    if rear_axle_offset is None:
        physics_control = vehicle.get_physics_control()
        offset_x, offset_y_ctrl = get_rear_axle_offset_from_physics(
            physics_control=physics_control,
            actor_transform=transform
        )
    else:
        offset_x, offset_y_ctrl = (
            float(rear_axle_offset[0]),
            float(rear_axle_offset[1])
        )

    rear_x_ctrl, rear_y_ctrl = estimate_rear_axle_position(
        carla_x=float(transform.location.x),
        carla_y=float(transform.location.y),
        carla_yaw_degrees=float(transform.rotation.yaw),
        rear_axle_offset_x=offset_x,
        rear_axle_offset_y_ctrl=offset_y_ctrl
    )
    yaw_ctrl = carla_to_controller_yaw(float(transform.rotation.yaw))

    return rear_x_ctrl, rear_y_ctrl, yaw_ctrl


# ==================================================
# 6. STEERING CONVERSION
# ==================================================

def get_max_steering_angle_from_physics(
    physics_control: PhysicsControlLike
) -> float:
    """
    Read the maximum front-wheel steering angle from CARLA vehicle physics
    and return it in radians.
    """
    wheels = list(physics_control.wheels)
    if not wheels:
        raise ValueError("physics_control.wheels must not be empty.")

    steered_angles_deg = [
        float(w.max_steer_angle)
        for w in wheels
        if math.isfinite(float(w.max_steer_angle)) and float(w.max_steer_angle) > 0.0
    ]

    if not steered_angles_deg:
        raise ValueError(
            "No positive front-wheel max_steer_angle found in physics_control."
        )

    max_steer_deg = max(steered_angles_deg)
    return float(math.radians(max_steer_deg))


def get_vehicle_max_steering_angle(vehicle: VehicleLike) -> float:
    """
    Read the maximum front-wheel steering angle (in radians) directly
    from a CARLA vehicle actor.
    """
    return get_max_steering_angle_from_physics(
        vehicle.get_physics_control()
    )


def controller_steering_to_carla_steer(
    controller_steering_angle: float,
    maximum_front_wheel_steering_angle: float
) -> float:
    """
    Convert a controller steering command in radians (+left / CCW) to a
    normalized CARLA steering command in [-1.0, 1.0] (+right / CW):

        normalized_steer = clip(
            -controller_steering_angle / maximum_front_wheel_steering_angle,
            -1.0,
            1.0
        )
    """
    if not math.isfinite(controller_steering_angle):
        raise ValueError(
            "controller_steering_angle must be a finite number."
        )

    if (
        not math.isfinite(maximum_front_wheel_steering_angle)
        or maximum_front_wheel_steering_angle <= 0.0
    ):
        raise ValueError(
            "maximum_front_wheel_steering_angle must be a finite positive number."
        )

    raw_normalized = (
        -float(controller_steering_angle)
        / float(maximum_front_wheel_steering_angle)
    )
    clamped = max(-1.0, min(1.0, raw_normalized))

    # Avoid returning -0.0
    return 0.0 if clamped == 0.0 else float(clamped)


def carla_steer_to_controller_steering(
    normalized_steer: float,
    maximum_front_wheel_steering_angle: float
) -> float:
    """
    Convert a normalized CARLA steering value in [-1.0, 1.0] (+right) back
    to an effective front-wheel steering angle in the controller frame (+left, rad).
    """
    if not math.isfinite(normalized_steer):
        raise ValueError("normalized_steer must be a finite number.")

    if (
        not math.isfinite(maximum_front_wheel_steering_angle)
        or maximum_front_wheel_steering_angle <= 0.0
    ):
        raise ValueError(
            "maximum_front_wheel_steering_angle must be a finite positive number."
        )

    clamped_steer = max(-1.0, min(1.0, float(normalized_steer)))
    angle_rad = -clamped_steer * float(maximum_front_wheel_steering_angle)
    return 0.0 if angle_rad == 0.0 else float(angle_rad)


def create_carla_vehicle_control(
    controller_steering_angle: float,
    maximum_front_wheel_steering_angle: float,
    throttle: float = 0.0,
    brake: float = 0.0,
    hand_brake: bool = False,
    reverse: bool = False
) -> Any:
    """
    Build a CARLA VehicleControl command from a controller steering angle (rad)
    and longitudinal throttle/brake inputs.
    """
    if not (math.isfinite(throttle) and math.isfinite(brake)):
        raise ValueError("throttle and brake must be finite numbers.")

    norm_steer = controller_steering_to_carla_steer(
        controller_steering_angle=controller_steering_angle,
        maximum_front_wheel_steering_angle=maximum_front_wheel_steering_angle
    )
    clamped_throttle = max(0.0, min(1.0, float(throttle)))
    clamped_brake = max(0.0, min(1.0, float(brake)))

    if carla is not None:
        return carla.VehicleControl(
            throttle=clamped_throttle,
            steer=norm_steer,
            brake=clamped_brake,
            hand_brake=bool(hand_brake),
            reverse=bool(reverse)
        )

    # Lightweight fallback object when running unit tests without CARLA installed
    @dataclass
    class _FallbackVehicleControl:
        throttle: float
        steer: float
        brake: float
        hand_brake: bool
        reverse: bool

    return _FallbackVehicleControl(
        throttle=clamped_throttle,
        steer=norm_steer,
        brake=clamped_brake,
        hand_brake=bool(hand_brake),
        reverse=bool(reverse)
    )


# ==================================================
# 7. STATEFUL ADAPTER CLASS
# ==================================================

class CarlaControllerAdapter:
    """
    Reusable adapter binding a CARLA Vehicle actor to the project's
    lane-keeping controllers.

    Caches the vehicle's physics-derived geometry (rear-axle offset,
    wheelbase, and maximum front-wheel steering angle) upon initialization
    so online control steps do not need to repeatedly query PhysX/Chaos
    physics structures.
    """

    def __init__(self, vehicle: VehicleLike) -> None:
        self.vehicle: VehicleLike = vehicle
        initial_transform = vehicle.get_transform()
        physics_control = vehicle.get_physics_control()

        self.geometry: VehicleGeometryParameters = (
            extract_vehicle_geometry_from_physics(
                physics_control=physics_control,
                actor_transform=initial_transform
            )
        )

    @property
    def rear_axle_offset(self) -> tuple[float, float]:
        """Rear-axle offset (x_m, y_ctrl_m) relative to actor origin."""
        return (
            self.geometry.rear_axle_offset_x,
            self.geometry.rear_axle_offset_y_ctrl
        )

    @property
    def wheelbase(self) -> float:
        """Vehicle wheelbase L in meters."""
        return self.geometry.wheelbase

    @property
    def max_front_wheel_steering_angle(self) -> float:
        """Maximum front-wheel steering angle in radians."""
        return self.geometry.max_front_wheel_steer_rad

    def get_rear_axle_pose(self) -> tuple[float, float, float]:
        """
        Return current (rear_x_ctrl, rear_y_ctrl, yaw_ctrl_rad) in controller
        coordinates using the cached physics rear-axle offset.
        """
        return extract_rear_axle_pose(
            vehicle=self.vehicle,
            rear_axle_offset=self.rear_axle_offset
        )

    def get_speed(self, planar_only: bool = False) -> float:
        """Return current scalar vehicle speed in m/s."""
        return extract_vehicle_speed(
            self.vehicle,
            planar_only=planar_only
        )

    def get_controller_state(self) -> dict[str, float]:
        """
        Return a dictionary of current vehicle state variables in the
        controller's coordinate frame.
        """
        tf = self.vehicle.get_transform()
        actor_x_ctrl, actor_y_ctrl, yaw_ctrl = (
            carla_transform_to_controller_pose(tf)
        )
        rear_x_ctrl, rear_y_ctrl, _ = self.get_rear_axle_pose()
        speed_mps = self.get_speed()

        return {
            "x": rear_x_ctrl,
            "y": rear_y_ctrl,
            "yaw": yaw_ctrl,
            "speed": speed_mps,
            "actor_x": actor_x_ctrl,
            "actor_y": actor_y_ctrl,
            "rear_axle_offset_x": self.geometry.rear_axle_offset_x,
            "rear_axle_offset_y_ctrl": self.geometry.rear_axle_offset_y_ctrl,
            "wheelbase": self.geometry.wheelbase,
            "max_steer_angle_rad": self.geometry.max_front_wheel_steer_rad
        }

    @staticmethod
    def convert_waypoints(
        waypoints: Sequence[WaypointLike | TransformLike]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Convert CARLA waypoints to controller (road_x, road_y, road_yaw)."""
        return convert_carla_waypoints(waypoints)

    def to_carla_steer(self, controller_steering_angle: float) -> float:
        """
        Convert controller steering angle (rad, +left) to normalized CARLA
        steer in [-1.0, 1.0] (+right).
        """
        return controller_steering_to_carla_steer(
            controller_steering_angle=controller_steering_angle,
            maximum_front_wheel_steering_angle=(
                self.geometry.max_front_wheel_steer_rad
            )
        )

    def to_vehicle_control(
        self,
        controller_steering_angle: float,
        throttle: float = 0.0,
        brake: float = 0.0,
        hand_brake: bool = False,
        reverse: bool = False
    ) -> Any:
        """
        Construct a CARLA VehicleControl from a controller steering angle (rad).
        """
        return create_carla_vehicle_control(
            controller_steering_angle=controller_steering_angle,
            maximum_front_wheel_steering_angle=(
                self.geometry.max_front_wheel_steer_rad
            ),
            throttle=throttle,
            brake=brake,
            hand_brake=hand_brake,
            reverse=reverse
        )
