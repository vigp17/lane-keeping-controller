from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from pathlib import Path

import numpy as np
import pytest


# Add src directory to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_integration.carla_adapter import (
    CarlaControllerAdapter,
    carla_location_to_controller_position,
    carla_steer_to_controller_steering,
    carla_to_controller_position,
    carla_to_controller_yaw,
    carla_transform_to_controller_pose,
    controller_steering_to_carla_steer,
    convert_carla_waypoints,
    create_carla_vehicle_control,
    estimate_rear_axle_position,
    extract_rear_axle_pose,
    extract_vehicle_geometry_from_physics,
    extract_vehicle_speed,
    get_max_steering_angle_from_physics,
    get_rear_axle_offset_from_physics,
    get_vehicle_max_steering_angle,
    normalize_angle,
)


# ==================================================
# LIGHTWEIGHT MOCK CARLA OBJECTS FOR UNIT TESTING
# ==================================================

@dataclass
class MockVector3D:
    x: float
    y: float
    z: float = 0.0


@dataclass
class MockRotation:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass
class MockTransform:
    location: MockVector3D
    rotation: MockRotation


@dataclass
class MockWaypoint:
    transform: MockTransform


@dataclass
class MockWheelPhysics:
    position: MockVector3D
    max_steer_angle: float


@dataclass
class MockPhysicsControl:
    wheels: list[MockWheelPhysics]


class MockCarlaVehicle:
    """
    Lightweight mock of a CARLA 0.9.16 Tesla Model 3 actor.
    Simulates CARLA's WheelPhysicsControl.position in world centimeters.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        yaw_deg: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        front_axle_x_m: float = 1.45,
        rear_axle_x_m: float = -1.425,
        half_track_m: float = 0.80,
        max_front_steer_deg: float = 70.0
    ) -> None:
        self._transform = MockTransform(
            location=MockVector3D(x=x, y=y, z=z),
            rotation=MockRotation(yaw=yaw_deg)
        )
        self._velocity = MockVector3D(x=vx, y=vy, z=vz)

        # Compute wheel world positions in centimeters (matching CARLA 0.9.16)
        yaw_rad = math.radians(yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        local_wheels = [
            (front_axle_x_m, -half_track_m, max_front_steer_deg),  # FL
            (front_axle_x_m, +half_track_m, max_front_steer_deg),  # FR
            (rear_axle_x_m, -half_track_m, 0.0),                   # RL
            (rear_axle_x_m, +half_track_m, 0.0),                   # RR
        ]

        wheels: list[MockWheelPhysics] = []
        for lx, ly, steer_deg in local_wheels:
            wx_m = x + lx * cos_y - ly * sin_y
            wy_m = y + lx * sin_y + ly * cos_y
            wz_m = z + 0.35
            wheels.append(
                MockWheelPhysics(
                    position=MockVector3D(
                        x=wx_m * 100.0,
                        y=wy_m * 100.0,
                        z=wz_m * 100.0
                    ),
                    max_steer_angle=steer_deg
                )
            )

        self._physics = MockPhysicsControl(wheels=wheels)

    def get_transform(self) -> MockTransform:
        return self._transform

    def get_velocity(self) -> MockVector3D:
        return self._velocity

    def get_physics_control(self) -> MockPhysicsControl:
        return self._physics


# ==================================================
# 1. POSITION CONVERSION TESTS
# ==================================================

@pytest.mark.parametrize(
    ("carla_x", "carla_y", "expected_x", "expected_y"),
    [
        (0.0, 0.0, 0.0, 0.0),
        (25.5, 10.2, 25.5, -10.2),
        (-67.25, 27.96, -67.25, -27.96),
        (100.0, -4.5, 100.0, 4.5),
    ]
)
def test_position_conversion(
    carla_x: float,
    carla_y: float,
    expected_x: float,
    expected_y: float
) -> None:
    """
    Verify CARLA (X forward, Y right) converts to controller (X forward, Y left):
        controller_x = carla_x
        controller_y = -carla_y
    """
    ctrl_x, ctrl_y = carla_to_controller_position(carla_x, carla_y)
    assert np.isclose(ctrl_x, expected_x, atol=1e-12)
    assert np.isclose(ctrl_y, expected_y, atol=1e-12)

    loc = MockVector3D(x=carla_x, y=carla_y, z=1.5)
    loc_x, loc_y = carla_location_to_controller_position(loc)
    assert np.isclose(loc_x, expected_x, atol=1e-12)
    assert np.isclose(loc_y, expected_y, atol=1e-12)

    tf = MockTransform(location=loc, rotation=MockRotation(yaw=90.0))
    tf_x, tf_y, tf_yaw = carla_transform_to_controller_pose(tf)
    assert np.isclose(tf_x, expected_x, atol=1e-12)
    assert np.isclose(tf_y, expected_y, atol=1e-12)
    assert np.isclose(tf_yaw, -0.5 * math.pi, atol=1e-12)



# ==================================================
# 2. YAW CONVERSION TESTS
# ==================================================

@pytest.mark.parametrize(
    ("carla_yaw_deg", "expected_ctrl_yaw_rad"),
    [
        (0.0, 0.0),
        (90.0, -0.5 * math.pi),
        (-90.0, 0.5 * math.pi),
        (45.0, -0.25 * math.pi),
        (-30.0, math.pi / 6.0),
        (180.0, -math.pi),
    ]
)
def test_yaw_conversion(
    carla_yaw_deg: float,
    expected_ctrl_yaw_rad: float
) -> None:
    """
    Verify CARLA clockwise yaw in degrees converts to controller
    counterclockwise yaw in radians:
        controller_yaw = -radians(carla_yaw_degrees)
    """
    ctrl_yaw = carla_to_controller_yaw(carla_yaw_deg)
    assert -math.pi - 1e-12 <= ctrl_yaw <= math.pi + 1e-12
    # Compare on unit circle for +-pi equivalence at 180 deg
    assert np.isclose(
        math.sin(ctrl_yaw),
        math.sin(expected_ctrl_yaw_rad),
        atol=1e-12
    )
    assert np.isclose(
        math.cos(ctrl_yaw),
        math.cos(expected_ctrl_yaw_rad),
        atol=1e-12
    )


# ==================================================
# 3. HEADING NORMALIZATION TESTS
# ==================================================

@pytest.mark.parametrize(
    ("raw_angle_rad", "expected_rad"),
    [
        (0.0, 0.0),
        (2.0 * math.pi, 0.0),
        (-2.0 * math.pi, 0.0),
        (1.5 * math.pi, -0.5 * math.pi),
        (-1.5 * math.pi, 0.5 * math.pi),
        (3.25 * math.pi, -0.75 * math.pi),
    ]
)
def test_heading_normalization(
    raw_angle_rad: float,
    expected_rad: float
) -> None:
    """
    Verify normalize_angle wraps arbitrary angles into [-pi, pi], and
    carla_to_controller_yaw normalizes multi-turn CARLA degree angles.
    """
    norm = normalize_angle(raw_angle_rad)
    assert -math.pi - 1e-12 <= norm <= math.pi + 1e-12
    assert np.isclose(norm, expected_rad, atol=1e-12)

    # Also test CARLA degree inputs outside [-180, 180], e.g. 270 deg -> +pi/2
    yaw_270 = carla_to_controller_yaw(270.0)
    assert np.isclose(yaw_270, 0.5 * math.pi, atol=1e-12)

    yaw_neg_270 = carla_to_controller_yaw(-270.0)
    assert np.isclose(yaw_neg_270, -0.5 * math.pi, atol=1e-12)


# ==================================================
# 4. WAYPOINT CONVERSION TESTS
# ==================================================

def test_waypoint_conversion() -> None:
    """
    Verify CARLA waypoint sequence converts into matching 1D NumPy arrays
    (road_x, road_y, road_yaw) with negated Y and negated/normalized yaw.
    """
    carla_waypoints = [
        MockWaypoint(
            MockTransform(
                MockVector3D(0.0, 10.0, 0.0),
                MockRotation(yaw=0.0)
            )
        ),
        MockWaypoint(
            MockTransform(
                MockVector3D(5.0, 12.0, 0.5),
                MockRotation(yaw=30.0)
            )
        ),
        MockWaypoint(
            MockTransform(
                MockVector3D(10.0, -4.0, 1.0),
                MockRotation(yaw=-45.0)
            )
        ),
    ]

    road_x, road_y, road_yaw = convert_carla_waypoints(carla_waypoints)

    assert road_x.shape == (3,)
    assert road_y.shape == (3,)
    assert road_yaw.shape == (3,)

    assert np.allclose(road_x, [0.0, 5.0, 10.0], atol=1e-12)
    assert np.allclose(road_y, [-10.0, -12.0, 4.0], atol=1e-12)
    assert np.allclose(
        road_yaw,
        [0.0, -math.radians(30.0), math.radians(45.0)],
        atol=1e-12
    )

    with pytest.raises(ValueError, match="must not be empty"):
        convert_carla_waypoints([])


# ==================================================
# 5. SPEED EXTRACTION TESTS
# ==================================================

def test_speed_extraction() -> None:
    """
    Verify scalar speed extraction from both a CARLA Vehicle and a Vector3D
    velocity object in 3D and planar modes.
    """
    vehicle = MockCarlaVehicle(vx=3.0, vy=4.0, vz=12.0)

    speed_3d = extract_vehicle_speed(vehicle, planar_only=False)
    speed_2d = extract_vehicle_speed(vehicle, planar_only=True)
    speed_from_vec = extract_vehicle_speed(
        MockVector3D(3.0, -4.0, 0.0)
    )

    assert np.isclose(speed_3d, 13.0, atol=1e-12)
    assert np.isclose(speed_2d, 5.0, atol=1e-12)
    assert np.isclose(speed_from_vec, 5.0, atol=1e-12)


# ==================================================
# 6. REAR-AXLE COORDINATE CONVERSION TESTS
# ==================================================

@pytest.mark.parametrize(
    ("actor_x", "actor_y", "carla_yaw_deg", "expected_rx", "expected_ry"),
    [
        # Facing +X (yaw = 0 deg): rear axle is 1.425 m behind actor along -X
        (10.0, 20.0, 0.0, 10.0 - 1.425, -20.0),
        # Facing CARLA +Y / Controller -Y (CARLA yaw = +90 deg, ctrl yaw = -90 deg):
        # rear axle is at controller Y = -20.0 - (-1.425) = -18.575
        (10.0, 20.0, 90.0, 10.0, -20.0 + 1.425),
        # Facing CARLA -Y / Controller +Y (CARLA yaw = -90 deg, ctrl yaw = +90 deg):
        # rear axle is at controller Y = -20.0 - 1.425 = -21.425
        (10.0, 20.0, -90.0, 10.0, -20.0 - 1.425),
        # Facing -X (CARLA yaw = 180 deg): rear axle is at X = 10.0 + 1.425
        (10.0, 20.0, 180.0, 10.0 + 1.425, -20.0),
    ]
)
def test_rear_axle_coordinate_conversion(
    actor_x: float,
    actor_y: float,
    carla_yaw_deg: float,
    expected_rx: float,
    expected_ry: float
) -> None:
    """
    Verify that rear-axle offset is accurately extracted from CARLA wheel
    physics and transformed to the controller's rear-axle coordinate frame
    without assuming the actor origin is the rear axle.
    """
    vehicle = MockCarlaVehicle(
        x=actor_x,
        y=actor_y,
        yaw_deg=carla_yaw_deg,
        front_axle_x_m=1.45,
        rear_axle_x_m=-1.425,
        max_front_steer_deg=70.0
    )

    offset_x, offset_y_ctrl = get_rear_axle_offset_from_physics(
        vehicle.get_physics_control(),
        vehicle.get_transform()
    )
    assert np.isclose(offset_x, -1.425, atol=1e-9)
    assert np.isclose(offset_y_ctrl, 0.0, atol=1e-9)

    geom = extract_vehicle_geometry_from_physics(
        vehicle.get_physics_control(),
        vehicle.get_transform()
    )
    assert np.isclose(geom.wheelbase, 1.45 - (-1.425), atol=1e-9)

    rx, ry, ryaw = extract_rear_axle_pose(vehicle)
    assert np.isclose(rx, expected_rx, atol=1e-9)
    assert np.isclose(ry, expected_ry, atol=1e-9)
    assert np.isclose(
        math.cos(ryaw),
        math.cos(carla_to_controller_yaw(carla_yaw_deg)),
        atol=1e-9
    )


# ==================================================
# 7. STEERING SIGN CONVERSION TESTS
# ==================================================

def test_steering_sign_conversion() -> None:
    """
    Verify experimentally measured steering sign conversion:
    - Positive controller steering (+left turn) -> Negative CARLA steer
    - Negative controller steering (-right turn) -> Positive CARLA steer
    - Zero controller steering -> 0.0 CARLA steer
    - Uses physics-derived max front-wheel steering angle (e.g., 70 deg),
      NOT a hardcoded 30 deg assumption.
    """
    vehicle = MockCarlaVehicle(max_front_steer_deg=70.0)
    max_steer_rad = get_vehicle_max_steering_angle(vehicle)

    assert np.isclose(max_steer_rad, math.radians(70.0), atol=1e-12)

    # Left turn in controller (+14 deg) -> CARLA steer = -14 / 70 = -0.20
    ctrl_left_rad = math.radians(14.0)
    carla_steer_left = controller_steering_to_carla_steer(
        ctrl_left_rad,
        max_steer_rad
    )
    assert carla_steer_left < 0.0
    assert np.isclose(carla_steer_left, -0.20, atol=1e-12)

    # Right turn in controller (-14 deg) -> CARLA steer = +14 / 70 = +0.20
    ctrl_right_rad = math.radians(-14.0)
    carla_steer_right = controller_steering_to_carla_steer(
        ctrl_right_rad,
        max_steer_rad
    )
    assert carla_steer_right > 0.0
    assert np.isclose(carla_steer_right, 0.20, atol=1e-12)

    # Zero steering
    carla_steer_zero = controller_steering_to_carla_steer(0.0, max_steer_rad)
    assert carla_steer_zero == 0.0

    # Round-trip inverse conversion
    recovered_left = carla_steer_to_controller_steering(
        carla_steer_left,
        max_steer_rad
    )
    assert np.isclose(recovered_left, ctrl_left_rad, atol=1e-12)


# ==================================================
# 8. STEERING SATURATION TESTS
# ==================================================

def test_steering_saturation() -> None:
    """
    Verify normalized CARLA steering is clamped strictly to [-1.0, 1.0]
    when controller steering exceeds the maximum front-wheel angle.
    """
    max_steer_rad = math.radians(65.0)

    steer_large_left = controller_steering_to_carla_steer(
        math.radians(90.0),
        max_steer_rad
    )
    steer_large_right = controller_steering_to_carla_steer(
        math.radians(-90.0),
        max_steer_rad
    )

    assert np.isclose(steer_large_left, -1.0, atol=1e-12)
    assert np.isclose(steer_large_right, 1.0, atol=1e-12)


# ==================================================
# 9. FINITE NUMERICAL OUTPUTS & ADAPTER CLASS TESTS
# ==================================================

def test_finite_numerical_outputs_and_adapter_class() -> None:
    """
    Verify CarlaControllerAdapter produces finite numerical outputs across
    all state, waypoint, and control methods and rejects non-finite inputs.
    """
    vehicle = MockCarlaVehicle(
        x=-67.25,
        y=27.96,
        z=0.05,
        yaw_deg=15.0,
        vx=8.0,
        vy=1.5,
        vz=0.0,
        front_axle_x_m=1.45,
        rear_axle_x_m=-1.425,
        max_front_steer_deg=70.0
    )

    adapter = CarlaControllerAdapter(vehicle)
    state = adapter.get_controller_state()

    for key, value in state.items():
        assert math.isfinite(value), f"Non-finite state value for '{key}'"

    steer_cmd = adapter.to_carla_steer(math.radians(10.5))
    assert math.isfinite(steer_cmd)
    assert -1.0 <= steer_cmd <= 1.0

    vc = adapter.to_vehicle_control(
        controller_steering_angle=math.radians(-7.0),
        throttle=0.35,
        brake=0.0
    )
    assert np.isclose(float(vc.steer), 0.10, atol=1e-12)
    assert np.isclose(float(vc.throttle), 0.35, atol=1e-12)
    assert np.isclose(float(vc.brake), 0.0, atol=1e-12)

    vc_direct = create_carla_vehicle_control(
        controller_steering_angle=math.radians(7.0),
        maximum_front_wheel_steering_angle=math.radians(70.0),
        throttle=0.4,
        brake=0.0
    )
    assert np.isclose(float(vc_direct.steer), -0.10, atol=1e-12)
    assert np.isclose(float(vc_direct.throttle), 0.4, atol=1e-12)


    # Verify non-finite inputs raise ValueError
    with pytest.raises(ValueError, match="finite"):
        carla_to_controller_position(float("nan"), 0.0)

    with pytest.raises(ValueError, match="finite"):
        carla_to_controller_yaw(float("inf"))

    with pytest.raises(ValueError, match="finite"):
        controller_steering_to_carla_steer(float("nan"), 1.0)

    with pytest.raises(ValueError, match="positive"):
        controller_steering_to_carla_steer(0.1, 0.0)

    with pytest.raises(ValueError, match="finite"):
        estimate_rear_axle_position(0.0, 0.0, 0.0, float("nan"))

    with pytest.raises(ValueError, match="positive"):
        get_max_steering_angle_from_physics(
            MockPhysicsControl(
                wheels=[MockWheelPhysics(MockVector3D(0.0, 0.0, 0.0), 0.0)]
            )
        )
