import numpy as np


class Vehicle:

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        wheelbase: float = 2.8
    ) -> None:

        self.x: float = x
        self.y: float = y
        self.yaw: float = yaw
        self.wheelbase: float = wheelbase

    def update(
        self,
        velocity: float,
        steering_angle: float,
        dt: float
    ) -> None:

        # Update vehicle position
        self.x += velocity * np.cos(self.yaw) * dt
        self.y += velocity * np.sin(self.yaw) * dt

        # Update vehicle heading
        self.yaw += (
            velocity / self.wheelbase
        ) * np.tan(steering_angle) * dt

    def get_front_axle_position(self) -> tuple[float, float]:

        front_x = self.x + self.wheelbase * np.cos(self.yaw)

        front_y = self.y + self.wheelbase * np.sin(self.yaw)

        return float(front_x), float(front_y)
