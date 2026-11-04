import numpy as np
from scipy.linalg import solve_discrete_are


class AugmentedLQRController:

    def __init__(
        self,
        velocity: float,
        wheelbase: float,
        dt: float
    ) -> None:

        self.velocity = velocity
        self.wheelbase = wheelbase
        self.dt = dt

        v = velocity
        L = wheelbase

        # ==================================
        # 1. Discrete State-Space Model
        # ==================================

        self.Ad = np.array([
            [1.0, v * dt, (v**2 * dt**2) / (2 * L)],
            [0.0, 1.0, (v * dt) / L],
            [0.0, 0.0, 1.0]
        ])

        self.Bd = np.array([
            [(v**2 * dt**3) / (6 * L)],
            [(v * dt**2) / (2 * L)],
            [dt]
        ])

        # ==================================
        # 2. LQR Cost Matrices
        # ==================================

        self.Q = np.diag([
            10.0,   # Lateral error penalty
            5.0,    # Heading error penalty
            50.0    # Steering angle penalty
        ])

        self.R = np.array([
            [100.0]  # Steering-rate penalty
        ])

        # ==================================
        # 3. Solve Riccati Equation
        # ==================================

        P = solve_discrete_are(
            self.Ad,
            self.Bd,
            self.Q,
            self.R
        )

        # ==================================
        # 4. Calculate LQR Gain
        # ==================================

        self.K = np.linalg.solve(
            self.R + self.Bd.T @ P @ self.Bd,
            self.Bd.T @ P @ self.Ad
        )

    def compute(
        self,
        lateral_error: float,
        heading_error: float,
        steering_angle: float
    ) -> float:

        state = np.array([
            [lateral_error],
            [heading_error],
            [steering_angle]
        ])

        # LQR calculates steering rate
        steering_rate = -self.K @ state

        return float(steering_rate.item())


if __name__ == "__main__":

    controller = AugmentedLQRController(
        velocity=30.0,
        wheelbase=2.8,
        dt=0.1
    )

    print("Discrete A Matrix:")
    print(controller.Ad)

    print("\nDiscrete B Matrix:")
    print(controller.Bd)

    print("\nAugmented LQR Gain Matrix K:")
    print(controller.K)

    steering_rate = controller.compute(
        lateral_error=2.0,
        heading_error=0.0,
        steering_angle=0.0
    )

    print(
        f"\nRequested Steering Rate: "
        f"{np.degrees(steering_rate):.4f} deg/s"
    )