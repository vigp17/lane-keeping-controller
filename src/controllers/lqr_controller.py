import numpy as np
from scipy.linalg import solve_discrete_are


class LQRController:

    def __init__(
        self,
        velocity: float,
        wheelbase: float,
        dt: float
    ) -> None:

        self.velocity = velocity
        self.wheelbase = wheelbase
        self.dt = dt

        # ----------------------------------
        # Continuous-time state-space model
        # ----------------------------------

        self.A = np.array([
            [0.0, velocity],
            [0.0, 0.0]
        ])

        self.B = np.array([
            [0.0],
            [velocity / wheelbase]
        ])

        # ----------------------------------
        # Discrete-time state-space model
        # ----------------------------------

        self.Ad = np.array([
            [1.0, velocity * dt],
            [0.0, 1.0]
        ])

        self.Bd = np.array([
            [0.5 * velocity**2 * dt**2 / wheelbase],
            [velocity * dt / wheelbase]
        ])

        # ----------------------------------
        # LQR cost matrices
        # ----------------------------------

        self.Q = np.diag([10.0, 1.0])

        self.R = np.array([[100.0]])

        # ----------------------------------
        # Solve discrete Riccati equation
        # ----------------------------------

        P = solve_discrete_are(
            self.Ad,
            self.Bd,
            self.Q,
            self.R
        )

        # ----------------------------------
        # Calculate optimal feedback gain
        # ----------------------------------

        self.K = np.linalg.solve(
            self.R + self.Bd.T @ P @ self.Bd,
            self.Bd.T @ P @ self.Ad
        )

    def compute(
        self,
        lateral_error: float,
        heading_error: float
    ) -> float:

        state = np.array([
            [lateral_error],
            [heading_error]
        ])

        steering = -self.K @ state

        return float(steering.item())


if __name__ == "__main__":

    controller = LQRController(
        velocity=30.0,
        wheelbase=2.8,
        dt=0.1
    )

    print("Discrete A Matrix:")
    print(controller.Ad)

    print("\nDiscrete B Matrix:")
    print(controller.Bd)

    print("\nLQR Gain Matrix K:")
    print(controller.K)

    steering = controller.compute(
        lateral_error=2.0,
        heading_error=0.0
    )

    print(
        f"\nSteering Command: "
        f"{np.degrees(steering):.4f} degrees"
    )