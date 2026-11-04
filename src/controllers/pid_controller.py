class PIDController:

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float
    ) -> None:

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.previous_error = 0.0

    def compute(
        self,
        error: float,
        dt: float
    ) -> float:

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * dt
        integral_term = self.ki * self.integral

        # Derivative term
        derivative = (
            error - self.previous_error
        ) / dt

        derivative_term = self.kd * derivative

        # PID output
        output = (
            proportional
            + integral_term
            + derivative_term
        )

        # Store previous error
        self.previous_error = error

        return output