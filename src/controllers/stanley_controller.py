import numpy as np
from numpy.typing import NDArray

from vehicle import Vehicle


class StanleyController:
    """
    Stanley geometric path-tracking controller for autonomous lane keeping.

    The controller measures signed cross-track error from the vehicle's
    front axle to the nearest waypoint on the reference path and combines
    heading alignment error with a velocity-scaled cross-track correction:

        delta(t) = psi_e(t) + atan2(k * e_fa(t), v(t) + epsilon)

    Note:
        Nearest-point lookup is performed over discrete waypoints
        (road_x, road_y). Waypoint spatial resolution directly affects
        cross-track error accuracy.
    """

    def __init__(
        self,
        stanley_gain: float = 1.5,
        epsilon: float = 1e-3
    ) -> None:
        """
        Initialize the Stanley geometric controller.

        Args:
            stanley_gain: Cross-track error proportional gain k (> 0).
            epsilon: Low-speed softening / regularization constant (> 0)
                to prevent division by zero as velocity approaches zero.
        """
        if not np.isfinite(stanley_gain) or stanley_gain <= 0.0:
            raise ValueError(
                "stanley_gain must be a finite positive number."
            )

        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(
                "epsilon must be a finite positive number."
            )

        self.stanley_gain: float = float(stanley_gain)
        self.k: float = float(stanley_gain)
        self.epsilon: float = float(epsilon)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize an angle in radians to the principal interval [-pi, pi].
        """
        return float(
            np.arctan2(np.sin(angle), np.cos(angle))
        )

    @staticmethod
    def validate_reference_path(
        road_x: NDArray[np.float64],
        road_y: NDArray[np.float64],
        road_yaw: NDArray[np.float64] | None = None
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64]
    ]:
        """
        Validate reference path arrays and compute tangent headings if omitted.

        Args:
            road_x: 1D array of reference X coordinates (m).
            road_y: 1D array of reference Y coordinates (m).
            road_yaw: Optional 1D array of reference headings (rad).

        Returns:
            Tuple of validated (road_x, road_y, road_yaw) float64 arrays.
        """
        rx = np.asarray(road_x, dtype=np.float64)
        ry = np.asarray(road_y, dtype=np.float64)

        if rx.ndim != 1 or ry.ndim != 1:
            raise ValueError(
                "Reference path arrays road_x and road_y must be 1D."
            )

        if rx.size < 2 or rx.size != ry.size:
            raise ValueError(
                "road_x and road_y must have matching lengths of at least 2."
            )

        if not (np.all(np.isfinite(rx)) and np.all(np.isfinite(ry))):
            raise ValueError(
                "road_x and road_y must contain only finite numerical values."
            )

        if road_yaw is None:
            dx = np.gradient(rx)
            dy = np.gradient(ry)
            ryaw = np.asarray(
                np.arctan2(dy, dx),
                dtype=np.float64
            )
        else:
            ryaw = np.asarray(road_yaw, dtype=np.float64)
            if ryaw.ndim != 1 or ryaw.size != rx.size:
                raise ValueError(
                    "road_yaw must be a 1D array matching road_x length."
                )
            if not np.all(np.isfinite(ryaw)):
                raise ValueError(
                    "road_yaw must contain only finite numerical values."
                )

        return rx, ry, ryaw

    def find_nearest_waypoint(
        self,
        front_x: float,
        front_y: float,
        road_x: NDArray[np.float64],
        road_y: NDArray[np.float64],
        road_yaw: NDArray[np.float64] | None = None
    ) -> tuple[int, float, float, float]:
        """
        Find the nearest reference waypoint to the vehicle's front axle.

        Note:
            This discrete nearest-waypoint search approximates continuous
            orthogonal projection onto the curve. Denser waypoint spacing
            improves cross-track error accuracy.

        Args:
            front_x: Front-axle X position in global coordinates (m).
            front_y: Front-axle Y position in global coordinates (m).
            road_x: Reference path X coordinates (m).
            road_y: Reference path Y coordinates (m).
            road_yaw: Optional reference path headings (rad).

        Returns:
            Tuple of (nearest_index, ref_x, ref_y, ref_yaw).
        """
        if not (np.isfinite(front_x) and np.isfinite(front_y)):
            raise ValueError(
                "Front-axle coordinates must be finite numbers."
            )

        rx, ry, ryaw = self.validate_reference_path(
            road_x,
            road_y,
            road_yaw
        )

        dx = rx - front_x
        dy = ry - front_y

        distance_squared = dx ** 2 + dy ** 2

        nearest_index = int(np.argmin(distance_squared))

        ref_x = float(rx[nearest_index])
        ref_y = float(ry[nearest_index])
        ref_yaw = float(ryaw[nearest_index])

        return nearest_index, ref_x, ref_y, ref_yaw

    def compute_cross_track_error(
        self,
        front_x: float,
        front_y: float,
        ref_x: float,
        ref_y: float,
        ref_yaw: float
    ) -> float:
        """
        Calculate signed cross-track error using the path's local left normal.

        For path tangent t = [cos(ref_yaw), sin(ref_yaw)], the left normal is:
            n = [-sin(ref_yaw), cos(ref_yaw)]

        Signed cross-track error is defined as:
            e_fa = dot(reference_point - front_axle_position, n)

        When the vehicle is to the left of a forward path (y > y_ref),
        e_fa < 0, producing a negative (rightward) corrective steering angle.
        """
        normal_x = -np.sin(ref_yaw)
        normal_y = np.cos(ref_yaw)

        error_x = ref_x - front_x
        error_y = ref_y - front_y

        cross_track_error = (
            error_x * normal_x + error_y * normal_y
        )

        return float(cross_track_error)

    def compute_heading_error(
        self,
        ref_yaw: float,
        vehicle_yaw: float
    ) -> float:
        """
        Calculate normalized heading error (reference_heading - vehicle_yaw)
        wrapped to [-pi, pi].
        """
        return self.normalize_angle(ref_yaw - vehicle_yaw)

    def compute_steering_from_errors(
        self,
        heading_error: float,
        cross_track_error: float,
        velocity: float
    ) -> float:
        """
        Evaluate the Stanley control law from precomputed errors.

        Args:
            heading_error: Heading error (ref_yaw - vehicle_yaw) in radians.
            cross_track_error: Signed front-axle cross-track error in meters.
            velocity: Nonnegative forward vehicle speed in m/s.

        Returns:
            Unconstrained steering angle command in radians.
        """
        if not np.isfinite(velocity) or velocity < 0.0:
            raise ValueError(
                "velocity must be a finite nonnegative number."
            )

        normalized_heading_error = self.normalize_angle(
            heading_error
        )

        cross_track_steering = float(
            np.arctan2(
                self.stanley_gain * cross_track_error,
                velocity + self.epsilon
            )
        )

        return float(
            normalized_heading_error + cross_track_steering
        )

    def compute_with_details(
        self,
        vehicle: Vehicle,
        road_x: NDArray[np.float64],
        road_y: NDArray[np.float64],
        velocity: float,
        road_yaw: NDArray[np.float64] | None = None
    ) -> dict[str, float | int]:
        """
        Compute the Stanley steering command and intermediate geometric terms.

        Args:
            vehicle: Vehicle instance providing rear-axle state and front-axle
                geometry via get_front_axle_position().
            road_x: Reference path X coordinates (m).
            road_y: Reference path Y coordinates (m).
            velocity: Nonnegative forward vehicle speed (m/s).
            road_yaw: Optional reference path headings (rad).

        Returns:
            Dictionary containing:
                - steering: Unconstrained steering command (rad)
                - cross_track_error: Signed front-axle cross-track error (m)
                - heading_error: Normalized heading error (rad)
                - ref_x: Nearest waypoint X coordinate (m)
                - ref_y: Nearest waypoint Y coordinate (m)
                - ref_yaw: Nearest waypoint heading (rad)
                - front_x: Front-axle X coordinate (m)
                - front_y: Front-axle Y coordinate (m)
                - nearest_index: Index of nearest waypoint
        """
        front_x, front_y = vehicle.get_front_axle_position()

        nearest_index, ref_x, ref_y, ref_yaw = (
            self.find_nearest_waypoint(
                front_x=front_x,
                front_y=front_y,
                road_x=road_x,
                road_y=road_y,
                road_yaw=road_yaw
            )
        )

        cross_track_error = self.compute_cross_track_error(
            front_x=front_x,
            front_y=front_y,
            ref_x=ref_x,
            ref_y=ref_y,
            ref_yaw=ref_yaw
        )

        heading_error = self.compute_heading_error(
            ref_yaw=ref_yaw,
            vehicle_yaw=vehicle.yaw
        )

        steering_command = self.compute_steering_from_errors(
            heading_error=heading_error,
            cross_track_error=cross_track_error,
            velocity=velocity
        )

        return {
            "steering": steering_command,
            "cross_track_error": cross_track_error,
            "heading_error": heading_error,
            "ref_x": ref_x,
            "ref_y": ref_y,
            "ref_yaw": ref_yaw,
            "front_x": front_x,
            "front_y": front_y,
            "nearest_index": nearest_index
        }

    def compute(
        self,
        vehicle: Vehicle,
        road_x: NDArray[np.float64],
        road_y: NDArray[np.float64],
        velocity: float,
        road_yaw: NDArray[np.float64] | None = None
    ) -> float:
        """
        Compute the unconstrained Stanley steering angle command (radians).
        """
        details = self.compute_with_details(
            vehicle=vehicle,
            road_x=road_x,
            road_y=road_y,
            velocity=velocity,
            road_yaw=road_yaw
        )

        return float(details["steering"])
