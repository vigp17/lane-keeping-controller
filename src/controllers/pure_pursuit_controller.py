import numpy as np
from numpy.typing import NDArray

from vehicle import Vehicle


class PurePursuitController:
    """
    Pure Pursuit geometric path-tracking controller for autonomous lane keeping.

    Uses the rear-axle reference point (vehicle.x, vehicle.y) and selects a
    lookahead target point ahead along the reference path using cumulative path
    arc length.

    Control law:
        alpha = normalize_angle(atan2(target_y - y, target_x - x) - yaw)
        delta = atan2(2 * L * sin(alpha), Ld)

    where Ld is the actual Euclidean distance from the rear axle to the
    selected lookahead waypoint.

    Note:
        Target selection operates on discrete path waypoints (road_x, road_y).
        Waypoint spatial resolution causes a small discrepancy between the
        desired arc-length lookahead and the actual Euclidean distance Ld.
    """

    def __init__(
        self,
        wheelbase: float = 2.8,
        lookahead_distance: float = 10.0,
        min_lookahead: float = 3.0,
        lookahead_gain: float = 0.5,
        use_dynamic_lookahead: bool = True,
        L_min: float | None = None,
        K_v: float | None = None
    ) -> None:
        """
        Initialize the Pure Pursuit controller.

        Args:
            wheelbase: Vehicle wheelbase L in meters (> 0).
            lookahead_distance: Fixed lookahead distance in meters (> 0),
                used when use_dynamic_lookahead is False.
            min_lookahead: Minimum lookahead distance L_min in meters (> 0)
                for speed-dependent lookahead.
            lookahead_gain: Speed-proportional lookahead gain K_v in seconds (>= 0)
                for speed-dependent lookahead: Ld_desired = L_min + K_v * v.
            use_dynamic_lookahead: Whether to use speed-dependent lookahead
                by default.
            L_min: Optional alias for min_lookahead.
            K_v: Optional alias for lookahead_gain.
        """
        if L_min is not None:
            min_lookahead = L_min
        if K_v is not None:
            lookahead_gain = K_v

        if not np.isfinite(wheelbase) or wheelbase <= 0.0:
            raise ValueError(
                "wheelbase must be a finite positive number."
            )

        if not np.isfinite(lookahead_distance) or lookahead_distance <= 0.0:
            raise ValueError(
                "lookahead_distance must be a finite positive number."
            )

        if not np.isfinite(min_lookahead) or min_lookahead <= 0.0:
            raise ValueError(
                "min_lookahead (L_min) must be a finite positive number."
            )

        if not np.isfinite(lookahead_gain) or lookahead_gain < 0.0:
            raise ValueError(
                "lookahead_gain (K_v) must be a finite nonnegative number."
            )

        self.wheelbase: float = float(wheelbase)
        self.L: float = float(wheelbase)
        self.lookahead_distance: float = float(lookahead_distance)
        self.min_lookahead: float = float(min_lookahead)
        self.L_min: float = float(min_lookahead)
        self.lookahead_gain: float = float(lookahead_gain)
        self.K_v: float = float(lookahead_gain)
        self.use_dynamic_lookahead: bool = bool(use_dynamic_lookahead)

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
        road_y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Validate 1D reference path arrays and ensure finite coordinates.
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

        return rx, ry

    @classmethod
    def compute_cumulative_arc_length(
        cls,
        road_x: NDArray[np.float64],
        road_y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute cumulative arc length along the reference path waypoints.

        Args:
            road_x: 1D array of waypoint X coordinates (m).
            road_y: 1D array of waypoint Y coordinates (m).

        Returns:
            1D array of cumulative arc lengths s (m) starting at 0.0.
        """
        rx, ry = cls.validate_reference_path(road_x, road_y)

        segment_lengths = np.hypot(np.diff(rx), np.diff(ry))
        arc_length = np.concatenate(
            ([0.0], np.cumsum(segment_lengths))
        )

        if float(arc_length[-1]) <= 0.0:
            raise ValueError(
                "Reference path must have positive total arc length."
            )

        return np.asarray(arc_length, dtype=np.float64)

    def compute_desired_lookahead(
        self,
        velocity: float = 0.0,
        use_dynamic_lookahead: bool | None = None
    ) -> float:
        """
        Calculate the desired lookahead distance Ld_desired.

        For speed-dependent lookahead:
            Ld_desired = L_min + K_v * velocity
        For fixed lookahead:
            Ld_desired = lookahead_distance
        """
        if not np.isfinite(velocity) or velocity < 0.0:
            raise ValueError(
                "velocity must be a finite nonnegative number."
            )

        dynamic = (
            self.use_dynamic_lookahead
            if use_dynamic_lookahead is None
            else bool(use_dynamic_lookahead)
        )

        if dynamic:
            return float(
                self.min_lookahead + self.lookahead_gain * velocity
            )

        return float(self.lookahead_distance)

    def find_nearest_waypoint(
        self,
        rear_x: float,
        rear_y: float,
        road_x: NDArray[np.float64],
        road_y: NDArray[np.float64],
        min_index: int = 0
    ) -> int:
        """
        Find the index of the nearest reference waypoint to the rear axle.

        Args:
            rear_x: Rear-axle X coordinate (m).
            rear_y: Rear-axle Y coordinate (m).
            road_x: Reference path X coordinates (m).
            road_y: Reference path Y coordinates (m).
            min_index: Minimum waypoint index to search from to enforce
                monotonic forward progression along the path.

        Returns:
            Index of the nearest waypoint (>= min_index).
        """
        if not (np.isfinite(rear_x) and np.isfinite(rear_y)):
            raise ValueError(
                "Vehicle rear-axle coordinates must be finite numbers."
            )

        rx, ry = self.validate_reference_path(road_x, road_y)

        start_idx = max(0, min(int(min_index), rx.size - 1))

        dx = rx[start_idx:] - rear_x
        dy = ry[start_idx:] - rear_y

        distance_squared = dx ** 2 + dy ** 2
        nearest_offset = int(np.argmin(distance_squared))

        return start_idx + nearest_offset

    def select_target_waypoint(
        self,
        vehicle: Vehicle,
        road_x: NDArray[np.float64],
        road_y: NDArray[np.float64],
        desired_lookahead: float,
        min_index: int = 0
    ) -> tuple[int, int, float, float, float]:
        """
        Select the lookahead target waypoint ahead along the path using
        cumulative path arc length.

        Note:
            Discrete waypoint selection approximates continuous arc-length
            interpolation. Waypoint spacing determines how closely the
            selected waypoint matches desired_lookahead.

        Args:
            vehicle: Vehicle instance with rear-axle (x, y, yaw).
            road_x: Reference path X coordinates (m).
            road_y: Reference path Y coordinates (m).
            desired_lookahead: Desired arc-length lookahead distance (m).
            min_index: Minimum waypoint index for forward progression.

        Returns:
            Tuple of:
                - nearest_index: Index of nearest path waypoint to rear axle
                - target_index: Index of selected lookahead target waypoint
                - target_x: Target waypoint X coordinate (m)
                - target_y: Target waypoint Y coordinate (m)
                - actual_lookahead: Euclidean distance from rear axle to target (m)
        """
        if not np.isfinite(desired_lookahead) or desired_lookahead <= 0.0:
            raise ValueError(
                "desired_lookahead must be a finite positive number."
            )

        if not (
            np.isfinite(vehicle.x)
            and np.isfinite(vehicle.y)
            and np.isfinite(vehicle.yaw)
        ):
            raise ValueError(
                "Vehicle state (x, y, yaw) must contain finite numbers."
            )

        rx, ry = self.validate_reference_path(road_x, road_y)
        arc_length = self.compute_cumulative_arc_length(rx, ry)

        nearest_index = self.find_nearest_waypoint(
            rear_x=vehicle.x,
            rear_y=vehicle.y,
            road_x=rx,
            road_y=ry,
            min_index=min_index
        )

        target_arc_length = (
            float(arc_length[nearest_index]) + desired_lookahead
        )

        # Find the first waypoint ahead of nearest_index reaching target_arc_length
        relative_index = int(
            np.searchsorted(
                arc_length[nearest_index:],
                target_arc_length,
                side="left"
            )
        )
        target_index = min(
            nearest_index + relative_index,
            rx.size - 1
        )

        # Avoid selecting waypoints behind or coincident with the vehicle
        # during forward tracking whenever further waypoints exist
        heading_cos = float(np.cos(vehicle.yaw))
        heading_sin = float(np.sin(vehicle.yaw))

        while target_index < rx.size - 1:
            dx = float(rx[target_index] - vehicle.x)
            dy = float(ry[target_index] - vehicle.y)
            forward_projection = dx * heading_cos + dy * heading_sin
            dist = float(np.hypot(dx, dy))

            if forward_projection > 0.0 and dist > 1e-6:
                break

            target_index += 1

        target_x = float(rx[target_index])
        target_y = float(ry[target_index])

        actual_lookahead = float(
            np.hypot(target_x - vehicle.x, target_y - vehicle.y)
        )

        return (
            nearest_index,
            target_index,
            target_x,
            target_y,
            actual_lookahead
        )

    def compute_with_details(
        self,
        vehicle: Vehicle,
        road_x: NDArray[np.float64],
        road_y: NDArray[np.float64],
        velocity: float = 0.0,
        use_dynamic_lookahead: bool | None = None,
        min_index: int = 0
    ) -> dict[str, float | int]:
        """
        Compute the Pure Pursuit steering command and diagnostic details.

        Returns:
            Dictionary containing:
                - steering: Requested steering angle delta (rad)
                - requested_steering: Requested steering angle delta (rad)
                - target_index: Selected lookahead waypoint index
                - nearest_index: Nearest waypoint index to rear axle
                - actual_lookahead_distance: Euclidean distance Ld (m)
                - desired_lookahead_distance: Desired lookahead Ld_desired (m)
                - heading_to_target_error: Normalized angle alpha (rad)
                - alpha: Normalized angle alpha (rad)
                - target_x: Selected target X coordinate (m)
                - target_y: Selected target Y coordinate (m)
        """
        desired_lookahead = self.compute_desired_lookahead(
            velocity=velocity,
            use_dynamic_lookahead=use_dynamic_lookahead
        )

        (
            nearest_index,
            target_index,
            target_x,
            target_y,
            actual_lookahead
        ) = self.select_target_waypoint(
            vehicle=vehicle,
            road_x=road_x,
            road_y=road_y,
            desired_lookahead=desired_lookahead,
            min_index=min_index
        )

        dx = target_x - vehicle.x
        dy = target_y - vehicle.y

        heading_cos = float(np.cos(vehicle.yaw))
        heading_sin = float(np.sin(vehicle.yaw))
        forward_projection = dx * heading_cos + dy * heading_sin

        # Safe endpoint handling when vehicle reaches or passes the final waypoint
        if actual_lookahead <= 1e-6 or (
            target_index == len(road_x) - 1 and forward_projection <= 0.0
        ):
            rx, ry = self.validate_reference_path(road_x, road_y)
            end_dx = float(rx[-1] - rx[-2])
            end_dy = float(ry[-1] - ry[-2])
            seg_len = max(float(np.hypot(end_dx, end_dy)), 1e-9)
            tangent_x = end_dx / seg_len
            tangent_y = end_dy / seg_len

            # Project a virtual point ahead along the terminal path tangent
            proj_along_end = (
                (vehicle.x - float(rx[-1])) * tangent_x
                + (vehicle.y - float(ry[-1])) * tangent_y
            )
            virtual_s = max(0.0, proj_along_end) + desired_lookahead
            virtual_tx = float(rx[-1]) + virtual_s * tangent_x
            virtual_ty = float(ry[-1]) + virtual_s * tangent_y

            v_dx = virtual_tx - vehicle.x
            v_dy = virtual_ty - vehicle.y
            effective_ld = max(
                float(np.hypot(v_dx, v_dy)),
                desired_lookahead
            )
            raw_alpha = float(np.arctan2(v_dy, v_dx) - vehicle.yaw)
            alpha = self.normalize_angle(raw_alpha)
            steering_command = float(
                np.arctan2(
                    2.0 * self.wheelbase * np.sin(alpha),
                    effective_ld
                )
            )
        else:
            raw_alpha = float(np.arctan2(dy, dx) - vehicle.yaw)
            alpha = self.normalize_angle(raw_alpha)
            steering_command = float(
                np.arctan2(
                    2.0 * self.wheelbase * np.sin(alpha),
                    actual_lookahead
                )
            )

        return {
            "steering": steering_command,
            "requested_steering": steering_command,
            "target_index": target_index,
            "nearest_index": nearest_index,
            "actual_lookahead_distance": actual_lookahead,
            "desired_lookahead_distance": desired_lookahead,
            "heading_to_target_error": alpha,
            "alpha": alpha,
            "target_x": target_x,
            "target_y": target_y
        }

    def compute(
        self,
        vehicle: Vehicle,
        road_x: NDArray[np.float64],
        road_y: NDArray[np.float64],
        velocity: float = 0.0,
        use_dynamic_lookahead: bool | None = None,
        min_index: int = 0
    ) -> float:
        """
        Compute the unconstrained Pure Pursuit steering command (radians).
        """
        details = self.compute_with_details(
            vehicle=vehicle,
            road_x=road_x,
            road_y=road_y,
            velocity=velocity,
            use_dynamic_lookahead=use_dynamic_lookahead,
            min_index=min_index
        )

        return float(details["steering"])
