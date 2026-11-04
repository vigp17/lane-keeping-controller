from typing import overload

import numpy as np
from numpy.typing import NDArray


class Road:

    def __init__(
        self,
        length: float = 100.0,
        amplitude: float = 2.0,
        frequency: float = 0.02,
        A: float | None = None,
        k: float | None = None
    ) -> None:

        if A is not None:
            amplitude = A
        if k is not None:
            frequency = k

        self.length: float = length
        self.amplitude: float = amplitude
        self.frequency: float = frequency
        self.A: float = amplitude
        self.k: float = frequency

    # ==========================================
    # 1. Straight-Road Generation
    # ==========================================

    def generate_straight_road(
        self,
        num_points: int = 500
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        x = np.linspace(
            0.0,
            self.length,
            num_points,
            dtype=np.float64
        )

        y = np.zeros_like(x)

        return x, y

    # ==========================================
    # 2. Sinusoidal Curved-Road Generation
    # ==========================================

    def generate_curved_road(
        self,
        num_points: int = 500
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        x = np.linspace(
            0.0,
            self.length,
            num_points,
            dtype=np.float64
        )

        y = self.reference_lateral_position(x)

        return x, y

    # ==========================================
    # 3. Reference Lateral Position
    # ==========================================

    @overload
    def reference_lateral_position(
        self,
        x: float
    ) -> float: ...

    @overload
    def reference_lateral_position(
        self,
        x: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def reference_lateral_position(
        self,
        x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:

        y_ref = self.A * np.sin(self.k * x)

        if np.ndim(x) == 0:
            return float(y_ref)

        return np.asarray(y_ref, dtype=np.float64)

    # ==========================================
    # 4. Reference Heading
    # ==========================================

    @overload
    def reference_heading(
        self,
        x: float
    ) -> float: ...

    @overload
    def reference_heading(
        self,
        x: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def reference_heading(
        self,
        x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:

        yaw_ref = np.arctan(
            self.A * self.k * np.cos(self.k * x)
        )

        if np.ndim(x) == 0:
            return float(yaw_ref)

        return np.asarray(yaw_ref, dtype=np.float64)

    # ==========================================
    # 5. Road Curvature
    # ==========================================

    @overload
    def road_curvature(
        self,
        x: float
    ) -> float: ...

    @overload
    def road_curvature(
        self,
        x: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def road_curvature(
        self,
        x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:

        dy_dx = self.A * self.k * np.cos(self.k * x)
        d2y_dx2 = (
            -self.A * (self.k ** 2) * np.sin(self.k * x)
        )

        curvature = d2y_dx2 / (
            (1.0 + dy_dx ** 2) ** (3.0 / 2.0)
        )

        if np.ndim(x) == 0:
            return float(curvature)

        return np.asarray(curvature, dtype=np.float64)

    @overload
    def curvature(
        self,
        x: float
    ) -> float: ...

    @overload
    def curvature(
        self,
        x: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def curvature(
        self,
        x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:

        return self.road_curvature(x)
