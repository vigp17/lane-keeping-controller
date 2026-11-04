from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from road import Road


# ============================================================
# 1. CURVED-ROAD CONFIGURATION
# ============================================================

ROAD_LENGTH: float = 300.0
AMPLITUDE: float = 2.0
FREQUENCY: float = 0.02
NUM_POINTS: int = 500

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. GENERATE CURVED-ROAD REFERENCE PROFILE
# ============================================================

def generate_curved_road_reference(
    length: float = ROAD_LENGTH,
    amplitude: float = AMPLITUDE,
    frequency: float = FREQUENCY,
    num_points: int = NUM_POINTS
) -> dict[str, NDArray[np.float64]]:

    road = Road(
        length=length,
        amplitude=amplitude,
        frequency=frequency
    )

    x, y_ref = road.generate_curved_road(
        num_points=num_points
    )

    yaw_ref = road.reference_heading(x)
    curvature = road.road_curvature(x)

    return {
        "x": x,
        "y_ref": y_ref,
        "yaw_ref": yaw_ref,
        "curvature": curvature
    }


# ============================================================
# 3. PRINT SUMMARY METRICS
# ============================================================

def print_road_summary(
    reference: dict[str, NDArray[np.float64]]
) -> None:

    x = reference["x"]
    y_ref = reference["y_ref"]
    yaw_ref = reference["yaw_ref"]
    curvature = reference["curvature"]

    max_y = float(np.max(np.abs(y_ref)))
    max_yaw_rad = float(np.max(np.abs(yaw_ref)))
    max_yaw_deg = float(np.degrees(max_yaw_rad))
    max_curvature = float(np.max(np.abs(curvature)))

    print("\nCURVED-ROAD REFERENCE SUMMARY")
    print("=" * 55)
    print(f"Road Length:                {x[-1]:.2f} m")
    print(f"Sinusoidal Amplitude (A):   {AMPLITUDE:.2f} m")
    print(f"Spatial Frequency (k):      {FREQUENCY:.4f} rad/m")
    print(f"Max Lateral Position:       {max_y:.4f} m")
    print(
        f"Max Reference Heading:      "
        f"{max_yaw_rad:.4f} rad ({max_yaw_deg:.2f} deg)"
    )
    print(f"Max Road Curvature:         {max_curvature:.6f} 1/m")
    print("=" * 55)


# ============================================================
# 4. PLOT CURVED-ROAD REFERENCE PROFILES
# ============================================================

def plot_curved_road(
    reference: dict[str, NDArray[np.float64]],
    show_plots: bool = True
) -> None:

    x = reference["x"]
    y_ref = reference["y_ref"]
    yaw_ref = reference["yaw_ref"]
    curvature = reference["curvature"]

    # ========================================================
    # PLOT 1: CURVED-ROAD CENTERLINE
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        x,
        y_ref,
        color="blue",
        linewidth=2,
        label="Curved-Road Centerline"
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        label="Straight Reference (y = 0)"
    )

    plt.xlabel("Longitudinal Position x (m)")
    plt.ylabel("Lateral Position y_ref (m)")
    plt.title("Sinusoidal Curved-Road Centerline")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "curved_road_centerline.png",
        dpi=300
    )

    # ========================================================
    # PLOT 2: REFERENCE HEADING VS LONGITUDINAL POSITION
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        x,
        np.degrees(yaw_ref),
        color="green",
        linewidth=2,
        label="Reference Heading (deg)"
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Longitudinal Position x (m)")
    plt.ylabel("Reference Heading yaw_ref (degrees)")
    plt.title("Reference Heading vs Longitudinal Position")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "curved_road_heading.png",
        dpi=300
    )

    # ========================================================
    # PLOT 3: ROAD CURVATURE VS LONGITUDINAL POSITION
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        x,
        curvature,
        color="purple",
        linewidth=2,
        label="Road Curvature (1/m)"
    )

    plt.axhline(
        y=0.0,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Longitudinal Position x (m)")
    plt.ylabel("Curvature (1/m)")
    plt.title("Road Curvature vs Longitudinal Position")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "curved_road_curvature.png",
        dpi=300
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")


# ============================================================
# 5. MAIN FUNCTION
# ============================================================

def main(show_plots: bool = True) -> None:

    reference = generate_curved_road_reference()

    print_road_summary(reference)

    plot_curved_road(
        reference,
        show_plots=show_plots
    )


if __name__ == "__main__":
    main()
