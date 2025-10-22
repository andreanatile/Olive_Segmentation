import os
import json
import numpy as np
import colour
from colour_checker_detection import detect_colour_checkers_segmentation
import glob
import argparse


# --- CONSTANTS ---
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS[
    "ColorChecker24 - After November 2014"
]

REFERENCE_SWATCHES = colour.XYZ_to_RGB(
    colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
    "sRGB",
    REFERENCE_COLOUR_CHECKER.illuminant,
)


# --- FUNCTIONS ---
def compute_deltaE(image, method, degree):
    """
    Compute mean ΔE for a single image and correction method and degree.

    Parameters
    ----------
    image : ndarray
        The input image array.
    method : str
        Colour correction method of the following ["Cheung 2004", "Finlayson 2015", "Vandermonde"].
    degree : int
        Polynomial degree used in correction.

    Returns
    -------
    tuple(float, float or None)
        Mean ΔE for corrected swatches and for the swatches detected in corrected image.
    """
    colour_checker_data = detect_colour_checkers_segmentation(
        image, additional_data=True
    )
    detected_swatches, swatch_masks, colour_checker_image, quadrilateral = (
        colour_checker_data[0].values
    )

    ref_Lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(REFERENCE_SWATCHES, "sRGB", D65))

    # --- 1. Swatches ΔE ---
    corrected_swatches = colour.colour_correction(
        detected_swatches,
        detected_swatches,
        REFERENCE_SWATCHES,
        method=method,
        degree=degree,
    )
    corrected_Lab = colour.XYZ_to_Lab(
        colour.RGB_to_XYZ(corrected_swatches, "sRGB", D65)
    )
    deltaE_swatches = colour.delta_E(corrected_Lab, ref_Lab, method="CIE 2000")
    mean_deltaE_swatches = np.mean(deltaE_swatches)

    # --- 2. Corrected image ΔE ---
    try:
        image_corrected = colour.colour_correction(
            image, detected_swatches, REFERENCE_SWATCHES, method=method, degree=degree
        )
        colour_checker_data2 = detect_colour_checkers_segmentation(
            image_corrected, additional_data=True
        )
        detected_swatches2, *_ = colour_checker_data2[0].values
        detected_Lab2 = colour.XYZ_to_Lab(
            colour.RGB_to_XYZ(detected_swatches2, "sRGB", D65)
        )
        deltaE_image = colour.delta_E(detected_Lab2, ref_Lab, method="CIE 2000")
        mean_deltaE_image = np.mean(deltaE_image)
    except Exception:
        print("⚠️ Colour checker not detected in corrected image.")
        mean_deltaE_image = None

    return mean_deltaE_swatches, mean_deltaE_image


def deltaE_original(image):
    """
    Compute mean ΔE for the original image without any correction.

    Parameters
    ----------
    image : ndarray
        The input image array.

    Returns
    -------
    float
        Mean ΔE for the swatches detected in the original image.
    """
    colour_checker_data = detect_colour_checkers_segmentation(
        image, additional_data=True
    )
    detected_swatches, swatch_masks, colour_checker_image, quadrilateral = (
        colour_checker_data[0].values
    )

    ref_Lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(REFERENCE_SWATCHES, "sRGB", D65))

    detected_Lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(detected_swatches, "sRGB", D65))
    deltaE_image = colour.delta_E(detected_Lab, ref_Lab, method="CIE 2000")
    mean_deltaE_image = np.mean(deltaE_image)

    return mean_deltaE_image


def save_dict_to_json(data_dict, filename):
    """Save a dictionary to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(data_dict, f, indent=4, sort_keys=True)
        print(f"✅ Dictionary successfully saved to {filename}")
    except IOError as e:
        print(f"❌ Error writing to file {filename}: {e}")


if __name__ == "__main__":
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(
        description="Run grid search on colour checker images."
    )
    parser.add_argument(
        "--path_dir",
        type=str,
        default="foto olivo del  07.08.24/da01c0",
        help="Directory containing the JPG images.",
    )
    parser.add_argument(
        "--file_to_save",
        type=str,
        default="grid_results.json",
        help="Output JSON filename for saving results.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=4,
        help="Max polynomial degree for colour correction method.",
    )
    args = parser.parse_args()

    # --- Use arguments (or defaults) ---
    path_dir = args.path_dir
    file_to_save = args.file_to_save
    deg = args.degree

    print(f"  Using images from: {path_dir}")
    print(f" Results will be saved to: {file_to_save}")
    print(f" Max polynomial degree setted: {deg}")

    # --- 3️⃣ Load images ---
    COLOUR_CHECKER_IMAGE_PATHS = glob.glob(os.path.join(path_dir, "*.jpg"))
    COLOUR_CHECKER_IMAGES = [
        colour.cctf_decoding(colour.io.read_image(path))
        for path in COLOUR_CHECKER_IMAGE_PATHS
    ]

    # --- 4️⃣ Define grid parameters ---
    methods = ["Cheung 2004", "Finlayson 2015", "Vandermonde"]
    sets = ["swatches", "corrected_image"]

    grid = {
        method: {degree: {s: [] for s in sets} for degree in range(1, deg + 1)}
        for method in methods
    }
    grid["NoCorrection"] = [deltaE_original(img) for img in COLOUR_CHECKER_IMAGES]
    print(grid["NoCorrection"])
    # --- 5️⃣ Run grid search ---
    for method in methods:
        print(f"Method: {method}")
        for degree in range(1, deg + 1):
            print(f"  Degree: {degree}")
            for img, path in zip(COLOUR_CHECKER_IMAGES, COLOUR_CHECKER_IMAGE_PATHS):
                mean_sw, mean_img = compute_deltaE(img, method, degree)
                grid[method][degree]["swatches"].append(mean_sw)
                grid[method][degree]["corrected_image"].append(mean_img)

    # --- 6️⃣ Save results ---
    save_dict_to_json(grid, file_to_save)
