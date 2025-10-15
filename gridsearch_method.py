import glob
import os

import colour
from colour.io import image
import numpy as np

from colour_checker_detection import (
    ROOT_RESOURCES_EXAMPLES,
    detect_colour_checkers_segmentation,
)

colour.plotting.colour_style()

colour.utilities.describe_environment()

path_dir="/home/girobat/Olive/foto olivo del  07.08.24/da01c0"
COLOUR_CHECKER_IMAGE_PATHS = glob.glob(
    os.path.join(path_dir, '*.jpg'))

COLOUR_CHECKER_IMAGES = [
    colour.cctf_decoding(colour.io.read_image(path))
    for path in COLOUR_CHECKER_IMAGE_PATHS
]

D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS[
    "ColorChecker24 - After November 2014"
]

colour_checker_rows = REFERENCE_COLOUR_CHECKER.rows
colour_checker_columns = REFERENCE_COLOUR_CHECKER.columns

# NOTE: The reference swatches values as produced by the "colour.XYZ_to_RGB"
# definition are linear by default.
# See https://github.com/colour-science/colour-checker-detection/discussions/59
# for more information.
REFERENCE_SWATCHES = colour.XYZ_to_RGB(
    colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
    "sRGB",
    REFERENCE_COLOUR_CHECKER.illuminant,
)

def compute_deltaE(image,method,degree):
    
    
    colour_checker_data=detect_colour_checkers_segmentation(
        image, additional_data=True
    )
    detected_swatches, swatch_masks, colour_checker_image, quadrilateral = colour_checker_data[0].values

    REFERENCE_SWATCHES_Lab= colour.XYZ_to_Lab(colour.RGB_to_XYZ(
        REFERENCE_SWATCHES, 'sRGB', D65))

    # 1 -- Difference between Corrected swatches and Reference swatches--
    #print("1. -- Difference between Corrected swatches and Reference swatches--")
    corrected_swatches = colour.colour_correction(detected_swatches, detected_swatches, REFERENCE_SWATCHES,method=method,degree=degree)

    # Convert to Lab for ΔE calculation
    corrected_swatches_Lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(
        corrected_swatches, 'sRGB', D65))

    delta_E_correctedswatches = colour.delta_E(corrected_swatches_Lab, REFERENCE_SWATCHES_Lab, method='CIE 2000')
    mean_delta_E_correctedswatches = np.mean(delta_E_correctedswatches)

    # 2 -- Difference between swatches detected from original images and reference swatches--
    #print("2. -- Difference between swatches detected from original images and reference swatches--")

    image_corrected=colour.colour_correction(
                image, detected_swatches, REFERENCE_SWATCHES,method=method,degree=degree
            )
    # Extract the colour checker from the corrected image, theoretically should have the same delta E as the corrected swatches
    try:
        colour_checker_data2=detect_colour_checkers_segmentation(
        image_corrected, additional_data=True)
        detected_swatches2, swatch_masks, colour_checker_image, quadrilateral = colour_checker_data2[0].values

        detected_swatches2_Lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(
        detected_swatches2, 'sRGB', D65))

        delta_E2 = colour.delta_E(detected_swatches2_Lab, REFERENCE_SWATCHES_Lab, method='CIE 2000')
        #print("ΔE (mean): {0:.4f}, ΔE (max): {1:.4f}".format(np.mean(delta_E2), np.max(delta_E2)))
        mean_delta_E2 = np.mean(delta_E2)

    except:
        print("Colour checker not detected in corrected image")
        mean_delta_E2=None

    return mean_delta_E_correctedswatches, mean_delta_E2


methods = [
    'Cheung 2004',
    'Finlayson 2015',
    'Vandermonde'
]
sets=["swatches",
     "corrected_image"]

grid = {method: {degree:{s: [] for s in sets} for degree in range(1,5)} for method in methods}

for method in methods:
    print(f"Method: {method}")
    for degree in range(1, 5):
        print(f"  Degree: {degree}")
        for img, path in zip(COLOUR_CHECKER_IMAGES, COLOUR_CHECKER_IMAGE_PATHS):
            mean_delta_E_correctedswatches, mean_delta_E2=compute_deltaE(img,method,degree)
            grid[method][degree]["swatches"].append(mean_delta_E_correctedswatches)
            grid[method][degree]["corrected_image"].append(mean_delta_E2)


import json
import os 

def save_dict_to_json(data_dict, filename):
    try:
        # 'w' means write mode
        with open(filename, 'w') as f:
            # indent=4 makes the JSON file human-readable (pretty-printed)
            # sort_keys=True is good practice for consistent file contents
            json.dump(data_dict, f, indent=4, sort_keys=True)
        print(f"✅ Dictionary successfully saved to **{filename}**")
        
    except IOError as e:
        print(f"❌ Error writing to file {filename}: {e}")



# The name of the file to save (will be created in the current directory)
file_to_save = "grid_results.json"

# Run the function
save_dict_to_json(grid, file_to_save)