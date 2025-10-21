import os
import glob
import shutil
import colour

root_dir = "/home/girobat/Olive/foto olivo del  07.08.24"
output_dir = "/home/girobat/Olive/foto_olivo_normalized"

from colour_checker_detection import (
    ROOT_RESOURCES_EXAMPLES,
    detect_colour_checkers_segmentation,
)

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


for dirpath, dirnames, filenames in os.walk(root_dir):
    # Compute relative path from root
    rel_path = os.path.relpath(dirpath, root_dir)

    # Create corresponding folder in output_dir
    out_folder = os.path.join(output_dir, rel_path)
    os.makedirs(out_folder, exist_ok=True)

    # Process all images in this folder
    images_paths = []
    images_paths.extend(glob.glob(os.path.join(dirpath, "*.jpg")))

    # Folder for images that fail correction
    not_detected_folder = os.path.join(out_folder, "not_detected")
    os.makedirs(not_detected_folder, exist_ok=True)

    for img_path in images_paths:
        # Normalization
        img = colour.cctf_decoding(colour.io.read_image(img_path))
        detected = detect_colour_checkers_segmentation(img, additional_data=True)
        if not detected:
            # Copy to not_detected folder with same filename
            dest_path = os.path.join(not_detected_folder, os.path.basename(img_path))
            shutil.copy(img_path, dest_path)
            continue

        # Example: save processed image to same structure
        # Here just copying as an example
        out_path = os.path.join(out_folder, os.path.basename(img_path))
        shutil.copy(img_path, out_path)


def Normalization(detected, method, degree):
    # Assuming there are only one colour checker in the image
    detected_swatches, swatch_masks, colour_checker_image, quadrilateral = detected[
        0
    ].values

    try:
        # Colour correction
        image_corrected = colour.colour_correction(
            img, detected_swatches, REFERENCE_SWATCHES, method=method, degree=degree
        )
        return image_corrected
    except Exception as e:
        print(f"Error during colour correction: {e}")
        return None
