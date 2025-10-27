import argparse
import cv2
import imageio
import numpy as np
import os
import glob
import json
from Normalization.scripts.Normalizer import Normalization_sw
import colour


"""manual_swatches_correction.py

Usage:
    python manual_swatches_correction.py \
        --input-dir /path/to/input_imgs \
        --output-dir /path/to/output_imgs \
        --output-json /path/to/output.json \
        --method "Cheung 2004" --degree 3

This script interactively lets you select ColorChecker patches from images,
saves the sampled RGBs into a JSON and applies the normalization function
`Normalization_sw` to produce corrected images.
"""

# --- DEFAULT CONFIG ---
DEFAULT_N_PATCHES = 24
DEFAULT_DISPLAY_SCALE = 0.5


json_data = {}


def select_patches(
    image_path, n_patches=DEFAULT_N_PATCHES, display_scale=DEFAULT_DISPLAY_SCALE
):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Cannot read {image_path}")
        return None

    orig_img = img.copy()
    img_display = cv2.resize(
        orig_img.copy(), (0, 0), fx=display_scale, fy=display_scale
    )
    completed_rects = []  # stores display coordinates
    patches = []
    points = []
    drawing = False

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, points, img_display, patches, completed_rects
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp = img_display.copy()
            for rect in completed_rects:
                cv2.rectangle(temp, rect[0], rect[1], (0, 255, 0), 2)
            cv2.rectangle(temp, points[0], (x, y), (0, 255, 0), 1)
            cv2.imshow("Select patches", temp)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            points.append((x, y))
            x1, y1 = int(points[0][0] / display_scale), int(
                points[0][1] / display_scale
            )
            x2, y2 = int(points[1][0] / display_scale), int(
                points[1][1] / display_scale
            )
            roi = orig_img[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]
            mean_color = roi.mean(axis=(0, 1))
            patches.append(mean_color[::-1].tolist())  # RGB
            print(f"Patch {len(patches)} avg RGB: {patches[-1]}")
            completed_rects.append((points[0], points[1]))
            img_display[:] = cv2.resize(
                orig_img.copy(), (0, 0), fx=display_scale, fy=display_scale
            )
            for rect in completed_rects:
                cv2.rectangle(img_display, rect[0], rect[1], (0, 255, 0), 2)
            cv2.imshow("Select patches", img_display)

    cv2.namedWindow("Select patches", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select patches", on_mouse)

    print(f"\n🖼️ Processing: {os.path.basename(image_path)}")
    print(f"🖱️ Draw rectangles around each ColorChecker patch ({n_patches} total).")
    print("Press 's' to save, 'q' to skip, or 'r' to reset.")
    zoom = 1.0
    zoom_step = 0.1
    while True:
        cv2.imshow("Select patches", img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("+") or key == ord("="):
            zoom = min(zoom + zoom_step, 3.0)
        elif key == ord("-") or key == ord("_"):
            zoom = max(zoom - zoom_step, 0.2)
        # Save
        elif key == ord("s") or len(patches) == n_patches:
            break
        # Skip
        elif key == ord("q"):
            patches = []
            break
        # Reset
        elif key == ord("r"):
            print("🔄 Resetting selections...")
            patches.clear()
            completed_rects.clear()
            img_display[:] = cv2.resize(
                orig_img.copy(), (0, 0), fx=display_scale, fy=display_scale
            )
            cv2.imshow("Select patches", img_display)

    cv2.destroyAllWindows()
    return np.array(patches) if len(patches) == n_patches else None


def process_images(
    input_dir,
    output_dir,
    output_json,
    method="Cheung 2004",
    degree=3,
    n_patches=DEFAULT_N_PATCHES,
    display_scale=DEFAULT_DISPLAY_SCALE,
    dry_run=False,
):
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    local_json = {}

    for img_path in image_paths:
        result = select_patches(
            img_path, n_patches=n_patches, display_scale=display_scale
        )
        if result is None:
            print(f"⚠️ Skipped {os.path.basename(img_path)}")
            continue

        result = np.array(result) / 255.0
        local_json[os.path.basename(img_path)] = result.tolist()

        if dry_run:
            print(
                f"Dry-run: would normalize {os.path.basename(img_path)} with {method}, degree={degree}"
            )
            continue

        img = colour.cctf_decoding(colour.io.read_image(img_path))
        corrected_img = Normalization_sw(img, result, method=method, degree=degree)
        if corrected_img is not None:
            corrected_encoded = colour.cctf_encoding(np.clip(corrected_img, 0, 1))
            corrected_rgb = (corrected_encoded * 255).astype(np.uint8)
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            os.makedirs(output_dir, exist_ok=True)
            imageio.imwrite(output_path, corrected_rgb)
            print(f"💾 Corrected image saved at {output_path}")
        print(f"✅ Saved RGB values for {os.path.basename(img_path)}")

    with open(output_json, "w") as f:
        json.dump(local_json, f, indent=4)

    print(f"\n💾 All results saved in {output_json}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive ColorChecker swatch selection and normalization"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory with undetected JPG images"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory containing corrected images"
    )
    parser.add_argument(
        "--output-json", required=True, help="Path to output JSON file for swatch RGBs"
    )
    parser.add_argument(
        "--method",
        default="Cheung 2004",
        help="Normalization method to pass to Normalization_sw",
    )
    parser.add_argument(
        "--degree", type=int, default=3, help="Degree parameter for Normalization_sw"
    )
    parser.add_argument(
        "--n-patches",
        type=int,
        default=DEFAULT_N_PATCHES,
        help="Number of patches to select (default: 24)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_DISPLAY_SCALE,
        help="Display scale for selection window",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only collect swatches and save JSON without writing corrected images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_json=args.output_json,
        method=args.method,
        degree=args.degree,
        n_patches=args.n_patches,
        display_scale=args.scale,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
