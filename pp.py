import cv2
import imageio
import numpy as np
import os
import glob
import json
from Normalization_script import Normalization_sw
import colour

# --- CONFIG ---
N_PATCHES = 24
INPUT_DIR = "/home/girobat/Olive/Corrected/foto 11.09.24 olivo università/not_detected"
OUTPUT_JSON = "/home/girobat/Olive/Corrected/foto 11.09.24 olivo università/not_detected/colorchecker_rgb.json"
OUTPUT_DIR = "/home/girobat/Olive/Corrected/foto 11.09.24 olivo università"
DISPLAY_SCALE = 0.5

json_data = {}


def select_patches(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Cannot read {image_path}")
        return None

    orig_img = img.copy()
    img_display = cv2.resize(
        orig_img.copy(), (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE
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
            x1, y1 = int(points[0][0] / DISPLAY_SCALE), int(
                points[0][1] / DISPLAY_SCALE
            )
            x2, y2 = int(points[1][0] / DISPLAY_SCALE), int(
                points[1][1] / DISPLAY_SCALE
            )
            roi = orig_img[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]
            mean_color = roi.mean(axis=(0, 1))
            patches.append(mean_color[::-1].tolist())  # RGB
            print(f"Patch {len(patches)} avg RGB: {patches[-1]}")
            completed_rects.append((points[0], points[1]))
            img_display[:] = cv2.resize(
                orig_img.copy(), (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE
            )
            for rect in completed_rects:
                cv2.rectangle(img_display, rect[0], rect[1], (0, 255, 0), 2)
            cv2.imshow("Select patches", img_display)

    cv2.namedWindow("Select patches", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select patches", on_mouse)

    print(f"\n🖼️ Processing: {os.path.basename(image_path)}")
    print(f"🖱️ Draw rectangles around each ColorChecker patch ({N_PATCHES} total).")
    print("Press 's' to save, 'q' to skip, or 'r' to reset.")

    while True:
        cv2.imshow("Select patches", img_display)
        key = cv2.waitKey(1) & 0xFF

        # Save
        if key == ord("s") or len(patches) == N_PATCHES:
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
                orig_img.copy(), (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE
            )
            cv2.imshow("Select patches", img_display)

    cv2.destroyAllWindows()
    return np.array(patches) if len(patches) == N_PATCHES else None


# --- MAIN LOOP ---
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))

for img_path in image_paths:
    result = np.array(select_patches(img_path)) / 255.0
    if result is not None:
        json_data[os.path.basename(img_path)] = result.tolist()
        img = colour.cctf_decoding(colour.io.read_image(img_path))
        corrected_img = Normalization_sw(img, result, method="Cheung 2004", degree=3)
        if corrected_img is not None:
            corrected_encoded = colour.cctf_encoding(np.clip(corrected_img, 0, 1))
            corrected_rgb = (corrected_encoded * 255).astype(np.uint8)
            output_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
            imageio.imwrite(output_path, corrected_rgb)
            print(f"💾 Corrected image saved at {output_path}")
        print(f"✅ Saved RGB values for {os.path.basename(img_path)}")
    else:
        print(f"⚠️ Skipped {os.path.basename(img_path)}")


with open(OUTPUT_JSON, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"\n💾 All results saved in {OUTPUT_JSON}")
