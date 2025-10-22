#!/usr/bin/env python3
import os
import cv2
import json
import shutil
import numpy as np
import colour
from colour_checker_detection import detect_colour_checkers_segmentation
from Normalization_script import Normalization
import imageio.v2 as imageio

# ---------------- CONFIG ----------------
input_folder = "/home/girobat/Olive/Corrected/foto olivo 28.08.24/not_detected"
corrected_folder = "/home/girobat/Olive/Corrected/foto olivo 28.08.24"


# ---------------- GLOBALS ----------------
ref_point = []
cropping = False
temp_image = None
display_image_global = None


# ---------------- FUNCTIONS ----------------
def click_and_crop(event, x, y, flags, param):
    """Mouse callback to draw rectangle live while dragging"""
    global ref_point, cropping, temp_image, display_image_global

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point.clear()
        ref_point.append((x, y))
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        temp_image = display_image_global.copy()
        cv2.rectangle(temp_image, ref_point[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        temp_image = display_image_global.copy()
        cv2.rectangle(temp_image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", temp_image)


# ---------------- MAIN PROCESS ----------------
all_detected = {}

for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_folder, filename)
    print(f"\n🖼️ Processing {filename} ...")

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read {filename}. Skipping.")
        continue
    # Convert to RGB and decode to linear for color correction
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_linear = colour.cctf_decoding(img_rgb / 255.0)
    clone = img.copy()

    # Resize for display
    max_display = 800
    scale = max_display / max(img.shape[:2])
    display_image = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    display_image_global = display_image.copy()
    temp_image = display_image.copy()

    cv2.imshow("image", display_image)
    cv2.setMouseCallback("image", click_and_crop)

    print("→ Select ColorChecker region, press 'c' to confirm or 'r' to reset.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break
        elif key == ord("r"):
            temp_image = display_image.copy()
            cv2.imshow("image", temp_image)

    if len(ref_point) == 2:
        x1, y1 = int(ref_point[0][0] / scale), int(ref_point[0][1] / scale)
        x2, y2 = int(ref_point[1][0] / scale), int(ref_point[1][1] / scale)
        crop = clone[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]

        if crop.size == 0:
            print(f"⚠️ Invalid crop for {filename}, skipping.")
            continue

        # Convert BGR → RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Detect ColourChecker
        detected_result = detect_colour_checkers_segmentation(
            crop_rgb, additional_data=True
        )

        if detected_result:
            print(f"✅ ColorChecker detected in {filename}")
            all_detected[filename] = detected_result[0]

            # Apply color correction
            corrected_img = Normalization(
                img, detected_result, method="Cheung 2004", degree=3
            )
            if corrected_img is not None:
                # Gamma encode and clip
                corrected_encoded = colour.cctf_encoding(np.clip(corrected_img, 0, 1))
                corrected_rgb = (corrected_encoded * 255).astype("uint8")
                corrected_path = os.path.join(corrected_folder, filename)
                imageio.imwrite(corrected_path, corrected_rgb)  # Save as true RGB
                print(f"💾 Corrected RGB image saved: {corrected_path}")
                # Move original image to corrected folder (for reference)
                archived_path = os.path.join(corrected_folder, f"orig_{filename}")
                shutil.move(img_path, archived_path)
                print(f"📦 Original image moved to: {archived_path}")

        else:
            print(f"❌ Detection failed for {filename}")

cv2.destroyAllWindows()
