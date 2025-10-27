#!/usr/bin/env python3
import argparse
import cv2
import imageio
import numpy as np
import os
import glob
import json
import colour
from Normalizer import Normalization_sw


class ManualSwatchesNormalizer:
    """Interactively select ColorChecker patches and apply manual normalization."""

    # --- CONSTANTS ---
    DEFAULT_N_PATCHES = 24
    DEFAULT_DISPLAY_SCALE = 0.5

    D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS[
        "ColorChecker24 - After November 2014"
    ]

    REFERENCE_SWATCHES = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
        "sRGB",
        REFERENCE_COLOUR_CHECKER.illuminant,
    )

    def __init__(
        self,
        input_dir,
        output_dir,
        output_json,
        method="Cheung 2004",
        degree=3,
        n_patches=DEFAULT_N_PATCHES,
        display_scale=DEFAULT_DISPLAY_SCALE,
        dry_run=False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_json = output_json
        self.method = method
        self.degree = degree
        self.n_patches = n_patches
        self.display_scale = display_scale
        self.dry_run = dry_run
        self.json_data = {}

    def select_patches(self, image_path):
        """Open an image, let the user select color patches manually."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Cannot read {image_path}")
            return None

        orig_img = img.copy()
        img_display = cv2.resize(
            orig_img.copy(), (0, 0), fx=self.display_scale, fy=self.display_scale
        )
        completed_rects = []
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
                x1, y1 = int(points[0][0] / self.display_scale), int(
                    points[0][1] / self.display_scale
                )
                x2, y2 = int(points[1][0] / self.display_scale), int(
                    points[1][1] / self.display_scale
                )
                roi = orig_img[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]
                mean_color = roi.mean(axis=(0, 1))
                patches.append(mean_color[::-1].tolist())  # Convert BGR → RGB
                print(f"Patch {len(patches)} avg RGB: {patches[-1]}")
                completed_rects.append((points[0], points[1]))
                img_display[:] = cv2.resize(
                    orig_img.copy(),
                    (0, 0),
                    fx=self.display_scale,
                    fy=self.display_scale,
                )
                for rect in completed_rects:
                    cv2.rectangle(img_display, rect[0], rect[1], (0, 255, 0), 2)
                cv2.imshow("Select patches", img_display)

        cv2.namedWindow("Select patches", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select patches", on_mouse)

        print(f"\n🖼️ Processing: {os.path.basename(image_path)}")
        print(
            f"🖱️ Draw rectangles around each ColorChecker patch ({self.n_patches} total)."
        )
        print("Press 's' to save, 'q' to skip, or 'r' to reset.")

        while True:
            cv2.imshow("Select patches", img_display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("+"), ord("=")):
                pass  # zoom omitted for simplicity
            elif key == ord("s") or len(patches) == self.n_patches:
                break
            elif key == ord("q"):
                patches = []
                break
            elif key == ord("r"):
                print("🔄 Resetting selections...")
                patches.clear()
                completed_rects.clear()
                img_display[:] = cv2.resize(
                    orig_img.copy(),
                    (0, 0),
                    fx=self.display_scale,
                    fy=self.display_scale,
                )
                cv2.imshow("Select patches", img_display)

        cv2.destroyAllWindows()
        return np.array(patches) if len(patches) == self.n_patches else None

    def process_images(self):
        """Process all JPGs in input_dir by manual patch selection and normalization."""
        image_paths = sorted(glob.glob(os.path.join(self.input_dir, "*.jpg")))
        if not image_paths:
            print(f"No images found in {self.input_dir}")
            return

        local_json = {}

        for img_path in image_paths:
            result = self.select_patches(img_path)
            if result is None:
                print(f"⚠️ Skipped {os.path.basename(img_path)}")
                continue

            result = np.array(result) / 255.0
            local_json[os.path.basename(img_path)] = result.tolist()

            if self.dry_run:
                print(
                    f"Dry-run: would normalize {os.path.basename(img_path)} with {self.method}, degree={self.degree}"
                )
                continue

            img = colour.cctf_decoding(colour.io.read_image(img_path))
            corrected_img = Normalization_sw(
                img, result, method=self.method, degree=self.degree
            )
            if corrected_img is not None:
                corrected_encoded = colour.cctf_encoding(np.clip(corrected_img, 0, 1))
                corrected_rgb = (corrected_encoded * 255).astype(np.uint8)
                output_path = os.path.join(self.output_dir, os.path.basename(img_path))
                os.makedirs(self.output_dir, exist_ok=True)
                imageio.imwrite(output_path, corrected_rgb)
                print(f"💾 Corrected image saved at {output_path}")
            print(f"✅ Saved RGB values for {os.path.basename(img_path)}")

        with open(self.output_json, "w") as f:
            json.dump(local_json, f, indent=4)

        print(f"\n💾 All results saved in {self.output_json}")

    def run(
        self,
        input_dir,
        output_dir,
        output_json,
        method,
        degree,
        n_patches,
        display_scale,
        dry_run,
    ):
        tool = ManualSwatchesNormalizer(
            input_dir=input_dir,
            output_dir=output_dir,
            output_json=output_json,
            method=method,
            degree=degree,
            n_patches=n_patches,
            display_scale=display_scale,
            dry_run=dry_run,
        )
        tool.process_images()
