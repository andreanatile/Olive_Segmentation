import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Paths
json_path = "/home/girobat/Olive/Normalization/grid_results.json"
output_dir = "/home/girobat/Olive/Normalization/plots"


# Load data
with open(json_path, "r") as f:
    data = json.load(f)

# Loop through all methods except "NoCorrection"
for method in [m for m in data.keys() if m != "NoCorrection"]:
    categories = list(data[method].keys())
    values = [np.mean(data[method][deg]["swatches"]) for deg in categories]

    # Add "NoCorrection" reference
    categories.append("NoCorrection")
    values.append(np.mean(data["NoCorrection"]))

    # Create a new figure
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, values, color="skyblue", edgecolor="black")

    # Titles and labels
    plt.xlabel("Degree")
    plt.ylabel("ΔE (mean)")
    plt.title(f"Delta E between reference and corrected swatches - {method}")
    plt.ylim(0, max(values) * 1.2)

    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max(values) * 0.02),
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Save figure to output folder
    filename = os.path.join(output_dir, f"{method}.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory

    print(f"Saved plot for {method}: {filename}")

print("✅ All plots saved successfully!")
