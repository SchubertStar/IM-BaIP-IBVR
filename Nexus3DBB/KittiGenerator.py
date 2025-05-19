#!/usr/bin/env python3
# YOLO to KITTI conversion script
#
# This script reads a dataset with YOLOv8 label format (normalized coordinates)
# and an accompanying CSV file, and converts the annotations into KITTI label format.
#
# Configuration:
# - YOLO images directory: path to images (not directly used except for reference)
# - YOLO labels directory: path to YOLO label text files (*.txt)
# - CSV file: path to the annotation CSV containing (index, x_m, y_m, z_m, alpha_rad, rotation_y_rad)
# - Output KITTI labels directory: path to save converted KITTI label files
#
# The script processes each YOLO label file, computes bounding box pixel coordinates,
# and joins with CSV data on the frame index to include 3D location and orientation.
# The output KITTI label has the format:
#   <Class> <truncated> <occluded> <alpha> <bbox_left> <bbox_top> <bbox_right> <bbox_bottom>
#   <dim_h> <dim_w> <dim_l> <loc_x> <loc_y> <loc_z> <rotation_y>
#
# Defaults can be set via command-line arguments; use -h for help.

import os
import re
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLOv8 labels + CSV to KITTI format")
    parser.add_argument("--yolo_images", type=str, default="images", help="Path to YOLO images directory (unused in conversion)")
    parser.add_argument("--yolo_labels", type=str, default="labels", help="Path to YOLO label files directory")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV annotation file")
    parser.add_argument("--kitti_labels", type=str, default="kitti_labels", help="Output directory for KITTI label files")
    return parser.parse_args()

def main():
    args = parse_args()
    yolo_images_dir = args.yolo_images
    yolo_labels_dir = args.yolo_labels
    csv_path = args.csv_file
    output_dir = args.kitti_labels

    # Image dimensions (given)
    img_width = 640.0
    img_height = 320.0

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV into DataFrame, keeping only needed columns
    csv_cols = ['index', 'x_m', 'y_m', 'z_m', 'rotation_y_rad', 'alpha_rad']
    try:
        df = pd.read_csv(csv_path, usecols=csv_cols)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    # Set 'index' as integer index for easy lookup
    df['index'] = df['index'].astype(int)
    df = df.set_index('index')

    # Process each YOLO label file
    for label_fname in sorted(os.listdir(yolo_labels_dir)):
        # Only consider .txt files (YOLO labels)
        if not label_fname.endswith(".txt"):
            continue
        label_path = os.path.join(yolo_labels_dir, label_fname)

        # Extract frame index from filename, assuming 'frame_<index>' pattern
        match = re.search(r'frame_(\d+)', label_fname)
        if not match:
            print(f"Filename {label_fname} does not match expected pattern. Skipping.")
            continue
        index = int(match.group(1))

        # Lookup the corresponding row in CSV data
        if index not in df.index:
            print(f"Index {index} not found in CSV, skipping file {label_fname}.")
            continue
        row = df.loc[index]
        loc_x = row['x_m']
        loc_y = row['y_m']
        loc_z = row['z_m']
        rot_y = row['rotation_y_rad']
        alpha = row['alpha_rad']

        # Read YOLO label lines (class and normalized bbox)
        with open(label_path, 'r') as f:
            yolo_lines = [line.strip() for line in f if line.strip()]

        kitti_lines = []
        # Process each detection (YOLO object) in this file
        for yolo_line in yolo_lines:
            parts = yolo_line.split()
            if len(parts) != 5:
                print(f"Skipping invalid YOLO line in {label_fname}: {yolo_line}")
                continue
            class_id, x_center, y_center, width, height = parts
            # Convert strings to float and de-normalize to pixels
            x_center = float(x_center) * img_width
            y_center = float(y_center) * img_height
            bbox_width = float(width) * img_width
            bbox_height = float(height) * img_height
            # Compute top-left and bottom-right pixel coordinates
            left = x_center - (bbox_width / 2)
            top = y_center - (bbox_height / 2)
            right = x_center + (bbox_width / 2)
            bottom = y_center + (bbox_height / 2)
            # Clip values to image boundaries (optional)
            left = max(0, left)
            top = max(0, top)
            right = min(img_width - 1, right)
            bottom = min(img_height - 1, bottom)
            # Prepare KITTI format fields
            cls = "NexusAMR"
            truncated = 0
            occluded = 0
            # 3D dimensions placeholder (h, w, l)
            dim_h, dim_w, dim_l = 1.0, 1.0, 1.0
            # Format as KITTI line
            # (Note: formatting floats with 6 decimal places for consistency)
            kitti_line = (
                f"{cls} {truncated} {occluded} "
                f"{alpha:.6f} "
                f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
                f"{dim_h:.2f} {dim_w:.2f} {dim_l:.2f} "
                f"{loc_x:.6f} {loc_y:.6f} {loc_z:.6f} "
                f"{rot_y:.6f}"
            )
            kitti_lines.append(kitti_line)

        # Write KITTI lines to corresponding output file (one file per image)
        output_fname = os.path.splitext(label_fname)[0] + ".txt"
        output_path = os.path.join(output_dir, output_fname)
        with open(output_path, 'w') as f_out:
            for line in kitti_lines:
                f_out.write(line + "\n")

    print("Conversion complete. KITTI labels saved in:", output_dir)

if __name__ == "__main__":
    main()
