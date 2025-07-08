#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np

# ====== Hardcoded Configuration ======
YOLO_LABELS_DIR = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/aadatasets/processedRoboFlow/finalbig.v1i.yolov11/labelsall"            # Path to directory of YOLO .txt label files
CSV_FILE         = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/rawROSbagdata/hedge_pos_ang/finalbig1noplateu.csv"  # Path to CSV file with transforms
OUTPUT_DIR       = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/kitti_labels/finalbig.v1i.noplateu"    # Where to save KITTI .txt files
IMG_WIDTH        = 640.0
IMG_HEIGHT       = 320.0

# KITTI 3D box dimensions (height, width, length) in meters
DIM_H = 0.36
DIM_W = 0.35
DIM_L = 0.40
# ======================================

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CSV into DataFrame
    cols = ['index', 'cam_x', 'cam_y', 'cam_z', 'cam_ang',
            'rel_X', 'rel_Y', 'rel_Z', 'rotation_y', 'alpha_rad']

    df = pd.read_csv(CSV_FILE, usecols=cols)
    df['index'] = df['index'].astype(int)
    df.set_index('index', inplace=True)

    # Iterate YOLO label files
    for fname in sorted(os.listdir(YOLO_LABELS_DIR)):
        if not fname.endswith('.txt'):
            continue
        # Extract frame index (expecting 'frame_123.txt')
        match = re.search(r'frame_(\d+)', fname)
        if not match:
            print(f"Skipping {fname}: filename pattern mismatch")
            continue
        idx = int(match.group(1))
        if idx not in df.index:
            print(f"Index {idx} not in CSV, skipping")
            continue
        row = df.loc[idx]

        # Precomputed KITTI fields from CSV
        loc_x = float(row['rel_X'])
        loc_y = float(row['rel_Y'])
        loc_z = float(row['rel_Z'])
        rotation_y = float(row['rotation_y'])
        alpha = float(row['alpha_rad'])

        # Read YOLO label lines
        label_path = os.path.join(YOLO_LABELS_DIR, fname)
        with open(label_path, 'r') as f:
            yolo_lines = [line.strip() for line in f if line.strip()]

        kitti_lines = []
        for line in yolo_lines:
            parts = line.split()
            if len(parts) != 5:
                print(f"Invalid YOLO line in {fname}: {line}")
                continue
            _, xc_n, yc_n, w_n, h_n = parts
            # Denormalize to pixel coords
            xc = float(xc_n) * IMG_WIDTH
            yc = float(yc_n) * IMG_HEIGHT
            bw = float(w_n)  * IMG_WIDTH
            bh = float(h_n)  * IMG_HEIGHT
            # Compute pixel bbox
            left   = max(0.0, xc - bw/2)
            top    = max(0.0, yc - bh/2)
            right  = min(IMG_WIDTH-1, xc + bw/2)
            bottom = min(IMG_HEIGHT-1, yc + bh/2)

            # Assemble KITTI label line
            cls_name  = 'nexusamr'
            truncated = 0
            occluded  = 0
            line_k = (
                f"{cls_name} {truncated} {occluded} "
                f"{alpha:.6f} "
                f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
                f"{DIM_H:.2f} {DIM_W:.2f} {DIM_L:.2f} "
                f"{loc_x:.6f} {loc_y:.6f} {loc_z:.6f} "
                f"{rotation_y:.6f}"
            )
            kitti_lines.append(line_k)

        # Write to output
        out_fname = fname
        out_path  = os.path.join(OUTPUT_DIR, out_fname)
        with open(out_path, 'w') as fw:
            fw.write("\n".join(kitti_lines))

    print(f"Conversion complete. KITTI labels saved in '{OUTPUT_DIR}'")

if __name__ == '__main__':
    main()
