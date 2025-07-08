import os
import glob
import numpy as np

# === Configuration ===
image_folder = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/cdatasets/IMfinalbig1.v1i.yolov11/image_2"   # CHANGE THIS
output_folder = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/cdatasets/IMfinalbig1.v1i.yolov11/calib"  # Save .txt files here
calib_file = "camera_calib.npz"        # Your calibration file from OpenCV

# === Load calibration data ===
data = np.load(calib_file)
K = data['K']  # Intrinsic matrix (3x3)

# Create projection matrix P = [K | 0]
P = np.hstack((K, np.zeros((3, 1))))  # (3x4)

# === Format KITTI-style string ===
P_str = ' '.join(f"{v:.12e}" for v in P.flatten())

# === Make output folder if it doesn't exist ===
os.makedirs(output_folder, exist_ok=True)

# === Find all .png images in the folder ===
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

# === Generate calib.txt for each image ===
for img_path in image_paths:
    img_filename = os.path.basename(img_path)
    name_no_ext = os.path.splitext(img_filename)[0]

    calib_txt_path = os.path.join(output_folder, f"{name_no_ext}.txt")

    with open(calib_txt_path, 'w') as f:
        f.write(f"P0: {P_str}\n")

print(f"Found {len(image_paths)} images.")
