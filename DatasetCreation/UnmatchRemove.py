import csv
import math
import os
import numpy as np

# Directories for image/label filtering (must exist)
images_dir = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/cdatasets/IMfinalbig1.v1i.filter/training/images_2"
labels_dir = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/cdatasets/IMfinalbig1.v1i.filter/training/labels_2"

# File extensions (include leading dot)
image_ext  = ".jpg"
label_ext  = ".txt"

# Build sets of base filenames in each directory
img_files = [f for f in os.listdir(images_dir) if f.endswith(image_ext)]
label_files = [f for f in os.listdir(labels_dir) if f.endswith(label_ext)]

img_bases = {os.path.splitext(f)[0] for f in img_files}
label_bases = {os.path.splitext(f)[0] for f in label_files}

# Identify mismatches
imgs_to_delete = img_bases - label_bases
labels_to_delete = label_bases - img_bases

# Delete unmatched images
for base in imgs_to_delete:
    img_path = os.path.join(images_dir, base + image_ext)
    try:
        os.remove(img_path)
        print(f"Deleted image without label: {img_path}")
    except FileNotFoundError:
        pass

# Delete unmatched labels
for base in labels_to_delete:
    lbl_path = os.path.join(labels_dir, base + label_ext)
    try:
        os.remove(lbl_path)
        print(f"Deleted label without image: {lbl_path}")
    except FileNotFoundError:
        pass

print(f"Image-label sync complete: removed {len(imgs_to_delete)} images and {len(labels_to_delete)} labels.")