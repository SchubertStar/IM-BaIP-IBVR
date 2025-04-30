import os

# Hardcoded paths based on your folder structure
LABEL_DIR = "training/label_2"      # KITTI label files
OUTPUT_DIR = "training/label_2YOLO" # YOLO label files

# Hardcoded image size (KITTI default: 1242x375)
IMG_WIDTH = 1242
IMG_HEIGHT = 375

# Class mapping
class_mapping = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Motorcyclist": 6,
    "Bus": 7,
    "Tram": 8,
    "Misc": 9,
    "DontCare": 10
}

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each KITTI label file in LABEL_DIR
for filename in os.listdir(LABEL_DIR):
    if not filename.endswith('.txt'):
        continue

    with open(os.path.join(LABEL_DIR, filename), 'r') as infile:
        lines = infile.readlines()

    with open(os.path.join(OUTPUT_DIR, filename), 'w') as outfile:
        for line in lines:
            parts = line.strip().split()
            class_name = parts[0]

            if class_name not in class_mapping:
                continue

            class_id = class_mapping[class_name]
            xmin, ymin, xmax, ymax = map(float, parts[4:8])

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / IMG_WIDTH
            y_center = ((ymin + ymax) / 2) / IMG_HEIGHT
            width = (xmax - xmin) / IMG_WIDTH
            height = (ymax - ymin) / IMG_HEIGHT

            outfile.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"All KITTI labels converted and saved to '{OUTPUT_DIR}'")
