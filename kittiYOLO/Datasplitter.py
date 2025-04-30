import os
import random
import shutil

# Define paths to your images and labels
image_dir = 'kittiYOLOData/images/master'  # Change this to your image directory path
label_dir = 'kittiYOLOData/labels/master'  # Change this to your label directory path

train_dir = 'kittiYOLOData/images/train'        # Training images
val_dir = 'kittiYOLOData/images/val'            # Validation images
train_label_dir = 'kittiYOLOData/labels/train'  # Training labels
val_label_dir = 'kittiYOLOData/labels/val'      # Validation labels

# Create train and val directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get list of all image and label files (assuming .png images)
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
label_files = [f.replace('.png', '.txt') for f in image_files]  # assuming labels are .txt

# Shuffle the file list
random.seed(42)  # For reproducibility
combined = list(zip(image_files, label_files))
random.shuffle(combined)
image_files, label_files = zip(*combined)

# Split the data (80% train, 20% val)
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

# Split the image and label files
train_images = image_files[:split_index]
val_images = image_files[split_index:]
train_labels = label_files[:split_index]
val_labels = label_files[split_index:]

# Move images and labels to train/val directories
def move_files(files, source_dir, dest_dir):
    for file in files:
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

move_files(train_images, image_dir, train_dir)
move_files(val_images, image_dir, val_dir)
move_files(train_labels, label_dir, train_label_dir)
move_files(val_labels, label_dir, val_label_dir)

print(f"Data split complete, Training images: {len(train_images)}, Validation images: {len(val_images)}")