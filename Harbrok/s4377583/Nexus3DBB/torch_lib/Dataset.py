import cv2
import numpy as np
import os
import random
import glob
import re
import torch

from torchvision import transforms
from torch.utils import data

from library.File import *
from .ClassAverages import ClassAverages

# TODO: clean up where this is
def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1,bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins

def strip_rf_hash(filename):

    match = re.match(r"(.+?)\.rf\.[^\.]+(?:\.\w+)?$", filename)
    if match:
        return match.group(1)
    return os.path.splitext(filename)[0]

class Dataset(data.Dataset):
    def __init__(self, path, bins=2, overlap=0.1):

        self.top_label_path = os.path.join(path, "label_2")
        self.top_img_path = os.path.join(path, "image_2")
        self.top_calib_path = os.path.join(path, "calib")

        # TODO: which camera cal to use, per frame or global one?
        self.proj_matrix = get_P(os.path.abspath(os.path.dirname(os.path.dirname(__file__)) + '/camera_cal/calib_cam_to_cam.txt'))

        img_paths = glob.glob(os.path.join(self.top_img_path, '*.jpg'))
        self.ids = sorted(set([strip_rf_hash(os.path.basename(p)) for p in img_paths]))
        self.num_images = len(self.ids)

        # create angle bins
        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = 2 * np.pi / bins
        for i in range(1,bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2 # center of the bin

        self.overlap = overlap
        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(0,bins):
            self.bin_ranges.append(( (i*self.interval - overlap) % (2*np.pi), \
                                (i*self.interval + self.interval + overlap) % (2*np.pi)) )

        # hold average dimensions
        class_list = KNOWN_CLASSES = ['Car', 'Pedestrian', 'Cyclist', 'NexusAMR']
        self.averages = ClassAverages(class_list)

        self.object_list = self.get_objects(self.ids)

        # pre-fetch all labels
        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id

            self.labels[id][str(line_num)] = label

        # hold one image at a time
        self.curr_id = ""
        self.curr_img = None

        print(f"Top image path: {self.top_img_path}")
        print(f"Number of image files found: {len(img_paths)}")
        print(f"Image IDs: {self.ids}")
        print(f"Number of images: {self.num_images}")


    # should return (Input, Label)
    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            pattern = os.path.join(self.top_img_path, f"{id}.rf.*.jpg")
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No image found matching pattern: {pattern}")
            
            image_path = matches[0]
            self.curr_img = cv2.imread(image_path)
            if self.curr_img is None:
                raise IOError(f"Failed to load image at path: {image_path}")

        label = self.labels[id][str(line_num)]
        # P doesn't matter here
        obj = DetectedObject(self.curr_img, label['Class'], label['Box_2D'], self.proj_matrix, label=label)

        return obj.img, label

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):
        objects = []
        for id in ids:
            pattern = os.path.join(self.top_label_path, f"{id}.rf.*.txt")
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No label file matches {pattern}")
            label_file = matches[0]

            with open(label_file, 'r') as file:
                for line_num, line in enumerate(file):
                    parts = line.strip().split(' ')
                    obj_class = parts[0]
                    if obj_class == "DontCare":
                        continue

                    print(f"Raw class string: '{obj_class}'")

                    # Add your check here, *inside* the loop, where obj_class is defined
                    if obj_class not in self.averages.dimension_map:
                        print(f"Unknown class found: {obj_class}")
                        continue  # or handle as you want

                    dimension = np.array([float(parts[8]), float(parts[9]), float(parts[10])], dtype=np.double)
                    self.averages.add_item(obj_class, dimension)

                    objects.append((id, line_num))
        self.averages.dump_to_file()
        return objects

        print(f"Number of objects found: {len(self.object_list)}")


    def get_label(self, id, line_num):
        pattern = os.path.join(self.top_label_path, f"{id}.rf.*.txt")
        label_path = glob.glob(pattern)
        if not label_path:
            raise FileNotFoundError(f"Label file not found for ID: {id}")
        lines = open(label_path[0]).read().splitlines()
        label = self.format_label(lines[line_num])
        return label

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def format_label(self, line):
        line = line[:-1].split(' ')

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        Alpha = line[3] # what we will be regressing
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]

        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double) # height, width, length
        # modify for the average
        Dimension -= self.averages.get_item(Class)

        Location = [line[11], line[12], line[13]] # x, y, z
        # Location[1] -= Dimension[0] / 2  bring the KITTI center up to the middle of the object

        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)

        # alpha is [-pi..pi], shift it to be [0..2pi]
        angle = Alpha + np.pi

        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]

            Orientation[bin_idx,:] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1

        label = {
                'Class': Class,
                'Box_2D': Box_2D,
                'Dimensions': Dimension,
                'Alpha': Alpha,
                'Orientation': Orientation,
                'Confidence': Confidence
                }

        return label

    def parse_label(self, label_path):
        buf = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line[:-1].split(' ')

                Class = line[0]
                if Class == "DontCare":
                    continue

                for i in range(1, len(line)):
                    line[i] = float(line[i])

                Alpha = line[3] # what we will be regressing
                Ry = line[14]
                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]

                Dimension = [line[8], line[9], line[10]] # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

                buf.append({
                        'Class': Class,
                        'Box_2D': Box_2D,
                        'Dimensions': Dimension,
                        'Location': Location,
                        'Alpha': Alpha,
                        'Ry': Ry
                    })
        return buf

    # will be deprc soon
    def all_objects(self):
        data = {}
        for id in self.ids:
            data[id] = {}
            img_path = self.top_img_path + '%s.jpg'%id
            img = cv2.imread(img_path)
            data[id]['Image'] = img

            # using p per frame
            calib_path = self.top_calib_path + '%s.txt'%id
            proj_matrix = get_calibration_cam_to_image(calib_path)

            # using P_rect from global calib file
            proj_matrix = self.proj_matrix

            data[id]['Calib'] = proj_matrix

            label_path = self.top_label_path + '%s.txt'%id
            labels = self.parse_label(label_path)
            objects = []
            for label in labels:
                box_2d = label['Box_2D']
                detection_class = label['Class']
                objects.append(DetectedObject(img, detection_class, box_2d, proj_matrix, label=label))

            data[id]['Objects'] = objects

        return data


"""
What is *sorta* the input to the neural net. Will hold the cropped image and
the angle to that image, and (optionally) the label for the object. The idea
is to keep this abstract enough so it can be used in combination with YOLO
"""
class DetectedObject:
    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):

        if isinstance(proj_matrix, str): # filename
            proj_matrix = get_P(proj_matrix)
            # proj_matrix = get_calibration_cam_to_image(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):

        # Should this happen? or does normalize take care of it. YOLO doesnt like
        # img=img.astype(np.float) / 255

        # torch transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        process = transforms.Compose ([
            transforms.ToTensor(),
            normalize
        ])

        # crop image
        pt1 = box_2d[0]
        pt2 = box_2d[1]
        crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        # recolor, reformat
        batch = process(crop)

        return batch
