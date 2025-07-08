import cv2
import numpy as np
from enum import Enum
import itertools

from .File import *
from .Math import *

class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)

def constraint_to_color(constraint_idx):
    return {
        0 : cv_colors.PURPLE.value, #left
        1 : cv_colors.ORANGE.value, #top
        2 : cv_colors.MINT.value, #right
        3 : cv_colors.YELLOW.value #bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    x1, y1, x2, y2 = box_2d

    pt1 = (int(round(x1)), int(round(y1)))
    pt2 = (int(round(x1)), int(round(y2)))
    pt3 = (int(round(x2)), int(round(y2)))
    pt4 = (int(round(x2)), int(round(y1)))

    return pt1, pt2, pt3, pt4
# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, cam_to_img, calib_file=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)
        R0_rect = get_R0(calib_file)
        Tr_velo_to_cam = get_tr_to_velo(calib_file)

    point = np.array(pt)
    point = np.append(point, 1)

    point = np.dot(cam_to_img, point)
    # point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2]/point[2]
    point = point.astype(np.int16)

    return point


# take in 3d points and plot them on image as red circles
def plot_3d_pts(img, pts, center, calib_file=None, cam_to_img=None, relative=False, constraint_idx=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)

    for pt in pts:
        if relative:
            pt = [i + center[j] for j,i in enumerate(pt)] # more pythonic

        point = project_3d_pt(pt, cam_to_img)

        color = cv_colors.RED.value

        if constraint_idx is not None:
            color = constraint_to_color(constraint_idx)

        cv2.circle(img, (point[0], point[1]), 3, color, thickness=-1)



def plot_3d_box(img, cam_to_img, ry, dimension, center):

    # plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)

    R = rotation_matrix(ry)

    corners = create_corners(dimension, location=center, R=R)
    # print('corners: %s'%corners)
    # to see the corners on image as red circles
    # plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False)

    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        # print('corners point in image: %s'%point)
        box_3d.append(point)

    #TODO put into loop
    # cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), cv_colors.GREEN.value, 1)
    # cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.GREEN.value, 1)
    # cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), cv_colors.GREEN.value, 1)
    # cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.GREEN.value, 1)

    # cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.GREEN.value, 1)
    # cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.GREEN.value, 1)
    # cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.GREEN.value, 1)
    # cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.GREEN.value, 1)

    # for i in range(0,7,2):
    #     cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), cv_colors.GREEN.value, 1)

    # front_mark = [(box_3d[i][0], box_3d[i][1]) for i in range(4)]

    # cv2.line(img, front_mark[0], front_mark[3], cv_colors.BLUE.value, 1)
    # cv2.line(img, front_mark[1], front_mark[2], cv_colors.BLUE.value, 1)

    return box_3d

def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)



def plot_3d_sphere(img,center_x,center_y,radius):
    # center_x = map(int, center_x)  # Ensures (int, int)
    # center_y = map(int, center_y)  # Ensures (int, int)
    # Get image dimensions
    height, width = img.shape[:2]

    # Create a transparent layer
    sphere_layer = np.zeros((height, width, 4), dtype=np.uint8)  # 4 channels (RGBA)

    # Define sphere parameters
    center = (center_x, center_y)  # Center of the sphere
    center = tuple(map(int, center))
    # radius = min(width, height) // 4    # Adjust radius to fit image
    # radius = 100
    color = (0, 0, 255, 125)  # Blue color with transparency
    # color = cv_colors.BLUE.value

    # Draw longitude lines (vertical curved lines)
    for angle in range(0, 181, 20):  # Spaced at 20 degrees
        cv2.ellipse(sphere_layer, center, (radius, radius//2), 0, 0, 360, color, 2)

    # Draw latitude lines (horizontal curved lines)
    for angle in range(-90, 91, 20):  # Spaced at 20 degrees
        cv2.ellipse(sphere_layer, center, (radius, radius), angle, 0, 360, color, 2)

     # Draw **one extra vertical plane (90° rotated longitude)**
    cv2.ellipse(sphere_layer, center, (radius // 2, radius), 180, 0, 360, color, 2)

    # Extract the RGB channels and Alpha channel
    sphere_rgb = sphere_layer[:, :, :3]
    alpha_mask = sphere_layer[:, :, 3] / 255.0  # Normalize alpha to [0, 1]

    # Blend the sphere onto the original image
    for c in range(3):  # For each color channel (B, G, R)
        img[:, :, c] = (1 - alpha_mask) * img[:, :, c] + alpha_mask * sphere_rgb[:, :, c]
