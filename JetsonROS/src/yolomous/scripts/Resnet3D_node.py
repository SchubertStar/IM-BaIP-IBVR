#!/usr/bin/env python3
import rospy
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import time
import gc

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg     import Point
from yolomous.msg import BB2dwIMG 
from yolomous.msg import IBVR3d

from cv_bridge import CvBridge
from torch.autograd import Variable
from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import ClassAverages
from torch_lib import Model
from torchvision.models import resnet18, ResNet18_Weights


class Regression3DNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.latest_img = None
        self.averages = ClassAverages.ClassAverages()
        self.angle_bins = generate_bins(2)
        self.cam_to_img = np.array([
            [487.9530498924, 0.0, 331.4574763671, 0.0],
            [0.0, 485.9382180841, 151.5902976004, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        self.model = self.load_model()

        self.detection_sub = rospy.Subscriber("/2Dnode/imgBB", BB2dwIMG, self.detection_callback, queue_size=1)
        self.info_pub = rospy.Publisher("/3Dnode/IBVR", IBVR3d, queue_size=1)
        self.debug_img_pub = rospy.Publisher("/3Dnode/debug_img", Image, queue_size=1)

    def load_model(self):
        backbone = resnet18(weights=None)
        features = nn.Sequential(*list(backbone.children())[:-2])  # 512x7x7 output
        model = Model.Model(features=features, bins=2).cuda()

        checkpoint_path = '/home/jetson5/IM_BaIP/noetic_ws/src/yolomous/weights/FilterFB_30.pkl'
        print(f"[INFO] Loading custom weights from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)

        try:
            # strict=True ensures exact match, will raise error if keys mismatch
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"[WARNING] Missing keys when loading state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"[WARNING] Unexpected keys in state_dict: {unexpected_keys}")
            print("[INFO] Model weights loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load model weights: {e}")

        model.eval()
        return model

    def detection_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")
        except Exception as e:
            rospy.logwarn(f"Could not convert image: {e}")
            return

        annotated_img = img.copy()

        # We'll collect outputs in parallel arrays:
        out_centers_x, out_centers_y = [], []
        out_diags, out_classes, out_ids = [], [], []

        for i, class_name in enumerate(msg.classes):
            # unpack 2D box and track ID
            x1, y1, x2, y2 = msg.boxes[i*4:(i+1)*4]
            track_id = msg.ids[i] if i < len(msg.ids) else -1

            # skip unrecognized classes
            if not self.averages.recognized_class(class_name):
                continue

            box_2d = [x1, y1, x2, y2]
            try:
                detected_obj = DetectedObject(img, class_name, box_2d, self.cam_to_img)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("Invalid box dims")
            except Exception as e:
                rospy.logwarn(f"Skipping invalid detection {box_2d}: {e}")
                continue

            # run your 3D regression
            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0] = detected_obj.img
            with torch.no_grad():
                orient, conf, dim = self.model(input_tensor)
            orient_np = orient.cpu().numpy()[0]
            conf_np   = conf.cpu().numpy()[0]
            dim_np    = dim.cpu().numpy()[0] + self.averages.get_item(class_name)
            torch.cuda.empty_cache()

            argmax = np.argmax(conf_np)
            alpha = np.arctan2(*orient_np[argmax][::-1]) + self.angle_bins[argmax] - np.pi

            cx, cy, theta_x, theta_y = calc_theta_ray(img, box_2d, self.cam_to_img)
            try:
                best_loc, _ = calc_location(dim_np, self.cam_to_img, box_2d, alpha, theta_x)
            except Exception as e:
                rospy.logwarn(f"calc_location() failed for {box_2d}: {e}")
                continue

            yaw_global = alpha + theta_x

            # draw the 2D box
            plot_2d_box(annotated_img, box_2d)

            # project and compute diagonal length
            corners3d = create_corners(dim_np, best_loc, rotation_matrix(yaw_global))
            corners3d_homo = np.hstack((corners3d, np.ones((8,1))))
            projected = (self.cam_to_img @ corners3d_homo.T).T
            pts_2d = projected[:, :2] / projected[:, 2:3]
            diag_length, _ = longest_diagonal(pts_2d)

            # overlay a sphere if you wish
            if diag_length > 0:
                plot_3d_sphere(annotated_img, cx, cy, int(diag_length/2))

            # --- NEW: putText for ID ---
            label = f"ID:{track_id}"
            cv2.putText(
                annotated_img, label,
                (int(cx) - 10, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,255), 2
            )

            # collect for message
            out_centers_x.append(float(cx))
            out_centers_y.append(float(cy))
            out_diags.append(float(diag_length))
            out_classes.append(class_name)
            out_ids.append(int(track_id))

        # publish IBVR3d
        msg_out = IBVR3d()
        msg_out.header = msg.header
        msg_out.center_x        = out_centers_x
        msg_out.center_y        = out_centers_y
        msg_out.diagonal_length = out_diags
        msg_out.class_name      = out_classes
        msg_out.ids        = out_ids       # NEW field
        self.info_pub.publish(msg_out)

        # publish debug image
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
            overlay_msg.header = msg.header
            self.debug_img_pub.publish(overlay_msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish overlay image: {e}")

if __name__ == "__main__":
    rospy.init_node("regression_3d_node") 
    node = Regression3DNode()
    rospy.spin()
