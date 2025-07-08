#!/usr/bin/env python3
import rospy
import torch
import torch.nn as nn
import numpy as np
import cv2

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from yolomous.msg import BB2dwIMG, IBVR3d

from cv_bridge import CvBridge
from torch.autograd import Variable
from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import ClassAverages
from torch_lib import Model
from torchvision.models import vgg19_bn, VGG19_BN_Weights

class Regression3DNode:
    def __init__(self):
        rospy.loginfo("[Regression3DNode] Initializing VGG19_BN node…")
        self.bridge = CvBridge()
        self.averages = ClassAverages.ClassAverages()
        self.angle_bins = generate_bins(2)
        self.cam_to_img = np.array([
            [487.9530498924, 0.0, 331.4574763671, 0.0],
            [0.0, 485.9382180841, 151.5902976004, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        rospy.loginfo("[Regression3DNode] Loading VGG19_BN model…")
        self.model = self.load_model()

        self.detection_sub = rospy.Subscriber(
            "/2Dnode/imgBB", BB2dwIMG,
            self.detection_callback, queue_size=1
        )
        self.info_pub = rospy.Publisher(
            "/3Dnode/IBVR", IBVR3d, queue_size=1
        )
        # self.debug_img_pub = rospy.Publisher("/3Dnode/debug_img", Image, queue_size=1)

        rospy.loginfo("[Regression3DNode] Subscribed and ready.")

    def load_model(self):
        weights = VGG19_BN_Weights.DEFAULT
        base = vgg19_bn(weights=weights).features.cuda()
        model = Model.Model(features=base, bins=2).cuda()
        checkpoint = torch.load(
            '/home/jetson5/IM_BaIP/noetic_ws/src/yolomous/weights/FilterFB_30.pkl'
        )
        rospy.loginfo(f"[Regression3DNode] Loading weights from FilterFB_30.pkl")
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        if missing:
            rospy.logwarn(f"[Regression3DNode] Missing keys: {missing}")
        if unexpected:
            rospy.logwarn(f"[Regression3DNode] Unexpected keys: {unexpected}")
        model.eval()
        rospy.loginfo("[Regression3DNode] Model ready in eval() mode.")
        return model

    def detection_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")
        except Exception as e:
            rospy.logwarn(f"Could not convert image: {e}")
            return

        annotated = img.copy()
        out_centers_x, out_centers_y = [], []
        out_diags, out_classes = [], []

        for i, class_name in enumerate(msg.classes):
            x1, y1, x2, y2 = msg.boxes[i*4:(i+1)*4]

            if not self.averages.recognized_class(class_name):
                continue

            try:
                dobj = DetectedObject(img, class_name, [x1,y1,x2,y2], self.cam_to_img)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("invalid box dims")
            except Exception as e:
                rospy.logwarn(f"Skipping invalid detection {box_2d}: {e}")
                continue

            # prepare input and forward
            inp = torch.zeros([1,3,224,224]).cuda()
            inp[0] = dobj.img
            with torch.no_grad():
                orient, conf, dim = self.model(inp)
            orient_np = orient.cpu().numpy()[0]
            conf_np   = conf.cpu().numpy()[0]
            dim_np    = dim.cpu().numpy()[0] + self.averages.get_item(class_name)

            arg = np.argmax(conf_np)
            alpha = np.arctan2(orient_np[arg][1], orient_np[arg][0])
            alpha += self.angle_bins[arg] - np.pi

            cx, cy, tx, ty = calc_theta_ray(img, [x1,y1,x2,y2], self.cam_to_img)

            try:
                best_loc, _ = calc_location(dim_np, self.cam_to_img, [x1,y1,x2,y2], alpha, tx)
            except Exception as e:
                rospy.logwarn(f"calc_location() failed for {box_2d}: {e}")
                continue

            yaw = alpha + tx

            plot_2d_box(annotated, [x1,y1,x2,y2])
            corners = create_corners(dim_np, best_loc, rotation_matrix(yaw))
            homo    = np.hstack((corners, np.ones((8,1))))
            proj    = (self.cam_to_img @ homo.T).T
            pts2d   = proj[:, :2] / proj[:, 2:3]
            diag, _ = longest_diagonal(pts2d)

            if diag > 0:
                plot_3d_sphere(annotated, cx, cy, int(diag/2))

            out_centers_x.append(float(cx))
            out_centers_y.append(float(cy))
            out_diags.append(float(diag))
            out_classes.append(class_name)

        # publish IBVR3d
        msg_out = IBVR3d()
        msg_out.header          = msg.header
        msg_out.center_x        = out_centers_x
        msg_out.center_y        = out_centers_y
        msg_out.diagonal_length = out_diags
        msg_out.class_name      = out_classes
        self.info_pub.publish(msg_out)
        rospy.loginfo(f"[Regression3DNode] Published IBVR3d with {len(out_classes)} entries.")

        # optionally: debug images
        # try:
        #     overlay = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        #     overlay.header = msg.header
        #     self.debug_img_pub.publish(overlay)
        #     rospy.loginfo("[Regression3DNode] Published debug image.")
        # except Exception as e:
        #     rospy.logwarn(f"[Regression3DNode] Debug img pub failed: {e}")

if __name__ == "__main__":
    rospy.init_node("regression_3d_node")
    Regression3DNode()
    rospy.spin()
