#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from yolomous.msg import BB2dwIMG #BB2d
from ultralytics import YOLO

import rospy
import cv2
import time
import numpy as np
import math
from collections import deque
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from yolomous.msg import BB2dwIMG
from cv_bridge import CvBridge
from ultralytics import YOLO


def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0.0

class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.next_id = 0
        self.tracks = {}  # id -> {'box': [...], 'class': str, 'lost': int}
        self.iou_thresh = iou_threshold
        self.max_lost = max_lost

    def update(self, det_boxes, det_classes):
        updated_ids = []
        new_tracks = {}
        used = set()

        for box, cls in zip(det_boxes, det_classes):
            best_id, best_iou = None, self.iou_thresh
            for tid, trk in self.tracks.items():
                if tid in used or trk['class'] != cls:
                    continue
                ov = iou(box, trk['box'])
                if ov >= best_iou:
                    best_iou, best_id = ov, tid

            if best_id is not None:
                new_tracks[best_id] = {'box': box, 'class': cls, 'lost': 0}
                used.add(best_id)
                updated_ids.append(best_id)
            else:
                new_tracks[self.next_id] = {'box': box, 'class': cls, 'lost': 0}
                updated_ids.append(self.next_id)
                self.next_id += 1

        for tid, trk in self.tracks.items():
            if tid not in used and tid not in new_tracks:
                trk['lost'] += 1
                if trk['lost'] <= self.max_lost:
                    new_tracks[tid] = trk

        self.tracks = new_tracks
        return updated_ids


class YOLO2DNode:
    def __init__(self):

        self.bridge = CvBridge()
        self.tracker = IoUTracker(iou_threshold=0.3, max_lost=5)

        model_path = "/home/jetson5/IM_BaIP/noetic_ws/src/yolomous/weights/bigyolo11s.pt"
        print(f"[INFO] Loading YOLO model from: {model_path}")

        try:
            self.model = YOLO(model_path)

            # Print verification details
            if hasattr(self.model, 'names') and self.model.names:
                print(f"[INFO] Model contains {len(self.model.names)} classes: {self.model.names}")
            else:
                print("[WARNING] Model loaded but contains no class names.")

            if hasattr(self.model, 'task'):
                print(f"[INFO] Model task type: {self.model.task}")
            else:
                print("[WARNING] Model task type is unknown.")

            print("[INFO] YOLO model weights loaded successfully.")

        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model from '{model_path}': {e}")
            raise

        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        self.pub = rospy.Publisher("/2Dnode/imgBB", BB2dwIMG, queue_size=1)

        self.last_processed_time = 0.0
        self.min_interval = 0.0001

    def image_callback(self, msg):

        current_time = time.time()
        if current_time - self.last_processed_time < self.min_interval:
            return  

        self.last_processed_time = current_time

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(frame, verbose=False)[0]

        det_boxes, det_classes, det_scores = [], [], []
        for det in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = det
            det_boxes.append([float(x1), float(y1), float(x2), float(y2)])
            det_classes.append(self.model.names[int(cls_id)])
            det_scores.append(float(conf))

        ids = self.tracker.update(det_boxes, det_classes)

        # build and publish message
        msg_out = BB2dwIMG()
        msg_out.header = msg.header
        msg_out.classes = det_classes
        msg_out.scores = det_scores
        msg_out.ids = ids
        msg_out.boxes = [coord for box in det_boxes for coord in box]
        msg_out.image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        msg_out.image.header = msg.header

        self.pub.publish(msg_out)

if __name__ == "__main__":
    rospy.init_node("yolo_2d_node")
    node = YOLO2DNode()
    rospy.spin()
