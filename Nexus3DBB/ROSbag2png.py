import rosbag
import cv2
from cv_bridge import CvBridge
import os

bag_file = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/RosBagsBaIP/bigbag1.bag"
output_dir = "images/bigbag1"
image_topic = "/sync/image_raw"

os.makedirs(output_dir, exist_ok=True)
bridge = CvBridge()

with rosbag.Bag(bag_file, "r") as bag:
    for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=[image_topic])):
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            filename = os.path.join(output_dir, f"frame_{idx}.png")
            cv2.imwrite(filename, cv_image)
        except Exception as e:
            print(f"Failed to convert image {idx}: {e}")
