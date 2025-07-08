#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from marvelmind_nav.msg import hedge_pos_ang
from message_filters import Subscriber, ApproximateTimeSynchronizer

image_pub = None
hedge_pub = None

def callback(image_msg, hedge_msg):
	image_pub.publish(image_msg)
	hedge_pub.publish(hedge_msg)
	rospy.loginfo("published synchronized messages")


if __name__ == '__main__':

	rospy.init_node('sync_node')

	rospy.loginfo("Starting sync_node...")
	rospy.loginfo("Waiting for messages from /nexus4/usb_cam/image_raw and /nexus4/hedge_pos_ang")

	image_sub = Subscriber('nexus4/usb_cam/image_raw', Image)
	hedge_sub = Subscriber('nexus4/hedge_pos_ang', hedge_pos_ang)

	image_pub = rospy.Publisher('sync/image_raw', Image, queue_size=10)
	hedge_pub = rospy.Publisher('sync/hedge_pos_ang', hedge_pos_ang, queue_size=10)

	sync = ApproximateTimeSynchronizer(
		[image_sub, hedge_sub],
		queue_size=10,
		slop=0.01,
		allow_headerless=True)

	sync.registerCallback(callback)
	rospy.spin()
