import rospy
from ibvr_control.msg import IBVRout
from marvelmind_nav.msg import hedge_pos_ang
from message_filters import Subscriber, ApproximateTimeSynchronizer

ibvr_pub = None
hedge_pub = None

def callback(ibvr_msg, hedge_msg):
    ibvr_pub.publish(ibvr_msg)
    hedge_pub.publish(hedge_msg)
    rospy.loginfo("Published synchronized IBVR and hedge messages")

if __name__ == "__main__":
    rospy.init_node('sync_ibvr_hedge')
    rospy.loginfo("Waiting for /nexus4/ibvr_output and /nexus4/hedge_pos_ang")

    # Subscribers (message_filters)
    ibvr_sub  = Subscriber('/nexus4/ibvr_output', IBVRout)
    hedge_sub = Subscriber('/nexus4/hedge_pos_ang', hedge_pos_ang)

    # Publishers for the synchronized streams
    ibvr_pub  = rospy.Publisher('sync/ibvr_output', IBVRout, queue_size=1)
    hedge_pub = rospy.Publisher('sync/hedge_pos_ang', hedge_pos_ang, queue_size=1)

    # ApproximateTimeSynchronizer setup
    sync = ApproximateTimeSynchronizer(
        [ibvr_sub, hedge_sub],
        queue_size=1,
        slop=0.01,
        allow_headerless=True
    )
    sync.registerCallback(callback)

    rospy.spin()