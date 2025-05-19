import rosbag
import csv
import math

# Inputs
bag_file = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/RosBagsBaIP/bigbag2.bag"
output_file = "positiondata/bigbag2.csv"
position_topic = "/sync/hedge_pos_ang"

# Fixed object position and orientation (in global frame)
object_global_yaw_deg = 0.0  # Change as needed
object_x = 0.0  # Replace with actual known X
object_z = 0.0  # Replace with actual known Z

object_yaw_rad = math.radians(object_global_yaw_deg)

with rosbag.Bag(bag_file, "r") as bag, open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'index', 'timestamp', 'address', 'timestamp_ms',
        'x_m', 'y_m', 'z_m', 'flags', 'angle_deg',
        'rotation_y_rad', 'alpha_rad'
    ])
    
    for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=[position_topic])):
        # Camera yaw in radians
        camera_yaw_rad = math.radians(msg.angle)

        # rotation_y: how the object is oriented relative to camera
        rotation_y = object_yaw_rad - camera_yaw_rad

        # Vector from camera to object
        dx = object_x - msg.x_m
        dz = object_z - msg.z_m

        # alpha: relative angle to object minus where it’s seen from
        alpha = rotation_y - math.atan2(dx, dz)

        writer.writerow([
            f"{idx:06d}",
            t.to_sec(),
            msg.address,
            msg.timestamp_ms,
            msg.x_m,
            msg.y_m,
            msg.z_m,
            msg.flags,
            msg.angle,
            rotation_y,
            alpha
        ])
