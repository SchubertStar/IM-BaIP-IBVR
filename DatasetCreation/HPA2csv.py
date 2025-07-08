import rosbag
import csv
import math

def wrap_to_pi(angle):
    """Wraps angle to [-π, π]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

# Inputs
bag_file = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/rawROSbagdata/ROSbags/finalbig1.bag"
output_file = "/mnt/c/Users/ianmi/Desktop/Integration Project/IM-BaIP-IBVR/Nexus3DBB/rawROSbagdata/hedge_pos_ang/finalbig1adjusted.csv"
position_topic = "/sync/hedge_pos_ang"

# Fixed object global position (in ENU) and orientation
object_global_yaw_deg = 0.0
object_global_yaw_rad = math.radians(object_global_yaw_deg)
object_GNSSx = 3.6
object_GNSSy = 3.7
object_GNSSz = 0

# Camera offset in robot frame (forward, right, up)
cam_offset_forward = 0.12  # 12 cm in front of GNSS
cam_offset_down = 0.06     # 6 cm lower than GNSS

with rosbag.Bag(bag_file, "r") as bag, open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'index', 'timestamp', 'cam_x', 'cam_y', 'cam_z',
        'cam_ang', 'rel_X', 'rel_Y', 'rel_Z',
        'rotation_y', 'alpha_rad'
    ])

    for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=[position_topic])):

        if idx == 0:
            start_time = t.to_sec()

        time_since_start = t.to_sec() - start_time

        # GNSS pose
        gnss_x = msg.x_m
        gnss_y = msg.y_m
        gnss_z = 0.14
        yaw_deg = msg.angle

        # Convert angle to radians and relevant coordinate system for rotation
        yaw_math = math.radians(90.0 - yaw_deg)

        # Apply camera offset in ENU frame (rotate offset by yaw)
        cam_x = gnss_x + cam_offset_forward * math.cos(yaw_math)
        cam_y = gnss_y + cam_offset_forward * math.sin(yaw_math)
        cam_z = - cam_offset_down  # lower = subtract

        # Compute relative vector from camera to object in world (ENU) frame
        dx = object_GNSSx - cam_x
        dy = object_GNSSy - cam_y
        dz = object_GNSSz - cam_z

        # Transform world-relative vector into KITTI camera frame
        # Using ENU → camera frame conversion: yaw_math = 90° - angle

        # Rotate dx, dy into camera frame
        rel_X = dx * math.cos(yaw_math) - dy * math.sin(yaw_math)
        rel_Z = dx * math.sin(yaw_math) + dy * math.cos(yaw_math)
        rel_Y = -dz 

        # Rotation of object in camera frame (yaw - camera yaw)
        rotation_y = wrap_to_pi(object_global_yaw_rad - yaw_math)

        # Observation angle (bearing from camera center)
        alpha = wrap_to_pi(rotation_y - math.atan2(rel_X, rel_Z))

        # Write to CSV
        writer.writerow([
            f"{idx:06d}",
            time_since_start,
            cam_x, cam_y, cam_z,
            yaw_deg,
            rel_X, rel_Y, rel_Z,
            rotation_y,
            alpha
        ])