#!/usr/bin/env python3

import rospy
import numpy as np

from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from yolomous.msg import IBVR3d
from ibvr_control.msg import IBVRout
from math import sqrt, pi

class KalmanFilter1D:
    def __init__(self):
        self.x = 0.0
        self.P = 1.0
        self.Q = 1e-6
        self.R = 1e-2

    def update(self, measurement):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)
        return self.x

class NexusIBVRNode:
    def __init__(self):
        rospy.init_node('nexus_IBVR', anonymous=True)

        rospy.loginfo("Waiting for 10 seconds…")
        rospy.sleep(10)

        # Camera Intrinsics
        self.f_x = 487.9530498924
        self.f_y = 485.9382180841
        self.f_avg = (self.f_x + self.f_y) / 2.0
        self.image_width  = 640
        self.image_height = 320

        # Rotation gain
        self.K_yaw =  0.75

        # IBVR variables
        self.s_r    = 0.306
        self.kappa = np.array([
            5e-7,   # y
            0, 
            2e-7]   # x
            , dtype=float)
        self.diag_correction = 0.625

        self.sign_map = {
            0: -1,
            1: -1,
        }

        self.s_star = 8000
        self.max_lin = 1.0
        self.max_ang = 0.785

        self.d_star_map = {
            0: 3.0,
        }
        self.expected_ids = set(self.d_star_map.keys())
        self.s_star_map = {}     # Object map for desired area (s*)

        for agent_id, d_star in self.d_star_map.items():
            # s* = s_r * f_avg^2 / d*^2
            area_star = (self.s_r * (self.f_avg**2)) / (d_star**2)
            self.s_star_map[agent_id] = area_star
            rospy.loginfo(f"ID {agent_id}: d*={d_star:.2f} → s*={area_star:.6f}")

        self.depth_filters = {}  # Kalman for depth variable (z_r)
        self.s_d_ij = {}         # Object map for percieved area from distance (s) 
        self.error_ij = {}       # Object map for error (e)

        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.cmd_vel2_pub = rospy.Publisher('cmd_vel2', Twist, queue_size=1)
        self.ibvrout_pub = rospy.Publisher('ibvr_output', IBVRout , queue_size=1)
        rospy.Subscriber("/3Dnode/IBVR", IBVR3d, self.ibvr_callback)
        rospy.loginfo("NexusIBVRNode up and running.")
        rospy.spin()


    def ibvr_callback(self, msg):
        
        out = IBVRout()
        out.header = Header(stamp=rospy.Time.now(), frame_id="camera")

        current_ids = set(msg.ids)
        if not self.expected_ids.issubset(current_ids):
            rospy.logwarn("One or more targets lost – stopping IBVR motion")
            stop = Twist()
            self.cmd_vel_pub.publish(stop)
            self.cmd_vel2_pub.publish(stop)
            return

        total_u = np.zeros(3)

        out.ids       = []
        out.error     = []
        out.distance  = []
        out.depth     = []
        out.depth_est = []

        for i, obj_id in enumerate(msg.ids):
            diag = msg.diagonal_length[i]

            diag_corr = diag * self.diag_correction
            area_raw = (pi / 4.0) * diag_corr**2
            z_r_est = sqrt((self.s_r / area_raw)) * self.f_avg

            if obj_id not in self.depth_filters:
                self.depth_filters[obj_id] = KalmanFilter1D()
            z_r = self.depth_filters[obj_id].update(z_r_est)

            p_x = msg.center_x[i] - (self.image_width  / 2.0)
            p_y = msg.center_y[i] - (self.image_height / 2.0)

            zij = np.array([
                (z_r / self.f_avg) * p_x,
                (z_r / self.f_avg) * p_y,
                z_r
            ])
            norm_zij = np.linalg.norm(zij)
            z_hat = zij / norm_zij

            s_d = (self.s_r * (self.f_avg**2)) / (norm_zij**2)
            self.s_d_ij[obj_id] = s_d
            s_star = self.s_star_map.get(obj_id, self.s_star)

            e_ij = s_star - s_d
            u_ij = (- self.kappa * z_hat * s_d**2 * e_ij) / (self.s_r * self.f_avg**2)
            sign = self.sign_map.get(obj_id, +1)
            total_u += sign * u_ij

            out.ids.append(obj_id)
            out.error.append(e_ij)
            out.distance.append(norm_zij)
            out.depth.append(z_r)
            out.depth_est.append(z_r_est)

            rospy.loginfo(f"[ID {obj_id}] D={norm_zij:.2f}m, s_d={s_d:.3f}, s*={s_star:.3f}, e={e_ij:.3f}")

        offsets = [cx - (self.image_width/2.0) for cx in msg.center_x]
        mid_px  = sum(offsets)/len(offsets) if offsets else 0.0
        yaw_err = np.arctan2(mid_px, self.f_avg)

        twist = Twist()
        v_x = np.clip(total_u[2], -self.max_lin, self.max_lin)
        v_y = np.clip(total_u[0], -self.max_lin, self.max_lin)
        v_z = np.clip(self.K_yaw * yaw_err, -self.max_ang, self.max_ang)

        twist.linear.x  = v_x
        twist.linear.y  = v_y
        twist.angular.z  = v_z

        rospy.loginfo(f"→ IBVR cmd:  vx={v_x:.2f}  vy={v_y:.2f} vz={v_z:.2f}")

        out.cmd_vx = v_x
        out.cmd_vy = v_y
        out.cmd_wz = v_z

        self.cmd_vel_pub.publish(twist)
        self.cmd_vel2_pub.publish(twist)
        self.ibvrout_pub.publish(out)
        
if __name__ == '__main__':
    try:
        NexusIBVRNode()
    except rospy.ROSInterruptException:
        pass
