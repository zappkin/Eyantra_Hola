#!/usr/bin/env python3
"""
Boilerplate controller for the HE bot.

Fetches a shape from the get_shape service and builds a list of waypoints
to trace it. Fill in the control loop to drive the robot through them.


"""

import argparse
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from shape_interface.srv import GetShape

# Wheel <-> body-velocity mapping, columns are [left, right, back] wheel
# speed (rad/s); rows are body frame [vx, vy, wz] per unit wheel speed.


_CTRL_LIMIT = 0.0      # rad/s, matches lekiwi.xml actuator ctrlrange

WAYPOINT_TOLERANCE = 0.0   # metres
CIRCLE_SEGMENTS     = 120
POSITION_KP         = 0.0
YAW_HOLD_KP         = 0.0
CONTROL_PERIOD      = 0.0  


   
def body_velocity_to_wheel_speeds(vx, vy, w):
    r = 0.0255
    d = 0.06412
    w_left  = ( np.cos(np.pi / 6) * vx + np.sin(np.pi / 6) * vy + d * w) / r
    w_right = (-np.cos(np.pi / 6) * vx + np.sin(np.pi / 6) * vy + d * w) / r
    w_back  = (-vy + d * w) / r
    
    return [float(w_left), float(w_right), float(w_back)]

def yaw_from_quat(w, x, y, z):
    # TODO: convert quaternion to yaw (radians).
    pass


def _regular_polygon(cx, cy, n_sides, side_length, start_angle=math.pi / 2):
    """Vertices of a regular polygon centred on (cx, cy), closed back to the
    first vertex so the last waypoint returns the robot to where it started
    drawing."""
    r = side_length / (2 * math.sin(math.pi / n_sides))
    pts = [
        (cx + r * math.cos(start_angle + 2 * math.pi * i / n_sides),
         cy + r * math.sin(start_angle + 2 * math.pi * i / n_sides))
        for i in range(n_sides)
    ]
    return pts + [pts[0]]


def build_waypoints(shape_name, data):
    """World-frame waypoints for `shape_name`, as returned by the get_shape
    service: data[0:2] is the shape's centre (x, y); the remaining entries
    are its size parameters (see shape_service.cpp's shape_map)."""
    cx, cy = data[0], data[1]

    if shape_name == "Circle":
        radius = data[2]
        return [
            (cx + radius * math.cos(2 * math.pi * i / CIRCLE_SEGMENTS),
             cy + radius * math.sin(2 * math.pi * i / CIRCLE_SEGMENTS))
            for i in range(1, CIRCLE_SEGMENTS + 1)
        ]

    if shape_name == "Square":
        return _regular_polygon(cx, cy, 4, data[2], start_angle=math.pi / 4)

    if shape_name == "Triangle":
        return _regular_polygon(cx, cy, 3, data[2])

    if shape_name == "Pentagon":
        return _regular_polygon(cx, cy, 5, data[2])

    if shape_name == "Rectangle":
        w, h = data[2], data[3]
        corners = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        return corners + [corners[0]]

    raise ValueError(f"unknown shape '{shape_name}'")


class ShapeController(Node):
    def __init__(self, speed):
        super().__init__("shape_controller")
        self.speed = speed
        
        # State variables
        self.curr_pose = None
        self.start_pose = None
        self.waypoint_index = 0
        self.done = False
        
        # Publishers and Subscribers (Renamed to cmd_pub!)
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/wheel_commands', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        
        # Request shape and unpack BOTH the name and the waypoints!
        self.shape_name, self.waypoints = self._request_shape()
        self.timer = self.create_timer(0.01, self.control_step)
#Add the publsiher and subscriber scripts
    
    def _request_shape(self):
        client = self.create_client(GetShape, "get_shape")
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for get_shape service...")

        future = client.call_async(GetShape.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(
                f"get_shape service call failed: {response and response.message}"
            )

        return response.shape_name, build_waypoints(response.shape_name, list(response.data))

    def odom_cb(self, msg):
        # Extract X and Y
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Convert Quaternion to Yaw (rotation around Z axis)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        import math
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Update current pose
        self.curr_pose = [x, y, yaw]
        
        # Record the start pose on the very first callback
        if self.start_pose is None:
            self.start_pose = [x, y, yaw]

    def _publish(self, wheels):
        self.cmd_pub.publish(Float64MultiArray(data=wheels))

    def control_step(self):
        if self.curr_pose is None or self.done:
            return

        curr_x, curr_y, curr_yaw = self.curr_pose
        target_x, target_y = self.waypoints[self.waypoint_index]

        # Waypoint threshold: 0.04m avoids getting stuck while staying tight
        distance = math.hypot(target_x - curr_x, target_y - curr_y)
        if distance < 0.04:
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.waypoints):
                self.done = True
                self._publish([0.0, 0.0, 0.0])
                self.get_logger().info("Shape complete!")
                return
            target_x, target_y = self.waypoints[self.waypoint_index]

        # Calculate tracking errors
        error_x = target_x - curr_x
        error_y = target_y - curr_y

        # Proportional velocity in world frame
        Kp = 2.4
        vx_world = Kp * error_x
        vy_world = Kp * error_y

        # Transform to robot body frame using current yaw
        vx_body = vx_world * math.cos(curr_yaw) + vy_world * math.sin(curr_yaw)
        vy_body = -vx_world * math.sin(curr_yaw) + vy_world * math.cos(curr_yaw)

        # Cap velocity to avoid wheel slip
        max_speed = 0.25
        speed = math.hypot(vx_body, vy_body)
        if speed > max_speed:
            vx_body = (vx_body / speed) * max_speed
            vy_body = (vy_body / speed) * max_speed

        # Convert to wheel speeds and send
        wheels = body_velocity_to_wheel_speeds(vx_body, vy_body, 0.0)
        self._publish(wheels) 
     
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.25,
                         help="max approach speed, m/s")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ShapeController(args.speed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish([0.0, 0.0, 0.0])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
