#!/usr/bin/env python3
# ============================================================================
# APPROACH 3: inverse-kinematics teleoperation (ROS 2)
#
# Uses the ikpy library to solve inverse kinematics for a target point from
# the arm's URDF, and publishes the resulting joint angles on /joint_states
# 20 times a second so the arm model follows the target.
# ============================================================================

import time
import ikpy.chain
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Load chain specifying the actual root link in your URDF
chain = ikpy.chain.Chain.from_urdf_file(
    "/home/surina/LSS-ROS2-Arms/lss_arm_description/urdf/lss_arm.urdf",
    base_elements=["lss_arm_link_0"]
)

class IKTeleop(Node):
    def __init__(self):
        super().__init__('ik_teleop')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # Initial target position [X, Y, Z] in meters relative to base
        self.target_position = [0.15, 0.0, 0.15]
        self.timer = self.create_timer(0.05, self.update_and_publish)
        
        self.get_logger().info("IK Teleop running! Publishing smooth joint states...")

    def update_and_publish(self):
        # Solve Inverse Kinematics for target position [x, y, z]
        joint_angles = chain.inverse_kinematics(self.target_position)
        
        # Get active joint names from ikpy chain (excluding fixed base)
        joint_names = [link.name for link in chain.links if link.name and 'joint' in link.name]
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = joint_names
        # Map solved angles to corresponding joint names
        msg.position = list(joint_angles[1:len(joint_names)+1])
        
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = IKTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
