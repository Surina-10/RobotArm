#!/usr/bin/env python3
import csv
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointRecorder(Node):
    def __init__(self):
        super().__init__('joint_recorder')
        
        # Subscribe to joint states from joint_state_publisher
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10
        )
        
        # Open CSV file to log data
        self.csv_file = open('motion_demonstration.csv', mode='w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.header_written = False
        
        self.get_logger().info('Recording joint states... Move sliders in GUI. Press Ctrl+C to stop.')

    def listener_callback(self, msg: JointState):
        # Write CSV header on first received message
        if not self.header_written:
            header = ['timestamp'] + list(msg.name)
            self.writer.writerow(header)
            self.header_written = True

        # Extract timestamp and joint positions
        timestamp = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        row = [timestamp] + list(msg.position)
        
        self.writer.writerow(row)

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = JointRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
