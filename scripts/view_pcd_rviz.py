import subprocess
import rclpy
from sensor_msgs.msg import PointCloud2
from pypcd4 import PointCloud

rviz_process = subprocess.Popen(['rviz2'])

rclpy.init()
node = rclpy.create_node('pcd_pub')
pub = node.create_publisher(PointCloud2, '/pcd_points', 10)
msg = PointCloud.from_path('empty_gap_front.pcd').to_msg()
msg.header.frame_id = 'lidar'

def pub_cb():
    msg.header.stamp = node.get_clock().now().to_msg()
    pub.publish(msg)

node.create_timer(1.0, pub_cb)

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    rviz_process.terminate()
    node.destroy_node()
    rclpy.shutdown()