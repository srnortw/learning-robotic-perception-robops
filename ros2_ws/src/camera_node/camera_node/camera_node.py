"""
RobOps camera_node — UDP MJPEG receiver → /camera/image_raw

Receives JPEG frames sent by the Pi OS host bridge (camera_bridge.py)
over a local UDP socket and republishes them as sensor_msgs/Image on
/camera/image_raw for DETR inference.

Protocol (matches camera_bridge.py):
    [ 4 bytes big-endian uint32: JPEG length ][ JPEG bytes ]

Parameters (ROS2):
    udp_host        (str,   default "0.0.0.0")  bind address
    udp_port        (int,   default 5000)        listen port
    frame_id        (str,   default "camera_link")
    recv_timeout_s  (float, default 2.0)         seconds to wait per recvfrom
"""

import socket
import struct

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

_MAX_UDP = 65536   # max UDP datagram size


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('udp_host',       '0.0.0.0')
        self.declare_parameter('udp_port',       5000)
        self.declare_parameter('frame_id',       'camera_link')
        self.declare_parameter('recv_timeout_s', 2.0)

        host     = self.get_parameter('udp_host').value
        port     = self.get_parameter('udp_port').value
        timeout  = self.get_parameter('recv_timeout_s').value
        self.frame_id = self.get_parameter('frame_id').value

        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge    = CvBridge()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(timeout)
        self._sock.bind((host, port))

        self.get_logger().info(
            f'camera_node listening on udp://{host}:{port} '
            f'(timeout {timeout}s) → /camera/image_raw'
        )

        # Poll the socket on every spin cycle instead of a fixed timer
        # so we publish as fast as frames arrive.
        self.create_timer(0.001, self._recv_and_publish)

    # ------------------------------------------------------------------
    def _recv_and_publish(self):
        try:
            data, _ = self._sock.recvfrom(_MAX_UDP)
        except socket.timeout:
            return
        except OSError as exc:
            self.get_logger().warn(f'UDP recv error: {exc}', throttle_duration_sec=5.0)
            return

        if len(data) < 4:
            return

        length = struct.unpack(">I", data[:4])[0]
        jpeg   = data[4:]

        if len(jpeg) < length:
            self.get_logger().warn(
                f'Truncated datagram: expected {length}B, got {len(jpeg)}B',
                throttle_duration_sec=5.0,
            )
            return

        arr   = np.frombuffer(jpeg[:length], dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('Failed to decode JPEG frame', throttle_duration_sec=5.0)
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)

    # ------------------------------------------------------------------
    def destroy_node(self):
        self._sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
