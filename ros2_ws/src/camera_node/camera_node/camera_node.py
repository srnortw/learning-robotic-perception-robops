import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('device_index', 0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 5)
        self.declare_parameter('frame_id', 'camera_link')

        self.device_index = self.get_parameter('device_index').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.frame_id = self.get_parameter('frame_id').value

        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.cap = None
        self._init_camera()

        period = 1.0 / self.fps
        self.timer = self.create_timer(period, self.publish_frame)
        self.get_logger().info(
            f'camera_node started: device={self.device_index} '
            f'{self.width}x{self.height} @ {self.fps} FPS'
        )

    def _init_camera(self):
        self.cap = cv2.VideoCapture(self.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera device {self.device_index}')

    def publish_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().warn('Camera not open — skipping frame')
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame — skipping')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)

    def destroy_node(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
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
