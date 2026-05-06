import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 5)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('use_picamera2', True)
        # fallback V4L2 device index (used only when picamera2 is unavailable)
        self.declare_parameter('device_index', 0)

        self.width   = self.get_parameter('width').value
        self.height  = self.get_parameter('height').value
        self.fps     = self.get_parameter('fps').value
        self.frame_id = self.get_parameter('frame_id').value
        use_picamera2 = self.get_parameter('use_picamera2').value

        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.picam  = None
        self.cap    = None

        if use_picamera2 and _PICAMERA2_AVAILABLE:
            self._init_picamera2()
        else:
            if use_picamera2 and not _PICAMERA2_AVAILABLE:
                self.get_logger().warn(
                    'picamera2 not installed — falling back to OpenCV V4L2'
                )
            self._init_opencv()

        period = 1.0 / self.fps
        self.timer = self.create_timer(period, self.publish_frame)

    # ------------------------------------------------------------------
    def _init_picamera2(self):
        try:
            self.picam = Picamera2()
            config = self.picam.create_video_configuration(
                main={"format": "RGB888", "size": (self.width, self.height)},
                controls={"FrameRate": float(self.fps)},
            )
            self.picam.configure(config)
            self.picam.start()
            self.get_logger().info(
                f'camera_node started via picamera2: '
                f'{self.width}x{self.height} @ {self.fps} FPS'
            )
        except Exception as exc:
            self.get_logger().error(
                f'picamera2 init failed ({exc}) — falling back to OpenCV V4L2'
            )
            self.picam = None
            self._init_opencv()

    def _init_opencv(self):
        idx = self.get_parameter('device_index').value
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera device {idx}')
        else:
            self.get_logger().info(
                f'camera_node started via OpenCV V4L2: '
                f'device={idx} {self.width}x{self.height} @ {self.fps} FPS'
            )

    # ------------------------------------------------------------------
    def publish_frame(self):
        frame = None

        if self.picam is not None:
            try:
                # picamera2 returns RGB888 — convert to BGR for cv_bridge
                rgb = self.picam.capture_array()
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as exc:
                self.get_logger().warn(f'picamera2 capture failed: {exc}')
                return
        elif self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('Failed to read frame — skipping')
                return
        else:
            self.get_logger().warn('No camera available — skipping frame', throttle_duration_sec=5.0)
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.header.frame_id  = self.frame_id
        self.publisher.publish(msg)

    # ------------------------------------------------------------------
    def destroy_node(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None and self.cap.isOpened():
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
