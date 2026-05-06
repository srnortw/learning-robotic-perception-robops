import json
import os
from collections import Counter
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge

from .onnx_inference import DetrOnnxInference


class DetrNode(Node):
    def __init__(self):
        super().__init__('detr_node')

        self.declare_parameter('model_path', '/models/detr/model.onnx')
        self.declare_parameter('shadow_mode', True)
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('upload_every_nth_frame', 30)
        self.declare_parameter('robot_id', 'pi3b-001')
        self.declare_parameter('architecture', 'detr')
        self.declare_parameter('class_names', ['person', 'chair', 'table', 'door'])

        self.model_path = self.get_parameter('model_path').value
        self.shadow_mode = self.get_parameter('shadow_mode').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.upload_every_nth = self.get_parameter('upload_every_nth_frame').value
        self.robot_id = self.get_parameter('robot_id').value
        self.architecture = self.get_parameter('architecture').value
        self.class_names = list(self.get_parameter('class_names').value)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.model = None
        self._load_model()

        # Queue depth 1 — always process latest frame, drop stale ones
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 1
        )
        self.detection_pub = self.create_publisher(Detection2DArray, '/detr/detections', 10)
        self.metrics_pub = self.create_publisher(String, '/detr/shadow_metrics', 10)

        self.get_logger().info(
            f'detr_node started | shadow_mode={self.shadow_mode} | '
            f'model={self.model_path} | robot_id={self.robot_id} | '
            f'classes={self.class_names}'
        )

    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.get_logger().warn(
                f'Model not found at {self.model_path} — inference disabled until model is present.'
            )
            return
        try:
            self.model = DetrOnnxInference(self.model_path, class_names=self.class_names)
            self.get_logger().info(f'ONNX model loaded from {self.model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')

    def image_callback(self, msg: Image):
        if self.model is None:
            return

        self.frame_count += 1
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        detections = self.model.run(frame)
        self._publish_detections(msg, detections)

        should_upload = (
            not detections
            or self._mean_confidence(detections) < self.confidence_threshold
            or self.frame_count % self.upload_every_nth == 0
        )
        self._publish_shadow_metrics(detections, should_upload, frame if should_upload else None)

    def _publish_detections(self, source_msg: Image, detections: list[dict]):
        array_msg = Detection2DArray()
        array_msg.header = source_msg.header

        for det in detections:
            d = Detection2D()
            d.header = source_msg.header

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(det['class_id'])
            hyp.hypothesis.score = det['confidence']
            d.results.append(hyp)

            cx, cy, w, h = det['box']
            d.bbox.center.position.x = cx
            d.bbox.center.position.y = cy
            d.bbox.size_x = w
            d.bbox.size_y = h

            array_msg.detections.append(d)

        self.detection_pub.publish(array_msg)

    def _publish_shadow_metrics(self, detections: list[dict], should_upload: bool, frame=None):
        mean_conf = self._mean_confidence(detections)
        class_dist = Counter(d['label'] for d in detections)

        doc = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'robot_id': self.robot_id,
            'architecture': self.architecture,
            'frame_count': self.frame_count,
            'mean_confidence': round(mean_conf, 4),
            'num_detections': len(detections),
            'class_distribution': dict(class_dist),
            'should_upload': should_upload,
            'source': 'shadow' if self.shadow_mode else 'production',
            's3_url': None,  # filled in by s3_uploader after upload
        }

        if should_upload and frame is not None:
            # Encode frame as JPEG bytes and attach to message for s3_uploader
            import cv2
            import base64
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            doc['frame_b64'] = base64.b64encode(buf.tobytes()).decode()

        self.metrics_pub.publish(String(data=json.dumps(doc)))

    @staticmethod
    def _mean_confidence(detections: list[dict]) -> float:
        if not detections:
            return 0.0
        return sum(d['confidence'] for d in detections) / len(detections)


def main(args=None):
    rclpy.init(args=args)
    node = DetrNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
