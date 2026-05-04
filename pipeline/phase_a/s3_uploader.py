"""
S3 image uploader — subscribes to /detr/shadow_metrics, uploads frames that
meet the upload criteria (low confidence or every-Nth), writes the S3 URL
back to MongoDB via mongo_writer's queue.
"""

import base64
import json
import os
import sys
from datetime import datetime, timezone

import boto3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

BUCKET = os.environ.get('ROBOPS_S3_BUCKET', 'my-perception-robops-data-2026-688567275774-eu-central-1-an')
AWS_REGION = os.environ.get('AWS_DEFAULT_REGION', 'eu-central-1')


class S3Uploader(Node):
    def __init__(self):
        super().__init__('s3_uploader')
        self.s3 = boto3.client('s3', region_name=AWS_REGION)
        self.subscription = self.create_subscription(
            String, '/detr/shadow_metrics', self.metrics_callback, 10
        )
        self.url_pub = self.create_publisher(String, '/detr/s3_urls', 10)
        self.get_logger().info(f's3_uploader started — bucket: {BUCKET}')

    def metrics_callback(self, msg: String):
        try:
            doc = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON parse error: {e}')
            return

        if not doc.get('should_upload') or 'frame_b64' not in doc:
            return

        frame_bytes = base64.b64decode(doc['frame_b64'])
        s3_key = self._build_s3_key(doc)

        try:
            self.s3.put_object(
                Bucket=BUCKET,
                Key=s3_key,
                Body=frame_bytes,
                ContentType='image/jpeg',
                Metadata={
                    'confidence': str(doc.get('mean_confidence', 0.0)),
                    'robot_id': doc.get('robot_id', ''),
                    'timestamp': doc.get('timestamp', ''),
                    'architecture': doc.get('architecture', ''),
                    'source': doc.get('source', ''),
                },
            )
            s3_url = f's3://{BUCKET}/{s3_key}'
            self.get_logger().info(f'Uploaded frame → {s3_url}')

            reply = {**doc, 's3_url': s3_url}
            reply.pop('frame_b64', None)
            self.url_pub.publish(String(data=json.dumps(reply)))

        except Exception as e:
            self.get_logger().error(f'S3 upload failed: {e}')

    @staticmethod
    def _build_s3_key(doc: dict) -> str:
        ts = datetime.now(timezone.utc)
        date_str = ts.strftime('%Y-%m-%d')
        robot_id = doc.get('robot_id', 'unknown')
        frame_num = doc.get('frame_count', 0)
        return f'images/detr/{date_str}/{robot_id}/frame_{frame_num:06d}.jpg'


def main(args=None):
    rclpy.init(args=args)
    node = S3Uploader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
