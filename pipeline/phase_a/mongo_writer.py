"""
MongoDB telemetry writer — subscribes to /detr/s3_urls (enriched docs from
s3_uploader) and batches inserts into MongoDB Atlas telemetry collection.
Batch size 10 reduces free-tier write ops.
"""

import json
import os

import pymongo
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

MONGO_URI = os.environ.get('MONGO_URI', '')
DB_NAME = 'robops'
COLLECTION = 'telemetry'
BATCH_SIZE = 10


class MongoWriter(Node):
    def __init__(self):
        super().__init__('mongo_writer')

        if not MONGO_URI:
            self.get_logger().error('MONGO_URI env var not set — mongo_writer will not persist data.')

        self.client = None
        self.collection = None
        self._connect()

        self.buffer: list[dict] = []
        self.subscription = self.create_subscription(
            String, '/detr/s3_urls', self.doc_callback, 10
        )
        # Flush on a timer even if buffer never fills (graceful partial batches)
        self.flush_timer = self.create_timer(30.0, self.flush)
        self.get_logger().info(f'mongo_writer started — db={DB_NAME}.{COLLECTION} batch={BATCH_SIZE}')

    def _connect(self):
        if not MONGO_URI:
            return
        try:
            self.client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db = self.client[DB_NAME]
            self.collection = db[COLLECTION]
            self._ensure_indexes()
            self.get_logger().info('MongoDB connected')
        except Exception as e:
            self.get_logger().error(f'MongoDB connection failed: {e}')

    def _ensure_indexes(self):
        if self.collection is None:
            return
        self.collection.create_index('timestamp')
        self.collection.create_index('architecture')
        self.collection.create_index('mean_confidence')
        self.collection.create_index('source')

    def doc_callback(self, msg: String):
        try:
            doc = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON parse error: {e}')
            return

        self.buffer.append(doc)
        if len(self.buffer) >= BATCH_SIZE:
            self.flush()

    def flush(self):
        if not self.buffer or self.collection is None:
            return
        try:
            self.collection.insert_many(self.buffer, ordered=False)
            self.get_logger().info(f'Flushed {len(self.buffer)} docs to MongoDB')
            self.buffer.clear()
        except Exception as e:
            self.get_logger().error(f'MongoDB insert failed: {e}')

    def destroy_node(self):
        self.flush()
        if self.client:
            self.client.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MongoWriter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
