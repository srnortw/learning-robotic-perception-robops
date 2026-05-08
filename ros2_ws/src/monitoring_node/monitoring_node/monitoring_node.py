"""
Phase F — Monitoring Node

Subscribes to /detr/shadow_metrics (JSON string from detr_node),
enriches each document with CPU/memory/node-liveness from psutil + ROS2,
and batch-writes to MongoDB production_metrics every BATCH_SIZE frames.

Topic:   /detr/shadow_metrics   (std_msgs/String, JSON)
Publishes: /monitoring/health   (std_msgs/String, JSON summary)
"""

import json
import os
from collections import deque
from datetime import datetime, timezone

import psutil
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False


BATCH_SIZE   = 10
MONGO_URI    = os.environ.get("MONGO_URI", "")
DB_NAME      = "robops"
COLLECTION   = "production_metrics"
ROBOT_ID     = os.environ.get("ROBOT_ID", "robops-pi3b-001")
MODEL_VER    = os.environ.get("DATASET_VERSION", "v1")

CRITICAL_NODES = ["camera_node", "detr_node", "monitoring_node"]


class MonitoringNode(Node):
    def __init__(self):
        super().__init__("monitoring_node")

        self._buffer: deque[dict] = deque()
        self._mongo_col = self._connect_mongo()

        self.subscription = self.create_subscription(
            String, "/detr/shadow_metrics", self._on_metrics, 10
        )
        self.health_pub = self.create_publisher(String, "/monitoring/health", 10)

        # Periodic health summary every 60 s
        self.create_timer(60.0, self._publish_health_summary)

        self.get_logger().info(
            f"monitoring_node started | robot={ROBOT_ID} | "
            f"mongo={'connected' if self._mongo_col else 'DISABLED'}"
        )

    # ── MongoDB ──────────────────────────────────────────────────────────────

    def _connect_mongo(self):
        if not MONGO_AVAILABLE or not MONGO_URI.strip():
            self.get_logger().warn("MONGO_URI not set — metrics will not be persisted.")
            return None
        uri = MONGO_URI.strip()
        if not uri.startswith(("mongodb://", "mongodb+srv://")):
            self.get_logger().warn(
                "MONGO_URI is not a valid MongoDB URI (e.g. Greengrass secret not "
                "resolved) — metrics will not be persisted."
            )
            return None
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.server_info()
            return client[DB_NAME][COLLECTION]
        except Exception as e:
            self.get_logger().error(f"MongoDB connection failed: {e}")
            return None

    # ── Metrics callback ─────────────────────────────────────────────────────

    def _on_metrics(self, msg: String):
        try:
            doc = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        doc["robot_id"]        = ROBOT_ID
        doc["model_version"]   = MODEL_VER
        doc["source"]          = "production"
        doc["cpu_percent"]     = psutil.cpu_percent(interval=None)
        doc["memory_mb"]       = psutil.virtual_memory().used // (1024 * 1024)
        doc["ros_nodes_alive"] = self._live_nodes()

        # Remove large frame payload before storing (s3_uploader handles that)
        doc.pop("frame_b64", None)

        self._buffer.append(doc)

        if len(self._buffer) >= BATCH_SIZE:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        docs = list(self._buffer)
        self._buffer.clear()
        if self._mongo_col is not None:
            try:
                self._mongo_col.insert_many(docs, ordered=False)
                self.get_logger().debug(f"Flushed {len(docs)} docs to MongoDB")
            except Exception as e:
                self.get_logger().error(f"MongoDB insert failed: {e}")

    # ── Node liveness ────────────────────────────────────────────────────────

    def _live_nodes(self) -> list[str]:
        """Return list of critical nodes that are currently alive."""
        try:
            node_names = [n for n, _ in self.get_node_names_and_namespaces()]
            return [n for n in CRITICAL_NODES if n in node_names]
        except Exception:
            return []

    # ── Periodic health summary ───────────────────────────────────────────────

    def _publish_health_summary(self):
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "robot_id": ROBOT_ID,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_mb": psutil.virtual_memory().used // (1024 * 1024),
            "ros_nodes_alive": self._live_nodes(),
            "buffer_pending": len(self._buffer),
        }
        self.health_pub.publish(String(data=json.dumps(summary)))

        # Flush any remaining docs on health tick
        self._flush()


def main(args=None):
    rclpy.init(args=args)
    node = MonitoringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._flush()   # drain buffer on shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
