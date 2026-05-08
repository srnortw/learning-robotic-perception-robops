#!/bin/bash
set -e

. /opt/ros/jazzy/setup.sh
. /ros2_ws/install/setup.sh

cleanup() {
    echo "[entrypoint] Shutting down nodes..."
    kill "$CAMERA_PID" "$DETR_PID" "$MONITOR_PID" 2>/dev/null || true
    wait "$CAMERA_PID" "$DETR_PID" "$MONITOR_PID" 2>/dev/null || true
    echo "[entrypoint] Shutdown complete."
}
trap cleanup SIGTERM SIGINT

# camera_node receives JPEG frames from the Pi OS camera bridge via UDP
ros2 run camera_node camera_node \
    --ros-args \
    -p udp_port:="${CAMERA_BRIDGE_PORT:-5000}" \
    -p frame_id:="camera_link" &
CAMERA_PID=$!

# detr_node subscribes to /camera/image_raw (same container = same DDS participant)
ros2 run detr_node detr_node \
    --ros-args \
    --params-file /ros2_ws/src/detr_node/config/detr_params.yaml \
    -p model_path:="${MODEL_PATH}" \
    -p shadow_mode:="${SHADOW_MODE:-false}" \
    -p confidence_threshold:="${CONFIDENCE_THRESHOLD:-0.5}" \
    -p robot_id:="${ROBOT_ID:-robops-pi3b-001}" &
DETR_PID=$!

ros2 run monitoring_node monitoring_node \
    --ros-args \
    -p robot_id:="${ROBOT_ID:-robops-pi3b-001}" \
    -p dataset_version:="${DATASET_VERSION:-v2}" &
MONITOR_PID=$!

wait -n "$CAMERA_PID" "$DETR_PID" "$MONITOR_PID"
EXIT_CODE=$?
echo "[entrypoint] A node exited with code ${EXIT_CODE} — stopping container."
cleanup
exit "$EXIT_CODE"
