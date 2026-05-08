#!/bin/bash
set -e

. /opt/ros/jazzy/setup.sh
. /ros2_ws/install/setup.sh

cleanup() {
    echo "[entrypoint] Shutting down nodes..."
    kill "$CAMERA_PID" "$MONITOR_PID" 2>/dev/null || true
    wait "$CAMERA_PID" "$MONITOR_PID" 2>/dev/null || true
    echo "[entrypoint] Shutdown complete."
}
trap cleanup SIGTERM SIGINT

# camera_node listens on UDP for JPEG frames sent by the Pi OS host bridge
# (camera_bridge.py / robops-camera-bridge.service).  --network host means
# 127.0.0.1:CAMERA_BRIDGE_PORT is the same namespace as the Pi host.
ros2 run camera_node camera_node \
    --ros-args \
    -p udp_port:="${CAMERA_BRIDGE_PORT:-5000}" \
    -p frame_id:="camera_link" &
CAMERA_PID=$!

ros2 run monitoring_node monitoring_node \
    --ros-args \
    -p robot_id:="${ROBOT_ID:-pi3b-001}" \
    -p dataset_version:="${DATASET_VERSION:-v1}" &
MONITOR_PID=$!

# Exit if either node dies
wait -n "$CAMERA_PID" "$MONITOR_PID"
EXIT_CODE=$?
echo "[entrypoint] A node exited with code ${EXIT_CODE} — stopping container."
cleanup
exit "$EXIT_CODE"
