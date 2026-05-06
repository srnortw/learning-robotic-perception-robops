#!/bin/bash
set -e

. /opt/ros/jazzy/setup.sh
. /ros2_ws/install/setup.sh

cleanup() {
    echo "[entrypoint] Shutting down nodes..."
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    echo "[entrypoint] Shutdown complete."
}
trap cleanup SIGTERM SIGINT

# camera_node runs as a host systemd service (robops-camera.service)
# using the Pi's native Python 3.11 + picamera2 to avoid Python ABI
# conflicts between Pi OS Bookworm (3.11) and this Ubuntu 24.04 image (3.12).

ros2 run monitoring_node monitoring_node \
    --ros-args \
    -p robot_id:="${ROBOT_ID:-pi3b-001}" \
    -p dataset_version:="${DATASET_VERSION:-v1}" &
MONITOR_PID=$!

wait "$MONITOR_PID"
EXIT_CODE=$?
echo "[entrypoint] monitoring_node exited with code ${EXIT_CODE} — stopping container."
cleanup
exit "$EXIT_CODE"
