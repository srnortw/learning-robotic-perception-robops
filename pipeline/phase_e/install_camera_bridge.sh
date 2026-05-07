#!/bin/bash
# Install the camera bridge on the Pi (run once from this workstation):
#
#   bash pipeline/phase_e/install_camera_bridge.sh
#
# What it does:
#   1. Copies camera_bridge.py to /opt/robops/ on the Pi
#   2. Installs the systemd service
#   3. Enables and starts it
#
# Requirements on the Pi: python3-picamera2, python3-opencv
#   sudo apt-get install -y python3-picamera2 python3-opencv

set -e

PI_HOST="${PI_HOST:-pi3b-001}"
PI_USER="${PI_USER:-serkanrob}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[install_camera_bridge] Copying files to ${PI_USER}@${PI_HOST} ..."
ssh "${PI_USER}@${PI_HOST}" "sudo mkdir -p /opt/robops"
scp "${SCRIPT_DIR}/camera_bridge.py" \
    "${PI_USER}@${PI_HOST}:/tmp/camera_bridge.py"
scp "${SCRIPT_DIR}/robops-camera-bridge.service" \
    "${PI_USER}@${PI_HOST}:/tmp/robops-camera-bridge.service"

ssh "${PI_USER}@${PI_HOST}" "
  sudo cp /tmp/camera_bridge.py /opt/robops/camera_bridge.py
  sudo chmod +x /opt/robops/camera_bridge.py
  sudo cp /tmp/robops-camera-bridge.service /etc/systemd/system/robops-camera-bridge.service
  sudo systemctl daemon-reload
  sudo systemctl enable robops-camera-bridge.service
  sudo systemctl restart robops-camera-bridge.service
  echo '--- service status ---'
  sudo systemctl status robops-camera-bridge.service --no-pager
"

echo "[install_camera_bridge] Done. Camera bridge is running on the Pi."
echo "  Frames stream to udp://127.0.0.1:5000"
echo "  camera_node (Docker) will pick them up automatically."
