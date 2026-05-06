#!/bin/bash
# Phase E — Install camera_node as a host systemd service on the Pi.
#
# Run this once on the Pi (or call from CI via SSH) after a deploy.
# Requires ROS2 Humble already installed on the Pi host.
#
# Usage:
#   bash pipeline/phase_e/camera_node_host_setup.sh
#   or via SSH:
#   ssh pi3b-001 "bash -s" < pipeline/phase_e/camera_node_host_setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INSTALL_DIR="/opt/robops/camera_node"
SERVICE_FILE="/etc/systemd/system/robops-camera.service"
NODE_SCRIPT="$INSTALL_DIR/camera_node.py"

echo "[camera_node_host_setup] Installing to $INSTALL_DIR ..."
sudo mkdir -p "$INSTALL_DIR"
sudo cp "$REPO_ROOT/ros2_ws/src/camera_node/camera_node/camera_node.py" "$NODE_SCRIPT"
sudo chmod +x "$NODE_SCRIPT"

echo "[camera_node_host_setup] Writing systemd service ..."
sudo tee "$SERVICE_FILE" > /dev/null <<'EOF'
[Unit]
Description=RobOps camera_node — publishes /camera/image_raw via picamera2
After=network.target

[Service]
Type=simple
User=serkanrob
Environment="PYTHONPATH=/usr/lib/python3/dist-packages"
ExecStart=/bin/bash -c ". /opt/ros/humble/setup.bash && python3 /opt/robops/camera_node/camera_node.py"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "[camera_node_host_setup] Enabling and starting service ..."
sudo systemctl daemon-reload
sudo systemctl enable robops-camera.service
sudo systemctl restart robops-camera.service
sudo systemctl status robops-camera.service --no-pager
echo "[camera_node_host_setup] Done."
