#!/usr/bin/env python3
"""
RobOps — Camera Bridge (Pi OS host, Python 3.11)

Captures frames from the Pi Camera V2 (IMX219) using picamera2 / libcamera
and sends them over UDP to the ROS2 camera_node running inside Docker.

Protocol (one datagram per frame):
    [ 4 bytes big-endian uint32: JPEG length ][ JPEG bytes ]

The receiver (camera_node.py) listens on the same host interface.
Because both containers use --network host, 127.0.0.1:CAMERA_BRIDGE_PORT
is reachable from inside Docker without any port mapping.

Usage:
    python3 camera_bridge.py               # default 640x480 @ 5 FPS, port 5000
    python3 camera_bridge.py --width 320 --height 240 --fps 3 --port 5001
    python3 camera_bridge.py --jpeg-quality 60

Install on Pi (once):
    sudo cp pipeline/phase_e/camera_bridge.py /opt/robops/camera_bridge.py
    sudo cp pipeline/phase_e/robops-camera-bridge.service \
            /etc/systemd/system/robops-camera-bridge.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now robops-camera-bridge
"""

import argparse
import socket
import struct
import sys
import time

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("picamera2 not available — run on Pi OS Bookworm with python3-picamera2 installed")

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="RobOps camera bridge: picamera2 → UDP MJPEG")
    p.add_argument("--host",         default="127.0.0.1",  help="Destination IP (default: 127.0.0.1)")
    p.add_argument("--port",         type=int, default=5000, help="Destination UDP port (default: 5000)")
    p.add_argument("--width",        type=int, default=640)
    p.add_argument("--height",       type=int, default=480)
    p.add_argument("--fps",          type=int, default=5)
    p.add_argument("--jpeg-quality", type=int, default=75,  help="JPEG quality 1-100 (default: 75)")
    return p.parse_args()


def main():
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.host, args.port)

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"format": "RGB888", "size": (args.width, args.height)},
        controls={"FrameRate": float(args.fps)},
    )
    cam.configure(config)
    cam.start()
    print(f"[camera_bridge] streaming {args.width}x{args.height}@{args.fps}fps "
          f"→ udp://{args.host}:{args.port}  quality={args.jpeg_quality}", flush=True)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    interval = 1.0 / args.fps
    last = time.monotonic()

    try:
        while True:
            rgb = cam.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, jpg = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                print("[camera_bridge] WARN: encode failed — skipping frame", flush=True)
                continue

            payload = jpg.tobytes()
            header  = struct.pack(">I", len(payload))
            # UDP max safe payload ~65507 bytes; JPEG at 640x480 q75 is usually <30 KB
            sock.sendto(header + payload, dest)

            # pace to target FPS
            elapsed = time.monotonic() - last
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            last = time.monotonic()

    except KeyboardInterrupt:
        print("[camera_bridge] stopping", flush=True)
    finally:
        cam.stop()
        sock.close()


if __name__ == "__main__":
    main()
