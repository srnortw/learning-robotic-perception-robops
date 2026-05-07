FROM arm64v8/ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    pymongo \
    boto3 \
    psutil

COPY ros2_ws/src/camera_node/     /ros2_ws/src/camera_node/
COPY ros2_ws/src/monitoring_node/ /ros2_ws/src/monitoring_node/

WORKDIR /ros2_ws
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build \
      --packages-select camera_node monitoring_node \
      --cmake-args -DCMAKE_BUILD_TYPE=Release

ENV ROBOT_ID=robops-pi3b-001
ENV CAMERA_DEVICE=0
ENV CAMERA_FPS=5
ENV DATASET_VERSION=v1

COPY docker/ros2-stack/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Disable Fast-DDS shared memory transport so messages cross Docker IPC
# namespace boundaries (both containers use --network host but separate IPC).
COPY docker/fastdds_no_shm.xml /etc/fastdds/dds_profile.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/etc/fastdds/dds_profile.xml

ENTRYPOINT ["/entrypoint.sh"]
