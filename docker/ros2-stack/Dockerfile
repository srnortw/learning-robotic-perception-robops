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
    boto3

COPY ros2_ws/src/camera_node/ /ros2_ws/src/camera_node/

WORKDIR /ros2_ws
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --packages-select camera_node --cmake-args -DCMAKE_BUILD_TYPE=Release

ENV ROBOT_ID=pi3b-001
ENV CAMERA_DEVICE=0
ENV CAMERA_FPS=5

ENTRYPOINT ["/bin/bash", "-c", \
    ". /opt/ros/jazzy/setup.sh && \
     . /ros2_ws/install/setup.sh && \
     ros2 run camera_node camera_node \
       --ros-args \
       -p device_index:=${CAMERA_DEVICE} \
       -p fps:=${CAMERA_FPS}"]
