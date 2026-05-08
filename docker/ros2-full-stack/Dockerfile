FROM arm64v8/ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    ros-jazzy-rmw-cyclonedds-cpp \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime for Pi3B+ (arm64 CPU-only build)
RUN pip3 install --no-cache-dir --break-system-packages \
    onnxruntime \
    numpy \
    pymongo \
    boto3 \
    psutil

COPY ros2_ws/src/camera_node/     /ros2_ws/src/camera_node/
COPY ros2_ws/src/detr_node/       /ros2_ws/src/detr_node/
COPY ros2_ws/src/monitoring_node/ /ros2_ws/src/monitoring_node/
COPY pipeline/phase_b/label_schema.yaml /config/label_schema.yaml

WORKDIR /ros2_ws
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build \
      --packages-select camera_node detr_node monitoring_node \
      --cmake-args -DCMAKE_BUILD_TYPE=Release

ENV ROBOT_ID=robops-pi3b-001
ENV DATASET_VERSION=v2
ENV MODEL_PATH=/greengrass/v2/packages/artifacts/com.robops.stack/current/model_int8.onnx
ENV SHADOW_MODE=false
ENV CONFIDENCE_THRESHOLD=0.5

# All nodes run in the same container = same DDS participant = no cross-container issues.
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

COPY docker/ros2-full-stack/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
