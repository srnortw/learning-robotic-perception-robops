FROM arm64v8/ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime for Pi3B+ (arm64 CPU-only build)
RUN pip3 install --no-cache-dir --break-system-packages \
    onnxruntime \
    numpy \
    pymongo \
    boto3

COPY ros2_ws/src/detr_node/ /ros2_ws/src/detr_node/
COPY pipeline/phase_b/label_schema.yaml /config/label_schema.yaml

WORKDIR /ros2_ws
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --packages-select detr_node --cmake-args -DCMAKE_BUILD_TYPE=Release

# Model path injected by Greengrass at runtime via -e MODEL_PATH={artifacts:path}/model_int8.onnx
ENV MODEL_PATH=/greengrass/v2/packages/artifacts/com.robops.inference/current/model_int8.onnx
ENV SHADOW_MODE=false
ENV CONFIDENCE_THRESHOLD=0.5
ENV ROBOT_ID=pi3b-001
# Restrict DDS discovery to loopback so it communicates with the ros2-stack
# container which also runs with --network host on the same Pi.
ENV ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

ENTRYPOINT ["/bin/bash", "-c", \
    ". /opt/ros/jazzy/setup.sh && \
     . /ros2_ws/install/setup.sh && \
     ros2 run detr_node detr_node \
       --ros-args \
       --params-file /ros2_ws/src/detr_node/config/detr_params.yaml \
       -p model_path:=${MODEL_PATH} \
       -p shadow_mode:=${SHADOW_MODE} \
       -p confidence_threshold:=${CONFIDENCE_THRESHOLD} \
       -p robot_id:=${ROBOT_ID}"]
