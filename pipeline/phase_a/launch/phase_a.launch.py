"""
Phase A launch file — starts camera_node, detr_node, s3_uploader, mongo_writer.
Usage:
    ros2 launch pipeline/phase_a/launch/phase_a.launch.py
Optional overrides:
    ros2 launch ... model_path:=/path/to/model.onnx robot_id:=pi3b-002
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_config = os.path.join(
        get_package_share_directory('camera_node'), 'config', 'camera_params.yaml'
    )
    detr_config = os.path.join(
        get_package_share_directory('detr_node'), 'config', 'detr_params.yaml'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/models/detr/model.onnx',
        description='Path to DETR ONNX model file',
    )
    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='pi3b-001',
        description='Unique robot identifier',
    )

    camera = Node(
        package='camera_node',
        executable='camera_node',
        name='camera_node',
        parameters=[camera_config],
        output='screen',
    )

    detr = Node(
        package='detr_node',
        executable='detr_node',
        name='detr_node',
        parameters=[
            detr_config,
            {'model_path': LaunchConfiguration('model_path')},
            {'robot_id': LaunchConfiguration('robot_id')},
        ],
        output='screen',
    )

    s3_uploader = Node(
        package='detr_node',
        executable='s3_uploader',
        name='s3_uploader',
        output='screen',
    )

    mongo_writer = Node(
        package='detr_node',
        executable='mongo_writer',
        name='mongo_writer',
        output='screen',
    )

    return LaunchDescription([
        model_path_arg,
        robot_id_arg,
        camera,
        detr,
        s3_uploader,
        mongo_writer,
    ])
