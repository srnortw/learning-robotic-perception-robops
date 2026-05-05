from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robops',
    maintainer_email='robops@local',
    description='USB/CSI camera publisher for RoboOps perception pipeline.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = camera_node.camera_node:main',
        ],
    },
)
