from setuptools import find_packages, setup

package_name = 'monitoring_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robops',
    maintainer_email='robops@robops.local',
    description='Phase F monitoring node — writes production metrics to MongoDB',
    license='MIT',
    entry_points={
        'console_scripts': [
            'monitoring_node = monitoring_node.monitoring_node:main',
        ],
    },
)
