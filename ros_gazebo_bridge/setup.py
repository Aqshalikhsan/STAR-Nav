from setuptools import find_packages, setup

package_name = "ros_gazebo_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/corridor_sim.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aqshal Nur Ikhsan",
    maintainer_email="aksalikhsan@gmail.com",
    description=(
        "Gazebo + MAVLink (PX4) + ROS 2 backend implementing STAR-Nav's "
        "BaseCorridorEnv contract."
    ),
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bridge_node = ros_gazebo_bridge.ros_bridge_node:main",
            "generate_world = ros_gazebo_bridge.world_gen:main",
            "manual_control = ros_gazebo_bridge.manual_control:main",
            "keyboard_control = ros_gazebo_bridge.keyboard_control:main",
            "vio_bridge = ros_gazebo_bridge.vio_bridge:main",
        ],
    },
)
