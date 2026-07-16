from setuptools import find_packages, setup
import os
from glob import glob

package_name = "rct_collector"

setup(
    name=package_name,
    version="1.0.0",
    # rct_collector/ holds the ROS nodes (trial_runner.py, environment_risk_node.py).
    # The non-node library modules live in the genuine nested subpackage
    # rct_collector/scripts/, so find_packages discovers it automatically and the
    # layout works under both a copy build and `colcon build --symlink-install`.
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/default_config.yaml"]),
        ("share/" + package_name + "/launch", ["launch/rct_collection.launch.py"]),
        (os.path.join('share', package_name, 'behavior_trees'), glob('behavior_trees/*.xml')),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Forough",
    maintainer_email="your@email.com",
    description="RCT data collector for Nav2 parameter tuning on PAL TIAGo",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Orchestrator CLI — the program you run to collect data. It drives
            # the TrialRunner node internally, one trial at a time.
            "rct_collect = rct_collector.scripts.main:main",
            # Standalone risk-state node; publishes /risk_state at 10 Hz. Run it
            # alongside Nav2 when collecting risk features:
            #   ros2 run rct_collector risk_state_node
            "risk_state_node = rct_collector.environment_risk_node:main",
            # Standalone footprint measurement utility; run it once per arm pose    
            "measure_footprint = rct_collector.measure_footprint:main"
        ],
    },
)
