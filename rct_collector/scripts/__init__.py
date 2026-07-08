"""Non-node library modules for rct_collector.

These are plain Python modules (config/pose sampling, the orchestrator, the
footprint collision checker, and the CLI entry point) — none of them create an
rclpy node. They live in this top-level ``scripts/`` folder so the package dir
(``rct_collector/``) holds only the ROS node (trial_runner.py).

setup.py maps this folder into the package namespace via ``package_dir``, so in
a built/installed workspace they import as ``rct_collector.scripts.<module>``.
"""
