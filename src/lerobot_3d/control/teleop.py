from __future__ import annotations

import argparse
import time

import numpy as np
import open3d as o3d

from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot_3d.common.types import Datapoint, RobotSnapshot
from lerobot_3d.teleop_config import TeleopSystemConfig, load_teleop_system_config
from lerobot_3d.point_clouds.system_vis import SystemStateViewer


class TeleopPointCloudSystem:

    def __init__(self, config: TeleopSystemConfig):
        self.config = config
        self.leaders = [
            SO101Leader(SO101LeaderConfig(port=ax.port, id=ax.id)) for ax in config.leaders
        ]
        self.viewer = SystemStateViewer(config)

    def connect(self) -> None:
        print("Connecting devices...")
        for leader in self.leaders:
            leader.connect()
        print("Connected.")

    def step(
        self, action=None, masks_by_serial=None
    ) -> tuple[
        list[Datapoint],
        o3d.geometry.PointCloud,
        list[np.ndarray],
        list[dict[str, np.ndarray]],
        list[RobotSnapshot],
    ]:
        """One control cycle: read leaders (or use ``action``), update followers and viewer.

        Args:
            action: Optional per-follower action list (one motor-space dict per
                follower, positionally aligned with ``config.followers`` -- same
                shape ``SystemStateViewer.update()`` already expects). When given,
                it's sent to the followers instead of the leaders' teleop action,
                letting a caller (e.g. a computed IK target) drive the robot directly.
                Omit to teleop from the leaders. Required when no leaders are
                configured. With no leaders and no followers, pass one action per
                virtual robot (``config.num_robots``); each poses its own URDF in the
                viser grid instead of being sent to hardware.
            masks_by_serial: Optional mapping ``{camera_serial: mask}`` or sequence of
                masks aligned with ``config.realsense_serials``. Nonzero/True mask
                pixels are kept in the fused scene point cloud.

        Returns:
            datapoints: Raw per-camera datapoints used to build the fused point cloud.
            scene_pcd: Open3D fused scene point cloud in world frame, including colors.
            robot_pcds: per robot, ``(M, 3)`` ``float64`` sampled mesh points in that robot's
                base frame (with followers: one entry, the first follower).
            robot_link_pcds: per robot, clouds keyed by URDF link name (base frame).
            robot_snapshots: per robot, a :class:`RobotSnapshot` -- motor-space
                ``joint_positions``, ``joint_radians``, ``pcd``, ``link_pcds``,
                ``link_poses`` (base frame) and the viser grid ``base_offset``.

        Poll ``self.viewer.quit`` to know when to stop the outer loop, then call :meth:`close`.
        """
        if action is not None:
            actions = action
        elif not self.leaders:
            raise ValueError(
                "No leaders configured; call step(action) with one action dict per follower "
                "(or one per virtual robot, config.num_robots, with no followers)."
            )
        elif self.config.followers:
            actions = [leader.get_action() for leader in self.leaders]
        else:
            # URDF-only mode visualizes a single virtual robot, driven by the first leader.
            actions = [self.leaders[0].get_action()]
        print(actions)
        return self.viewer.update(*actions, masks_by_serial=masks_by_serial)

    def close(self) -> None:
        """Release cameras, robots, and recording resources."""
        self.viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleop + point cloud viewer.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Teleop system config YAML (cwd, LEROBOT_3D_TELEOP_CONFIG, or src/ next to "
        "package). Omit to use the default teleop_config.yaml.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=60.0,
        help="Main loop rate in Hz (default 60). Use 0 or negative for no sleep (full speed).",
    )

    args = parser.parse_args()

    config = load_teleop_system_config(args.config)
    if not config.leaders:
        parser.exit(
            1,
            "No leaders configured: the CLI loop only teleops from leaders. Drive the system "
            "from your own script with TeleopPointCloudSystem.step(action) instead.\n",
        )
    system = TeleopPointCloudSystem(config)

    system.connect()

    period_s = None if args.hz is None or args.hz <= 0 else 1.0 / args.hz
    try:
        while not system.viewer.quit:
            t_iter_start = time.monotonic()
            _datapoints, _scene_pcd, _robot_pcds, _robot_link_pcds, _snapshots = system.step()

            if period_s is not None:
                elapsed = time.monotonic() - t_iter_start
                time.sleep(max(0.0, period_s - elapsed))
    finally:
        system.close()


if __name__ == "__main__":
    main()
