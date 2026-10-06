"""``TeleopSystemConfig``: everything needed to run the teleop + point cloud system.

Values live in ``teleop_config.yaml``, not here -- see :func:`load_teleop_system_config`.
Edit that file (or point ``--config`` / ``LEROBOT_3D_TELEOP_CONFIG`` at your own copy)
to change hardware wiring or run settings; no code changes needed.

``leaders``, ``followers``, and ``realsense_serials`` may each be empty:

- no leaders: actions come only from ``TeleopPointCloudSystem.step(action)``.
- no followers: nothing is commanded; the action poses the URDF in viser (digital twin).
- no leaders *and* no followers: ``num_robots`` virtual robots, each posed by its own
  ``step(actions)`` entry and laid out on a viser grid ``robot_grid_spacing`` apart.
- no cameras: no RealSense streams; the fused scene cloud is empty.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from lerobot_3d.paths import resolve_teleop_config_yaml

DEFAULT_TELEOP_CONFIG_YAML = "teleop_config.yaml"
"""Default filename for the editable system config; resolved via
:func:`lerobot_3d.paths.resolve_teleop_config_yaml`."""


@dataclass(frozen=True)
class SO101AxisConfig:
    """One SO101 arm on a serial device (leader teleop or follower robot)."""

    port: str
    """Device path, e.g. ``/dev/ttyACM0``."""
    id: str
    """LeRobot calibration / bus id, e.g. ``bender_leader_arm``."""


def _axis_configs(entries: Sequence[Mapping[str, Any]], *, key: str) -> tuple[SO101AxisConfig, ...]:
    axes = []
    for i, entry in enumerate(entries):
        try:
            axes.append(SO101AxisConfig(port=str(entry["port"]), id=str(entry["id"])))
        except KeyError as e:
            raise ValueError(f"{DEFAULT_TELEOP_CONFIG_YAML}: {key}[{i}] missing required field {e}") from e
    return tuple(axes)


def _validate_axis_sets(
    leaders: Sequence[SO101AxisConfig],
    followers: Sequence[SO101AxisConfig],
    realsense_serials: Sequence[str],
    robot_calibration_ids: Sequence[str],
    robot_calibration_paths: Sequence[str | Path] | None = None,
    num_robots: int = 1,
) -> None:
    if leaders and followers and len(leaders) != len(followers):
        raise ValueError(
            f"Leaders ({len(leaders)}) and followers ({len(followers)}) must be the same count."
        )
    if num_robots > 1 and (leaders or followers):
        raise ValueError(
            "num_robots > 1 is only supported with no leaders and no followers "
            "(virtual robots driven by step(actions))."
        )
    if followers:
        if len(robot_calibration_ids) != len(followers):
            raise ValueError(
                "robot_calibration_ids must have one entry per follower "
                f"(got {len(robot_calibration_ids)} ids for {len(followers)} followers)."
            )
        if robot_calibration_paths is not None and len(robot_calibration_paths) != len(followers):
            raise ValueError(
                "robot_calibration_paths must have one entry per follower "
                f"(got {len(robot_calibration_paths)} paths for {len(followers)} followers)."
            )
        return
    # Without followers, num_robots virtual robots are visualized from the URDF; one
    # calibration entry is shared by all of them, or there is one per robot.
    allowed = {1, num_robots}
    if len(robot_calibration_ids) not in allowed:
        raise ValueError(
            "robot_calibration_ids must have 1 entry (shared) or one per robot "
            f"(got {len(robot_calibration_ids)} ids for num_robots={num_robots})."
        )
    if robot_calibration_paths is not None and len(robot_calibration_paths) not in allowed:
        raise ValueError(
            "robot_calibration_paths must have 1 entry (shared) or one per robot "
            f"(got {len(robot_calibration_paths)} paths for num_robots={num_robots})."
        )


@dataclass(frozen=True)
class TeleopSystemConfig:
    """Everything needed to construct :class:`TeleopPointCloudSystem` / :class:`SystemStateViewer`.

    Construct this via :func:`load_teleop_system_config` in normal use; build it directly only
    for scripts/tests that don't want to touch a YAML file.

    ``leaders``, ``followers``, and ``realsense_serials`` may each be empty (see the module
    docstring). With no followers, the URDF calibration comes from ``robot_calibration_ids`` /
    ``robot_calibration_paths`` if set, else the first leader's LeRobot calibration.
    """

    leaders: tuple[SO101AxisConfig, ...]
    followers: tuple[SO101AxisConfig, ...]
    realsense_serials: tuple[str, ...]
    extrinsic_json: str = "extrinsic_calibration.json"
    urdf_path: str | None = None
    """``None`` → bundled ``so101_new_calib.urdf`` under package ``calibration/``."""
    robot_calibration_ids: tuple[str, ...] | None = None
    """HF / LeRobot calibration name per follower; ``None`` → each follower's ``id`` (or, with
    no followers, the first leader's ``id``). With no followers, one entry is broadcast to all
    ``num_robots``."""
    robot_calibration_dir: str | Path | None = None
    """Directory containing ``<robot_calibration_id>.json``; ``None`` uses LeRobot defaults."""
    robot_calibration_paths: tuple[str | Path, ...] | None = None
    """Explicit calibration JSON path per follower; overrides ``robot_calibration_dir``."""
    tune: bool = True
    """If true, show the viser GUI's Quit/Capture/Save-subgoal controls."""
    point_size: float = 0.003
    """Viser point radius in world-space meters (not pixels)."""
    camera_width: int = 848
    camera_height: int = 480
    camera_fps: int = 60
    action_interpolation_duration_s: float = 0.12
    """Seconds to blend from the current command to a new target. ``0`` disables smoothing."""
    action_command_hz: float = 50.0
    """Follower command loop rate when action interpolation is enabled."""
    viser_port: int = 8080
    """Port for the viser server hosting the scene + robot point clouds and GUI controls."""
    num_robots: int = 1
    """Virtual robots driven by ``step(actions)``; ``> 1`` only with no leaders and no followers."""
    robot_grid_spacing: float = 0.5
    """Meters between neighboring robot bases in the viser grid (display only)."""
    dataset_repo_id: str = ""
    """Non-empty → record teleop episodes into this ``LeRobotDataset`` (e.g. ``local/my_task``).
    Enter starts an episode, Space ends it, then y/n keeps or discards it."""
    dataset_root: str | None = None
    """Dataset directory; ``None`` → LeRobot's default ``HF_LEROBOT_HOME/<dataset_repo_id>``.
    An existing dataset there is appended to."""
    dataset_task: str = "teleop"
    """Task string stored with every recorded frame."""
    dataset_fps: int = 15
    """Recording rate; the teleop loop runs at this rate while a dataset is configured."""
    segment_on_capture: bool = True
    """After viser **Capture**, open each camera's image to click-segment the robot with SAM2
    and write ``calibration_files/<serial>/mask.png`` (needs the ``segment`` extra)."""
    sam2_model_id: str = "facebook/sam2.1-hiera-large"
    """Hugging Face SAM2 checkpoint used for click-to-segment."""
    calibration_robot_type: str = dataclasses.field(default="so101_follower", init=False)
    """LeRobot device type whose calibration poses the URDF: ``"so101_leader"`` when there are
    no followers and ``robot_calibration_ids`` was defaulted from a leader. Derived, not set."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "realsense_serials", tuple(self.realsense_serials))
        object.__setattr__(self, "leaders", tuple(self.leaders))
        object.__setattr__(self, "followers", tuple(self.followers))
        if self.camera_width <= 0 or self.camera_height <= 0 or self.camera_fps <= 0:
            raise ValueError("camera_width, camera_height, and camera_fps must be positive.")
        if self.action_interpolation_duration_s < 0:
            raise ValueError("action_interpolation_duration_s must be >= 0.")
        if self.action_command_hz <= 0:
            raise ValueError("action_command_hz must be positive.")
        if self.viser_port <= 0:
            raise ValueError("viser_port must be positive.")
        if self.num_robots < 1:
            raise ValueError("num_robots must be >= 1.")
        if self.robot_grid_spacing <= 0:
            raise ValueError("robot_grid_spacing must be positive.")
        if self.dataset_fps <= 0:
            raise ValueError("dataset_fps must be positive.")
        if self.dataset_repo_id and self.num_robots > 1:
            raise ValueError("Dataset recording supports a single robot (num_robots must be 1).")
        if self.robot_calibration_ids is None:
            if self.followers:
                ids = tuple(f.id for f in self.followers)
            elif self.leaders:
                ids = (self.leaders[0].id,)
                object.__setattr__(self, "calibration_robot_type", "so101_leader")
            elif self.robot_calibration_paths is not None:
                ids = ("",)
            else:
                raise ValueError(
                    "With no followers or leaders, set robot_calibration_ids or "
                    "robot_calibration_paths so the URDF can be posed."
                )
            object.__setattr__(self, "robot_calibration_ids", ids)
        else:
            object.__setattr__(self, "robot_calibration_ids", tuple(self.robot_calibration_ids))
        if self.robot_calibration_paths is not None:
            object.__setattr__(
                self,
                "robot_calibration_paths",
                tuple(Path(p).expanduser() for p in self.robot_calibration_paths),
            )
        _validate_axis_sets(
            self.leaders,
            self.followers,
            self.realsense_serials,
            self.robot_calibration_ids,
            self.robot_calibration_paths,
            self.num_robots,
        )
        # Broadcast a single shared calibration entry so every virtual robot can index [i].
        if not self.followers and self.num_robots > 1:
            if len(self.robot_calibration_ids) == 1:
                object.__setattr__(
                    self, "robot_calibration_ids", self.robot_calibration_ids * self.num_robots
                )
            if self.robot_calibration_paths is not None and len(self.robot_calibration_paths) == 1:
                object.__setattr__(
                    self, "robot_calibration_paths", self.robot_calibration_paths * self.num_robots
                )


_SPECIAL_FIELDS = {"leaders", "followers", "realsense_serials", "calibration_robot_type"}


def load_teleop_system_config(path: str | Path | None = None) -> TeleopSystemConfig:
    """Load a full :class:`TeleopSystemConfig` from ``teleop_config.yaml``.

    Uses the same search order as camera extrinsics (env var, cwd, package-adjacent ``src/``);
    see :func:`lerobot_3d.paths.resolve_teleop_config_yaml`. Pass ``None`` (or omit) to
    use the default filename, :data:`DEFAULT_TELEOP_CONFIG_YAML`.
    """
    resolved = resolve_teleop_config_yaml(DEFAULT_TELEOP_CONFIG_YAML if path is None else path)
    with open(resolved) as f:
        data = yaml.safe_load(f) or {}

    known_fields = {f.name for f in dataclasses.fields(TeleopSystemConfig)} - _SPECIAL_FIELDS
    kwargs = {k: v for k, v in data.items() if k in known_fields and v is not None}

    return TeleopSystemConfig(
        leaders=_axis_configs(data.get("leaders") or [], key="leaders"),
        followers=_axis_configs(data.get("followers") or [], key="followers"),
        realsense_serials=tuple(str(s) for s in data.get("realsense_serials") or []),
        **kwargs,
    )
