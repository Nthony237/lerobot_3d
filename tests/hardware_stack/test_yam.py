"""Offline YAM URDF, Viser and mesh calibration checks; no motor interface."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("urchin")
pytest.importorskip("open3d")
pytest.importorskip("viser")

from lerobot_3d.common.types import RobotSnapshot
from lerobot_3d.point_clouds.viser_viewer import ViserSceneViewer
from lerobot_3d.point_clouds.yam_calibration import camera_pose, fit_mesh, mesh_residual
from lerobot_3d.point_clouds.yam_robot_state import YamRobotState, transform_points

pytestmark = pytest.mark.hardware_stack


@pytest.fixture
def station():
    checkout = os.environ.get("I2RT_CHECKOUT")
    if not checkout:
        pytest.skip("Set I2RT_CHECKOUT to the pinned vendor checkout for real URDF tests")
    path = (
        Path(checkout)
        / "i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf"
    )
    return {side: YamRobotState(path, side) for side in ("left", "right")}


@pytest.fixture
def observation():
    return {"position_rad": [0.2, 1.2, 1.5, 0.1, -0.2, 0.3], "gripper_open": 0.4}


def test_station_mesh_poses_match_direct_urdf_fk(station, observation):
    for side, state in station.items():
        joints = state.get_joint_positions(observation)
        assert [joints[f"{side}_joint{i}"] for i in range(1, 7)] == observation["position_rad"]
        assert joints[f"{side}_joint7"] == pytest.approx(0.4 * 0.04695)
        assert joints[f"{side}_joint8"] == pytest.approx(0.4 * 0.04695)
        snapshot = state.get_robot_snapshot(observation)
        base = state.get_link_transform(observation, side + "_base")
        world = state.robot_urdf.link_fk(cfg=joints, use_names=True)
        for name, (translation, quaternion) in snapshot.link_poses.items():
            local = np.eye(4)
            local[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
            local[:3, 3] = translation
            np.testing.assert_allclose(base @ local, world[name], atol=1e-10)
        dense = state.get_mesh_points(observation)
        assert len(dense) > len(snapshot.pcd) and np.isfinite(dense).all()
        changed = {**observation, "position_rad": [0.4, *observation["position_rad"][1:]]}
        moved = state.get_robot_snapshot(changed)
        assert not np.allclose(snapshot.pcd, moved.pcd)
        assert len(state.get_static_meshes()) >= 8


def test_bad_feedback_is_rejected(station, observation):
    state = station["left"]
    for bad in ([0] * 5, [np.nan] * 6, [20] * 6):
        with pytest.raises(ValueError):
            state.get_robot_snapshot({**observation, "position_rad": bad})
    with pytest.raises(ValueError):
        state.get_robot_snapshot({**observation, "gripper_open": 1.1})


def test_wrist_calibration_follows_new_pose_and_fixed_camera_does_not(station, observation):
    gripper_camera = np.eye(4)
    gripper_camera[:3, 3] = [0.03, -0.04, 0.07]
    candidate = {"camera": "right", "T_gripper_camera": gripper_camera.tolist()}
    poses = {side: observation for side in station}
    original = camera_pose(candidate, station, poses)
    changed = {**observation, "position_rad": [0.7, *observation["position_rad"][1:]]}
    poses["right"] = changed
    moved = camera_pose(candidate, station, poses)
    assert not np.allclose(original, moved)
    expected = station["right"].get_link_transform(changed, "right_gripper") @ gripper_camera
    np.testing.assert_allclose(moved, expected)
    np.testing.assert_array_equal(
        camera_pose({"camera": "overhead", "X_WC": original.tolist()}, station, poses), original
    )


def test_icp_recovers_camera_transform_and_heldout_residual(station, observation):
    sampled = station["left"].get_mesh_points(observation)
    target, independent = sampled[::2], sampled[1::2]
    world_camera = np.eye(4)
    world_camera[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.1]).as_matrix()
    world_camera[:3, 3] = [0.2, -0.1, 0.5]
    source = transform_points(np.linalg.inv(world_camera), independent)
    initial = world_camera.copy()
    initial[:3, 3] += [0.001, -0.001, 0.001]
    result = fit_mesh(source, target, initial)
    np.testing.assert_allclose(result["X_WC"], world_camera, atol=0.002)
    moved = {**observation, "position_rad": [0.5, *observation["position_rad"][1:]]}
    heldout = station["left"].get_mesh_points(moved)
    heldout_source = transform_points(np.linalg.inv(world_camera), heldout)
    residual = mesh_residual(heldout_source, heldout, result["X_WC"])
    assert residual["p95_distance_m"] < 0.002
    wrong = world_camera.copy()
    wrong[0, 3] += 0.06
    assert mesh_residual(heldout_source, heldout, wrong)["p95_distance_m"] > 0.02


def test_viser_applies_base_rotation_and_so101_default_is_identity():
    viewer = ViserSceneViewer.__new__(ViserSceneViewer)
    frame = SimpleNamespace(position=None, wxyz=None)
    viewer._scene_handle = None
    viewer._robot_frames = {0: frame}
    viewer._robot_handles = {}
    viewer._upsert_point_cloud = lambda *args: None
    viewer.update_link_poses = lambda *args, **kwargs: None
    snapshot = RobotSnapshot(0, {}, {}, np.zeros((1, 3)), {}, {}, np.array([1.0, 2.0, 3.0]))
    assert snapshot.base_wxyz == (1, 0, 0, 0)
    snapshot.base_wxyz = tuple(Rotation.from_euler("z", 0.7).as_quat()[[3, 0, 1, 2]])
    viewer.update(np.empty((0, 3)), None, [snapshot])
    np.testing.assert_array_equal(frame.position, snapshot.base_offset)
    np.testing.assert_array_equal(frame.wxyz, snapshot.base_wxyz)
