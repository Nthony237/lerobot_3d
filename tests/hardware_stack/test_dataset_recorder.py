"""End-to-end LeRobotDataset recording with synthetic frames (no cameras or robot needed):
record -> finalize -> reload -> rebuild Datapoints -> fuse, and append to an existing dataset.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("open3d")
pytest.importorskip("lerobot")
pytest.importorskip("cv2")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_3d.common.types import Datapoint, RobotSnapshot
from lerobot_3d.point_clouds.camera_stream import get_fused_point_cloud
from lerobot_3d.recording.dataset_loader import (
    frame_joint_positions,
    frame_link_pcds,
    frame_to_datapoints,
    read_sidecar,
)
from lerobot_3d.recording.dataset_recorder import LeRobotDatasetRecorder

pytestmark = pytest.mark.hardware_stack

H, W = 48, 64
SERIALS = ["111", "222"]
JOINTS = ["shoulder_pan.pos", "gripper.pos"]
REPO_ID = "local/test_recorder"


def _step_data(rng, t):
    datapoints = []
    for i, serial in enumerate(SERIALS):
        X_WC = np.eye(4)
        X_WC[:3, 3] = [0.1 * i, 0.2, 0.5]
        datapoints.append(
            Datapoint(
                serial=serial,
                color=rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
                depth=rng.integers(200, 3000, (H, W), dtype=np.uint16),
                depth_scale=0.001,
                max_depth=10.0,
                X_WC=X_WC,
                color_intrinsics=SimpleNamespace(fx=60.0 + i, fy=61.0, ppx=32.0, ppy=24.0),
            )
        )
    snapshot = RobotSnapshot(
        index=0,
        joint_positions={"shoulder_pan.pos": float(t), "gripper.pos": 50.0},
        joint_radians={},
        pcd=np.zeros((0, 3)),
        link_pcds={
            "base": rng.random((100, 3)),
            "gripper_link": rng.random((200, 3)),
        },
        link_poses={},
        base_offset=np.zeros(3),
    )
    action = {"shoulder_pan.pos": float(t) + 0.5, "gripper.pos": 40.0}
    return datapoints, snapshot, action


def _record(recorder, rng, n_frames, keep, log):
    recorder.start_episode()
    for t in range(n_frames):
        step = _step_data(rng, t)
        recorder.add_frame(*step)
        if keep:
            log.append(step)
    recorder.stop_episode()
    if keep:
        recorder.save_episode()
    else:
        recorder.discard_episode()


@pytest.fixture
def recorded(tmp_path):
    root = tmp_path / "ds"
    rng = np.random.default_rng(0)
    kept = []
    recorder = LeRobotDatasetRecorder(REPO_ID, root=root, fps=10, task="push")
    _record(recorder, rng, 4, keep=True, log=kept)
    _record(recorder, rng, 3, keep=False, log=kept)
    _record(recorder, rng, 5, keep=True, log=kept)
    recorder.finalize()
    return root, kept


def test_discarded_episode_is_not_saved(recorded):
    root, kept = recorded

    ds = LeRobotDataset(REPO_ID, root=root)

    assert ds.meta.total_episodes == 2
    assert len(ds) == len(kept) == 9


def test_frames_roundtrip(recorded):
    root, kept = recorded
    ds = LeRobotDataset(REPO_ID, root=root)
    joint_names = ds.meta.features["observation.state"]["names"]

    for i, (datapoints, snapshot, action) in enumerate(kept):
        item = ds[i]
        restored = frame_to_datapoints(item, joint_names=joint_names)

        assert [dp.serial for dp in restored] == SERIALS
        for orig, back in zip(datapoints, restored):
            assert np.array_equal(back.depth, orig.depth)  # lossless
            assert back.depth_scale == pytest.approx(orig.depth_scale)
            assert np.allclose(back.X_WC, orig.X_WC)
            assert back.color_intrinsics.fx == pytest.approx(orig.color_intrinsics.fx)
            assert back.color_intrinsics.ppy == pytest.approx(orig.color_intrinsics.ppy)
            assert back.color.shape == orig.color.shape and back.color.dtype == np.uint8
            assert back.joint_positions == snapshot.joint_positions

        links = frame_link_pcds(item)
        assert set(links) == set(snapshot.link_pcds)
        for name, pts in snapshot.link_pcds.items():
            assert np.allclose(links[name], pts, atol=1e-6)

        assert np.allclose(item["action"].numpy(), [action[k] for k in JOINTS])
        assert frame_joint_positions(item, joint_names) == snapshot.joint_positions
        assert item["task"] == "push"


def test_fused_cloud_from_dataset_matches_live(recorded):
    root, kept = recorded
    ds = LeRobotDataset(REPO_ID, root=root)
    datapoints, _, _ = kept[0]

    live, _ = get_fused_point_cloud(datapoints)
    replay, _ = get_fused_point_cloud(frame_to_datapoints(ds[0]))

    # float32 storage of depth_scale/X_WC: sub-micron differences only.
    assert np.allclose(np.asarray(live.points), np.asarray(replay.points), atol=1e-6)


def test_sidecar_describes_dataset(recorded):
    root, _ = recorded

    sidecar = read_sidecar(root)

    assert sidecar["depth_encoding"] == "uint16_packed_rg8"
    assert sidecar["camera_serials"] == SERIALS
    assert sidecar["robot_links"] == ["base", "gripper_link"]
    assert sidecar["joint_names"] == JOINTS


def test_reopening_appends(recorded):
    root, kept = recorded
    rng = np.random.default_rng(1)
    extra = []

    recorder = LeRobotDatasetRecorder(REPO_ID, root=root, fps=10, task="push")
    assert recorder.num_saved_episodes == 2
    _record(recorder, rng, 2, keep=True, log=extra)
    recorder.finalize()

    ds = LeRobotDataset(REPO_ID, root=root)
    assert ds.meta.total_episodes == 3
    assert len(ds) == len(kept) + 2
    assert np.array_equal(frame_to_datapoints(ds[len(ds) - 1])[0].depth, extra[-1][0][0].depth)


def test_reopening_with_different_fps_raises(recorded):
    root, _ = recorded

    with pytest.raises(ValueError, match="fps"):
        LeRobotDatasetRecorder(REPO_ID, root=root, fps=30)


def test_mismatched_frame_shape_raises(recorded):
    root, _ = recorded
    recorder = LeRobotDatasetRecorder(REPO_ID, root=root, fps=10)
    datapoints, snapshot, action = _step_data(np.random.default_rng(2), 0)
    snapshot.link_pcds["base"] = np.zeros((7, 3))

    recorder.start_episode()
    with pytest.raises(ValueError, match="doesn't match"):
        recorder.add_frame(datapoints, snapshot, action)
    recorder.finalize()
