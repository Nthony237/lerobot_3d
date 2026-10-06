"""Preview a saved YAM pose or calibrate a masked camera cloud against its URDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_3d.point_clouds.alignment_viewer import AlignmentViewer
from lerobot_3d.point_clouds.viser_viewer import ViserSceneViewer
from lerobot_3d.point_clouds.yam_calibration import (
    camera_pose,
    interactive_fit,
    mesh_residual,
    rigid_transform,
)
from lerobot_3d.point_clouds.yam_robot_state import YamRobotState, transform_points


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_station(server):
    @server.on_client_connect
    def connected(client):
        client.camera.position = (1.6, 0.8, 1.2)
        client.camera.look_at = (0.15, -0.3, 0.35)


def add_meshes(server, states, observations):
    """Show the same URDF visuals beside the alignment viewer's camera cloud."""
    for side, state in states.items():
        poses = state.robot_urdf.link_fk(cfg=state.get_joint_positions(observations[side]), use_names=True)
        for link, name, vertices, faces in state.get_static_meshes():
            matrix = poses[link]
            server.scene.add_mesh_simple(
                f"/urdf/{side}/{link}/{name}",
                vertices=vertices,
                faces=faces,
                position=matrix[:3, 3],
                wxyz=Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]],
                color=(160, 190, 210),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("view", "calibrate", "validate"))
    parser.add_argument("--urdf", required=True, help="I2RT composed YAM v1 / linear_4310 station URDF")
    parser.add_argument("--state", required=True, help="Saved paired state JSON; mode + arms")
    parser.add_argument("--camera", choices=("left", "right", "overhead"))
    parser.add_argument("--points", help="Masked camera-optical XYZ array (.npy), already in metres")
    parser.add_argument("--initial", help="JSON containing initial X_WC for manual alignment")
    parser.add_argument("--candidate", help="Candidate JSON from a different calibration capture")
    parser.add_argument("--output", help="New candidate/validation report JSON")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    metadata = load(args.state)
    if metadata.get("mode") not in ("live", "synthetic"):
        parser.error("State mode must be live (saved measurements) or synthetic (offline example)")
    observations = metadata["arms"]
    if set(observations) != {"left", "right"}:
        parser.error("Saved state must include both arms")
    states = {side: YamRobotState(args.urdf, side) for side in observations}
    snapshots = [
        state.get_robot_snapshot(observations[side], index=i)
        for i, (side, state) in enumerate(states.items())
    ]
    if args.mode == "view":
        viewer = ViserSceneViewer(port=args.port, host="127.0.0.1", controls=False)
        frame_station(viewer.server)
        try:
            for state, snapshot in zip(states.values(), snapshots):
                viewer.load_static_meshes(state.get_static_meshes(), snapshot.index, snapshot.base_offset)
            viewer.update(np.empty((0, 3)), None, snapshots)
            viewer.set_status(f"Saved {metadata['mode']} snapshot; no live feedback or motor commands")
            print("Ctrl-C to close the saved-pose preview.")
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()
        return
    if not args.camera or not args.points or not args.output:
        parser.error("Calibration/validation requires --camera, --points and --output")
    if Path(args.output).exists():
        parser.error("Refusing to overwrite an existing report")
    if args.mode == "calibrate" and not args.initial or args.mode == "validate" and not args.candidate:
        parser.error("Use --initial for calibrate, or --candidate for validate")
    if metadata["mode"] == "live" and not metadata.get("stationary_capture_confirmed"):
        parser.error("Confirm the saved camera points and joint states belong to one stationary capture")
    points = np.load(args.points, allow_pickle=False)
    target = np.concatenate([state.get_mesh_points(observations[side]) for side, state in states.items()])
    provenance = {
        "urdf_sha256": digest(args.urdf),
        "state_sha256": digest(args.state),
        "points_sha256": digest(args.points),
    }
    viewer = AlignmentViewer(port=args.port, host="127.0.0.1")
    frame_station(viewer.server)
    try:
        viewer.server.gui.add_markdown(
            f"**Saved {metadata['mode']} capture — inspect URDF and masked camera cloud.**"
        )
        add_meshes(viewer.server, states, observations)
        if args.mode == "calibrate":
            result = interactive_fit(viewer, points, target, load(args.initial)["X_WC"])
            result.update(camera=args.camera, source_mode=metadata["mode"], **provenance)
            if args.camera != "overhead":
                world_gripper = states[args.camera].get_link_transform(
                    observations[args.camera], args.camera + "_gripper"
                )
                result["T_gripper_camera"] = (
                    np.linalg.inv(world_gripper) @ rigid_transform(result["X_WC"])
                ).tolist()
        else:
            candidate = load(args.candidate)
            if candidate["camera"] != args.camera or candidate["urdf_sha256"] != provenance["urdf_sha256"]:
                raise ValueError("Camera or URDF differs from the candidate")
            if candidate["source_mode"] != metadata["mode"]:
                raise ValueError("Cannot validate a synthetic calibration with real data or vice versa")
            if candidate["points_sha256"] == provenance["points_sha256"]:
                raise ValueError("Validation must use a separate camera capture")
            matrix = camera_pose(candidate, states, observations)
            result = mesh_residual(points, target, matrix)
            result.update(
                camera=args.camera,
                source_mode=metadata["mode"],
                candidate_sha256=digest(args.candidate),
                **provenance,
            )
            viewer.show("/moving", transform_points(matrix, points), None)
            viewer.wait_for_confirmation("Inspect held-out alignment; this does not accept calibration")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Saved {output}; physical calibration acceptance remains a joint review.")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
