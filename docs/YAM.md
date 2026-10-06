# YAM URDF and mesh calibration review

This is a small offline addition: render both YAMs from saved joint feedback in
Viser and calibrate masked camera clouds against their URDF meshes. It reuses
`RobotSnapshot`, `ViserSceneViewer`, `AlignmentViewer` and the existing multiscale
ICP solver. It does not connect to motors or cameras. Live Jetson acquisition
and control belong in a separate PR.

**Keep this PR unmerged until Sergio and Nandika validate it end to end.**
Synthetic checks do not validate the physical rig or replace that review.

## Model and saved state

Use I2RT's original composed YAM v1 / linear_4310 station URDF and its relative
mesh directory, unchanged. No model conversion or added dependency is needed:

```bash
pip install -e '.[test]'
git clone https://github.com/i2rt-robotics/i2rt /tmp/i2rt-yam
git -C /tmp/i2rt-yam checkout 120c3c81400171174604e503943f8d1ebc891058
export YAM_URDF=/tmp/i2rt-yam/i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf
python examples/yam_viser.py view --urdf "$YAM_URDF" --state pose.json
```

Open `http://127.0.0.1:8080`. `pose.json` has this shape; the values here are an
explicitly **synthetic example**, not measured feedback:

```json
{
  "mode": "synthetic",
  "arms": {
    "left": {"position_rad": [0, 1.5, 1.5, 0, 0, 0], "gripper_open": 0.5},
    "right": {"position_rad": [0, 1.5, 1.5, 0, 0, 0], "gripper_open": 0.5}
  }
}
```

Six arm joints use radians in vendor joint order. Gripper opening is normalized
0=closed, 1=open for the station's positive-travel linear_4310 fingers. The
original standalone arm URDF uses different gripper conventions and is not an
input to this adapter. Confirm installed hardware and signs before using real
feedback. `view` shows a saved snapshot, not a live stream.

The station's base and camera placements are vendor CAD, not measurements of
this lab rig. The adapter keeps their full rigid transforms, including base
rotation. Confirm or measure the relative arm-base pose before fitting both arms
in a common world. This PR does not estimate that pose or claim calibrated
camera-aligned URDF exports.

## Mesh-based calibration

For each stationary capture, prepare:

- The paired state JSON above with `mode: "live"` for actual measurements.
- `points.npy`: an N×3 NumPy array of **masked robot points**, in that camera's
  optical coordinates (+X right, +Y down, +Z forward), already in metres.
- `initial.json`: an approximate camera-to-URDF-world transform under `X_WC`.

Keep the robot, grippers and cameras still while capturing. The state and camera
cloud must belong to the same capture; after verifying that pairing, add
`stationary_capture_confirmed: true` to the state JSON. This is an operator
assertion, not an automatic timing/stationarity check. Remove table/background
points from the cloud. Use the captured depth intrinsics, distortion and depth
scale when deprojecting; if masking with RGB, register depth into the RGB frame
first. The saved XYZ interface deliberately leaves capture/deprojection outside
this PR. Never treat D405 raw depth counts as millimetres without checking scale.

```bash
python examples/yam_viser.py calibrate --urdf "$YAM_URDF" --state pose.json \
  --camera left --points left_points.npy --initial initial.json --output left.candidate.json
```

The Viser window shows the full URDF meshes and sampled target cloud. Manually
align the masked camera cloud, confirm, then inspect the existing ICP refinement.
At the second review, Abort keeps the manual alignment. The output always has
`status: "candidate"`; fit quality is not acceptance. Files are never overwritten.
Run independently for `--camera right` and `--camera overhead`.

For overhead, `X_WC` is fixed in URDF world. For a wrist camera, the output also
contains `T_gripper_camera = inverse(T_world_gripper) @ X_WC`; at another arm pose,
`camera_pose()` composes it with current measured FK. **Do not reuse the captured
wrist `X_WC` as a fixed world transform.** All transforms map child coordinates
into parent coordinates, with distances in metres.

Validate with a different saved pose/cloud, without refitting:

```bash
python examples/yam_viser.py validate --urdf "$YAM_URDF" --state heldout_pose.json \
  --camera left --points heldout_left_points.npy --candidate left.candidate.json \
  --output left.validation.json
```

The display moves the wrist camera using held-out FK. The report records median
and p95 distance to the sampled mesh, while preserving the original fit. Nearby
wrong links, partial views and symmetric geometry can produce misleadingly small
residuals; inspect several distinct poses for all three cameras. Captures, model
and candidate are hashed for traceability. Synthetic and real data cannot be
mixed in validation. No result automatically updates a live calibration file.

## Review session

Before merge, run the existing SO101 tests and check its viewer still behaves the
same; load the actual vendor URDF in Viser; check both arms' joint signs and
fingers against saved measured states; inspect the segmented clouds and frame
conventions; perform manual alignment and ICP for each camera; then inspect
held-out poses, including that wrist cameras move and overhead stays fixed.
Record failures and acceptance criteria together before enabling a live twin.

```bash
I2RT_CHECKOUT=/tmp/i2rt-yam pytest tests/hardware_stack/test_yam.py tests/hardware_stack/test_icp.py tests/hardware_stack/test_viser_viewer.py tests/pure -q
```

New tests check both vendor URDFs through the station, gripper mapping, invalid
feedback, base orientation in Viser, ICP recovery on a synthetic cloud and
held-out residuals. Model tests skip without `I2RT_CHECKOUT`. These are software
checks only; live hardware behavior and real calibration remain unvalidated.
