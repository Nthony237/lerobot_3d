# YAM URDF adapter and existing mesh calibration

This addition maps saved YAM feedback to the original I2RT URDF, renders it in
Viser, and exports the robot mesh cloud consumed by the existing `icp.py` workflow.
The calibration solver, manual alignment, confirmation, segmentation and SO101
teleoperation behavior are unchanged. Camera acquisition stays outside this PR.
Keep the PR draft until Sergio and Nandika validate it end to end together.

## Model and state

Use the original composed YAM v1 / linear_4310 station URDF and its mesh directory:

```bash
git clone https://github.com/i2rt-robotics/i2rt /tmp/i2rt-yam
git -C /tmp/i2rt-yam checkout 120c3c81400171174604e503943f8d1ebc891058
export YAM_URDF=/tmp/i2rt-yam/i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf
python examples/yam_viser.py view --urdf "$YAM_URDF" --state pose.json
```

Open `http://127.0.0.1:8080`. Example `pose.json` (synthetic values, not feedback):

```json
{"mode":"synthetic","arms":{
  "left":{"position_rad":[0,1.5,1.5,0,0,0],"gripper_open":0.5},
  "right":{"position_rad":[0,1.5,1.5,0,0,0],"gripper_open":0.5}
}}
```

Actual captures use `mode: "live"`. Six arm angles are radians in vendor joint
order; normalized gripper opening is 0=closed, 1=open. The station has two
positive-travel fingers per arm; the standalone arm's different convention is
not supported. Confirm installed hardware, joint signs and gripper mapping.

The station's base/camera placements are nominal CAD. Verify the actual relative
base pose before fitting both arms in a common world. This adapter preserves the
full base rotation/translation supplied in the URDF; it does not estimate them.

## Feed the existing calibration

For a stationary, paired joint/camera capture, add
`stationary_capture_confirmed: true` to the measured state only after verifying
that pairing. This is an operator assertion, not a timing measurement. Export:

```bash
python examples/yam_viser.py export-mesh --urdf "$YAM_URDF" --state pose.json \
  --output calibration_files/robot_pcd.npz
```

This exports both arms in the URDF world to the existing `pcd` array format.
Supply the matching `calibration_files/<serial>/{color.png,depth.npz,mask.png}`,
`intrinsic_calibration.json` and initial `extrinsic_calibration.json`, then follow
[Performing calibration](../README.md#performing-calibration). `icp.main(viewer)`
also works offline without starting its subsequent local-camera preview.

**Preserve the existing input conventions:** `depth.npz` is divided by 1000 in
`icp.py`, so export depth in millimetres, not unconverted D405 counts. Register
and rectify color/depth to the same pinhole grid described by the intrinsic JSON;
`mask.png` uses its alpha channel. D405 depth units must be read from the device.
These conversions belong in the acquisition/export adapter, not the ICP solver.

The existing result is `X_WC` (camera coordinates to URDF world) at that capture.
For a wrist camera, the acquisition adapter must store
`T_gripper_camera = inverse(T_world_gripper(q_capture)) @ X_WC` and recompute
`X_WC(q) = T_world_gripper(q) @ T_gripper_camera` for later frames. An overhead
camera stays fixed. Do not apply the captured wrist pose as a fixed extrinsic.

Review several independent physical poses and all camera overlaps before accepting
any calibration. Verify joint mapping and base geometry first; camera fitting can
otherwise conceal those errors. Low ICP residual alone does not prove alignment.
This PR does not claim physical calibration or live motion validation.

```bash
I2RT_CHECKOUT=/tmp/i2rt-yam pytest tests/hardware_stack/test_yam.py tests/hardware_stack/test_icp.py tests/hardware_stack/test_viser_viewer.py tests/pure -q
```
