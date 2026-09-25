"""Pure helpers of calibration_mask (no SAM2, no GUI windows)."""
import numpy as np
import pytest

pytest.importorskip("cv2")
PIL_Image = pytest.importorskip("PIL.Image")

from lerobot_3d.calibration_mask import PromptPoints, save_mask_png

pytestmark = pytest.mark.hardware_stack


def test_prompt_points_add_undo_reset():
    prompts = PromptPoints()
    prompts.add(10, 20, positive=True)
    prompts.add(30, 40, positive=False)

    coords, labels = prompts.as_arrays()
    assert coords.dtype == np.float32
    assert coords.tolist() == [[10, 20], [30, 40]]
    assert labels.tolist() == [1, 0]

    prompts.undo()
    assert prompts.as_arrays()[1].tolist() == [1]

    prompts.reset()
    coords, labels = prompts.as_arrays()
    assert coords.shape == (0, 2)
    assert len(prompts) == 0


def test_prompt_points_undo_empty_is_noop():
    prompts = PromptPoints()
    prompts.undo()
    assert len(prompts) == 0


def test_save_mask_png_alpha_matches_icp_convention(tmp_path):
    color = np.random.default_rng(0).integers(0, 255, (6, 8, 3), dtype=np.uint8)
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:6] = True
    path = tmp_path / "mask.png"

    save_mask_png(path, color, mask)

    # Exactly how icp.py reads it.
    rgba = np.array(PIL_Image.open(path))
    alpha = rgba[..., 3]
    assert np.array_equal(alpha == 255, mask)
    assert np.array_equal(alpha[~mask], np.zeros((~mask).sum(), dtype=np.uint8))
    assert np.array_equal(rgba[..., :3], color[..., ::-1])


def test_save_mask_png_shape_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        save_mask_png(tmp_path / "m.png", np.zeros((4, 4, 3), np.uint8), np.zeros((3, 4), bool))
