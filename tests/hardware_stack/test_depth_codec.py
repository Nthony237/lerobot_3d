import numpy as np
import pytest

from lerobot_3d.recording.depth_codec import pack_depth, unpack_depth

pytestmark = pytest.mark.hardware_stack


def _full_range_depth():
    return np.arange(65536, dtype=np.uint16).reshape(256, 256)


def test_roundtrip_uint8_hwc_is_exact():
    depth = _full_range_depth()

    packed = pack_depth(depth)

    assert packed.dtype == np.uint8
    assert packed.shape == (256, 256, 3)
    assert (packed[..., 2] == 0).all()
    assert np.array_equal(unpack_depth(packed), depth)


def test_roundtrip_float_chw_as_loaded_by_lerobot_is_exact():
    depth = _full_range_depth()
    loaded = np.moveaxis(pack_depth(depth), -1, 0).astype(np.float32) / 255.0

    assert np.array_equal(unpack_depth(loaded), depth)


def test_roundtrip_torch_tensor():
    torch = pytest.importorskip("torch")
    depth = _full_range_depth()
    loaded = torch.from_numpy(np.moveaxis(pack_depth(depth), -1, 0).copy()).float() / 255.0

    assert np.array_equal(unpack_depth(loaded), depth)


def test_pack_rejects_non_uint16():
    with pytest.raises(ValueError, match="uint16"):
        pack_depth(np.zeros((2, 2), dtype=np.float32))


def test_pack_rejects_wrong_ndim():
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        pack_depth(np.zeros((2, 2, 1), dtype=np.uint16))
