"""Testes do render.py (contrato IA-1 ↔ IA-2): geometria, determinismo e paridade de lote."""
import math

import cv2
import numpy as np
import pandas as pd
import torch

from dal.v3.sensor.render import TAN_HALF_FOV, RasterGeoref, batch_from_spec, render_crops, sample_geometry

GEOREF = RasterGeoref(x_origin=1000.0, y_origin=5000.0, res=1.0)


def _image(seed: int = 0) -> np.ndarray:
    """Imagem sintética com textura suave, para a interpolação ser comparável."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    return cv2.resize(small, (400, 400), interpolation=cv2.INTER_CUBIC)


def _spec(n: int = 6, **overrides) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    spec = pd.DataFrame({
        "crop_id": np.arange(n, dtype=np.int64) + 1000, "x": 1200 + rng.uniform(-20, 20, n),
        "y": 4800 + rng.uniform(-20, 20, n), "footprint_m": 60.0, "yaw_deg": rng.uniform(-5, 5, n),
        "roll_deg": 0.0, "pitch_deg": 0.0, "brightness": 0.05, "contrast": 1.1, "blur_sigma": 0.0,
        "noise_std": 3.0, "out_px": 32,
    })
    for column, value in overrides.items():
        spec[column] = value
    return spec


def _v1_affine_crop(image, x, y, footprint, out_px, theta):
    """Recorte da v1 (`dal.features.crop`) reescrito sobre a mesma georreferência."""
    s = footprint / out_px / GEOREF.res
    t = math.radians(theta)
    a, b = s * math.cos(t), s * math.sin(t)
    cu, cv = (x - GEOREF.x_origin) / GEOREF.res - 0.5, (GEOREF.y_origin - y) / GEOREF.res - 0.5
    h = (out_px - 1) / 2
    matrix = np.array([[a, -b, cu - a * h + b * h], [b, a, cv - b * h - a * h]])
    out_px = int(out_px)
    return cv2.warpAffine(image, matrix, (out_px, out_px), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def test_zero_tilt_matches_v1_affine_crop():
    image, spec = _image(), _spec()
    ours = sample_geometry(torch.as_tensor(image), GEOREF, batch_from_spec(spec)).numpy()
    for i, row in spec.iterrows():
        v1 = _v1_affine_crop(image, row.x, row.y, row.footprint_m, row.out_px, row.yaw_deg)
        assert np.abs(ours[i].astype(int) - v1.astype(int)).max() <= 2


def test_batch_and_single_row_rendering_are_identical():
    image = torch.as_tensor(_image())
    spec = _spec(blur_sigma=[0.0, 0.7, 1.5, 0.0, 1.0, 0.3], roll_deg=3.0, pitch_deg=-4.0)
    whole = render_crops(image, GEOREF, batch_from_spec(spec))
    for i in range(len(spec)):
        single = render_crops(image, GEOREF, batch_from_spec(spec.iloc[[i]]))
        assert torch.equal(whole[i], single[0])


def test_rendering_is_deterministic_and_noise_depends_on_crop_id():
    image = torch.as_tensor(_image())
    spec = _spec()
    first = render_crops(image, GEOREF, batch_from_spec(spec))
    assert torch.equal(first, render_crops(image, GEOREF, batch_from_spec(spec)))
    other_ids = render_crops(image, GEOREF, batch_from_spec(spec.assign(crop_id=spec.crop_id + 7)))
    assert not torch.equal(first, other_ids)


def test_tilt_moves_the_footprint_center_like_a_tilted_camera():
    # Pitch positivo inclina o raio central para o norte (y menor no raster): o centro do recorte se desloca.
    image = torch.as_tensor(_image())
    spec = _spec(n=1, yaw_deg=0.0, footprint_m=100.0, out_px=33, noise_std=0.0, brightness=0.0, contrast=1.0)
    from dal.v3.sensor.render import ground_offsets

    east, south = ground_offsets(batch_from_spec(spec.assign(pitch_deg=5.0)))
    height = 50.0 / TAN_HALF_FOV
    assert abs(float(south[0, 16, 16]) + height * math.tan(math.radians(5.0))) < 1e-6
    assert abs(float(east[0, 16, 16])) < 1e-9
    assert render_crops(image, GEOREF, batch_from_spec(spec)).shape == (1, 33, 33, 3)
