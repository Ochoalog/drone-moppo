"""Renderização de recortes da câmera a partir de um raster georreferenciado (contrato IA-1 ↔ IA-2).

Uma única implementação em PyTorch, usada na CPU (máquina local) e na GPU (Colab), para os dois
lados renderizarem igual. Cada linha de uma spec (`colab/specs/*.parquet`) vira um recorte RGB
`out_px × out_px` em uint8.

Etapas, na ordem da v1 (`dal.features.crop` + `dal.features.photometric`), mais as da v3:
1. **Geometria:** para cada pixel de saída, um raio da câmera (FOV horizontal de 84°) girado por
   roll e pitch é intersectado com o solo; depois aplica-se o yaw no plano da imagem, como na v1.
   Com roll = pitch = 0, a geometria é exatamente o afim da v1 (footprint quadrado, rotação θ).
2. **Amostragem bilinear** com a convenção do OpenCV (centros dos pixels nos inteiros, borda
   constante 0) direto sobre a imagem uint8, sem convertê-la inteira para float. Arredonda para uint8.
3. **Desfoque gaussiano** opcional (σ em pixels de saída; σ = 0 não altera nada).
4. **Fotometria:** `clip(img·contrast + 128·brightness + ruído)` truncado para uint8, como a v1.
   O ruído é gaussiano, gerado por hash de (crop_id, pixel): determinístico dada a spec, igual em
   CPU e GPU e independente da ordem ou do tamanho do lote.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

HORIZONTAL_FOV_DEG = 84.0
TAN_HALF_FOV = math.tan(math.radians(HORIZONTAL_FOV_DEG / 2))
SPEC_COLUMNS = ["crop_id", "site", "date_id", "cog_url", "x", "y", "footprint_m", "yaw_deg", "roll_deg",
                "pitch_deg", "brightness", "contrast", "blur_sigma", "noise_std", "out_px"]
# Coluna extra (compatível com o contrato): resolução da fonte a ler, em m/px. A v1 usa o overview 4×.
SOURCE_RES_COLUMN = "src_res_m"
# Raio fixo do núcleo de desfoque: um raio que dependesse do maior σ do lote mudaria o resultado
# conforme o lote, quebrando a paridade entre CPU e GPU. Cobre σ ≤ 3 px (truncado em 3σ).
BLUR_RADIUS_PX = 9
MAX_BLUR_SIGMA_PX = BLUR_RADIUS_PX / 3
_HASH_MULTIPLIER = 0x45D9F3B  # < 2^27: os produtos cabem em int64 sem estouro
_UINT32_MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class RasterGeoref:
    """Georreferência de um raster alinhado aos eixos (bordas esquerda e superior em UTM, m/px)."""

    x_origin: float
    y_origin: float
    res: float


@dataclass
class CropBatch:
    """Parâmetros de um lote de recortes (tensores 1-D de mesmo comprimento, no mesmo dispositivo)."""

    crop_id: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    footprint_m: torch.Tensor
    yaw_deg: torch.Tensor
    roll_deg: torch.Tensor
    pitch_deg: torch.Tensor
    brightness: torch.Tensor
    contrast: torch.Tensor
    blur_sigma: torch.Tensor
    noise_std: torch.Tensor
    out_px: int

    def __len__(self) -> int:
        return int(self.crop_id.shape[0])


def batch_from_spec(spec, device: str | torch.device = "cpu") -> CropBatch:
    """Converte linhas de uma spec (DataFrame) num CropBatch. Todas as linhas precisam do mesmo out_px."""
    out_px = spec["out_px"].unique()
    if len(out_px) != 1:
        raise ValueError("um lote precisa ter um único out_px")

    def column(name: str, dtype=torch.float64) -> torch.Tensor:
        return torch.as_tensor(spec[name].to_numpy(copy=True), dtype=dtype, device=device)

    return CropBatch(
        crop_id=column("crop_id", torch.int64), x=column("x"), y=column("y"), footprint_m=column("footprint_m"),
        yaw_deg=column("yaw_deg"), roll_deg=column("roll_deg"), pitch_deg=column("pitch_deg"),
        brightness=column("brightness"), contrast=column("contrast"), blur_sigma=column("blur_sigma"),
        noise_std=column("noise_std"), out_px=int(out_px[0]),
    )


# ============================================================================ geometria
def ground_offsets(batch: CropBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Deslocamento no solo (m; leste e sul) de cada pixel de saída em relação ao centro do recorte."""
    n, out = len(batch), batch.out_px
    device = batch.x.device
    centered = torch.arange(out, dtype=torch.float64, device=device) - (out - 1) / 2
    du = centered.view(1, 1, out).expand(n, out, out)  # coluna (leste)
    dv = centered.view(1, out, 1).expand(n, out, out)  # linha (sul)
    half = (batch.footprint_m / 2).view(n, 1, 1)
    meters_per_px = (batch.footprint_m / out).view(n, 1, 1)

    # Raio da câmera (x leste, y sul, z para o solo), com o plano da imagem normalizado pelo footprint.
    ray_x = du * meters_per_px / half * TAN_HALF_FOV
    ray_y = dv * meters_per_px / half * TAN_HALF_FOV
    ray_z = torch.ones_like(ray_x)
    pitch = torch.deg2rad(batch.pitch_deg).view(n, 1, 1)
    roll = torch.deg2rad(batch.roll_deg).view(n, 1, 1)
    # Pitch gira em torno do eixo leste (mexe em y, z); roll gira em torno do eixo sul (mexe em x, z).
    ray_y, ray_z = ray_y * torch.cos(pitch) - ray_z * torch.sin(pitch), ray_y * torch.sin(pitch) + ray_z * torch.cos(pitch)
    ray_x, ray_z = ray_x * torch.cos(roll) + ray_z * torch.sin(roll), -ray_x * torch.sin(roll) + ray_z * torch.cos(roll)
    height = half / TAN_HALF_FOV
    east, south = height * ray_x / ray_z, height * ray_y / ray_z

    # Yaw no plano da imagem, igual à v1: [cos −sin; sin cos] aplicado a (coluna, linha).
    yaw = torch.deg2rad(batch.yaw_deg).view(n, 1, 1)
    return east * torch.cos(yaw) - south * torch.sin(yaw), east * torch.sin(yaw) + south * torch.cos(yaw)


def source_pixel_coords(batch: CropBatch, georef: RasterGeoref) -> tuple[torch.Tensor, torch.Tensor]:
    """Coordenadas contínuas (coluna, linha) na fonte, com centros dos pixels nos inteiros (OpenCV)."""
    east, south = ground_offsets(batch)
    n = len(batch)
    center_col = ((batch.x - georef.x_origin) / georef.res - 0.5).view(n, 1, 1)
    center_row = ((georef.y_origin - batch.y) / georef.res - 0.5).view(n, 1, 1)
    return center_col + east / georef.res, center_row + south / georef.res


def bilinear_sample(image: torch.Tensor, cols: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Bilinear sobre a imagem uint8 (H, W, 3) com borda constante 0; devolve float64 (..., 3)."""
    height, width = image.shape[:2]
    flat = image.reshape(-1, 3)
    col0, row0 = torch.floor(cols), torch.floor(rows)
    frac_c, frac_r = (cols - col0).unsqueeze(-1), (rows - row0).unsqueeze(-1)
    col0, row0 = col0.long(), row0.long()
    out = torch.zeros(cols.shape + (3,), dtype=torch.float64, device=image.device)
    for d_row, d_col, weight in ((0, 0, (1 - frac_r) * (1 - frac_c)), (0, 1, (1 - frac_r) * frac_c),
                                 (1, 0, frac_r * (1 - frac_c)), (1, 1, frac_r * frac_c)):
        r, c = row0 + d_row, col0 + d_col
        inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
        index = (r.clamp(0, height - 1) * width + c.clamp(0, width - 1))
        values = flat[index].to(torch.float64) * inside.unsqueeze(-1)
        out += weight * values
    return out


def sample_geometry(image: torch.Tensor, georef: RasterGeoref, batch: CropBatch) -> torch.Tensor:
    """Recortes (N, out, out, 3) uint8 só com a geometria (sem desfoque nem fotometria)."""
    cols, rows = source_pixel_coords(batch, georef)
    return torch.round(bilinear_sample(image, cols, rows)).clamp(0, 255).to(torch.uint8)


# ============================================================================ desfoque e fotometria
def gaussian_blur(crops: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Desfoque gaussiano separável com σ por recorte (px); recortes com σ = 0 não mudam."""
    if not bool((sigma > 0).any()):
        return crops.to(torch.float64)
    if bool((sigma > MAX_BLUR_SIGMA_PX).any()):
        raise ValueError(f"blur_sigma acima de {MAX_BLUR_SIGMA_PX} px não é suportado")
    radius = BLUR_RADIUS_PX
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float64, device=crops.device)
    safe_sigma = torch.where(sigma > 0, sigma, torch.ones_like(sigma)).view(-1, 1)
    kernels = torch.exp(-0.5 * (offsets.view(1, -1) / safe_sigma) ** 2)
    kernels = kernels / kernels.sum(dim=1, keepdim=True)
    identity = (offsets == 0).to(torch.float64).view(1, -1).expand_as(kernels)
    kernels = torch.where((sigma > 0).view(-1, 1), kernels, identity)

    n, out, _, channels = crops.shape
    x = crops.to(torch.float64).permute(0, 3, 1, 2).reshape(1, n * channels, out, out)
    weight = kernels.repeat_interleave(channels, dim=0)
    x = torch.nn.functional.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = torch.nn.functional.conv2d(x, weight.view(-1, 1, 1, 2 * radius + 1), groups=n * channels)
    x = torch.nn.functional.conv2d(x, weight.view(-1, 1, 2 * radius + 1, 1), groups=n * channels)
    return x.view(n, channels, out, out).permute(0, 2, 3, 1)


def _hash32(values: torch.Tensor) -> torch.Tensor:
    """Hash inteiro de 32 bits (xorshift-multiplica), sem estouro em int64."""
    x = values & _UINT32_MASK
    for _ in range(3):
        x = ((x >> 16) ^ x) & _UINT32_MASK
        x = (x * _HASH_MULTIPLIER) & _UINT32_MASK
    return ((x >> 16) ^ x) & _UINT32_MASK


def deterministic_noise(crop_id: torch.Tensor, out_px: int) -> torch.Tensor:
    """Ruído N(0, 1) de forma (N, out, out, 3), função só de (crop_id, pixel, canal)."""
    n = crop_id.shape[0]
    per_crop = out_px * out_px * 3
    pixel = torch.arange(per_crop, dtype=torch.int64, device=crop_id.device).view(1, -1)
    key = (crop_id.view(-1, 1) & _UINT32_MASK) * per_crop * 2 + pixel * 2
    u1 = (_hash32(_hash32(key) ^ (key >> 32)).to(torch.float64) + 0.5) / 2**32
    u2 = (_hash32(_hash32(key + 1) ^ (key >> 32)).to(torch.float64) + 0.5) / 2**32
    normal = torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * math.pi * u2)
    return normal.view(n, out_px, out_px, 3)


def apply_photometric(crops: torch.Tensor, batch: CropBatch, noise: torch.Tensor | None = None) -> torch.Tensor:
    """`clip(img·contrast + 128·brightness + noise_std·ruído)` truncado para uint8 (mesma ordem da v1).

    `noise` permite injetar um ruído já pronto (usado só para conferir a reprodução dos recortes v1).
    """
    n = len(batch)
    if noise is None:
        noise = deterministic_noise(batch.crop_id, batch.out_px) * batch.noise_std.view(n, 1, 1, 1)
    out = crops.to(torch.float64) * batch.contrast.view(n, 1, 1, 1) + 128 * batch.brightness.view(n, 1, 1, 1) + noise
    return out.clamp(0, 255).to(torch.uint8)  # truncamento, como o astype(np.uint8) da v1


def render_crops(image: torch.Tensor, georef: RasterGeoref, batch: CropBatch) -> torch.Tensor:
    """Função do contrato: recortes (N, out, out, 3) uint8 para um lote de linhas de spec da mesma fonte."""
    crops = sample_geometry(image, georef, batch)
    blurred = gaussian_blur(crops, batch.blur_sigma)
    return apply_photometric(blurred, batch)


def render_spec_rows(image: np.ndarray, georef: RasterGeoref, spec, device: str = "cpu",
                     chunk: int = 512) -> np.ndarray:
    """Conveniência: renderiza um DataFrame de spec em blocos; devolve numpy (N, out, out, 3) uint8."""
    image_t = torch.as_tensor(image, device=device)
    parts = []
    for start in range(0, len(spec), chunk):
        batch = batch_from_spec(spec.iloc[start:start + chunk], device)
        parts.append(render_crops(image_t, georef, batch).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty((0,) * 4, np.uint8)
