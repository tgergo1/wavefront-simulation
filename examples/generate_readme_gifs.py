#!/usr/bin/env python3
"""Generate animated GIF examples from real wavefront simulations.

Requirements:
- Built Python extension available via PYTHONPATH=build/python
- ffmpeg available in PATH
"""

from __future__ import annotations

import binascii
import csv
import json
import math
import shutil
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    import wavefront as wf  # type: ignore
except ImportError:
    import _wavefront as wf  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
CHECKS_DIR = ASSETS / "_checks"
VALIDATION_SUMMARY: dict[str, dict[str, float | int]] = {}


RGB = Tuple[int, int, int]
Field2D = List[List[float]]
Overlay2D = List[List[int]]
Volume3D = List[List[List[float]]]
HypersliceGrid = List[List[Field2D]]  # [row][col] -> Field2D for 4D visualisation


@dataclass
class Layout:
    cell: int = 4
    margin: int = 16
    panel_gap: int = 12
    panel_pad: int = 6
    stripe_h: int = 14
    legend_h: int = 10
    legend_margin: int = 10


BITMAP_FONT = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
}

BT709_LUMINANCE = (0.2126, 0.7152, 0.0722)
LIGHT_TEXT = (248, 250, 252)
DARK_TEXT = (31, 41, 55)
TEXT_LUMINANCE_THRESHOLD = 145.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def blend(c0: RGB, c1: RGB, alpha: float) -> RGB:
    a = clamp(alpha, 0.0, 1.0)
    return (
        int((1.0 - a) * c0[0] + a * c1[0]),
        int((1.0 - a) * c0[1] + a * c1[1]),
        int((1.0 - a) * c0[2] + a * c1[2]),
    )


def color_for(v: float) -> RGB:
    """Scientific diverging map: deep blue -> pale neutral -> warm red."""
    n = clamp(v, -1.0, 1.0)
    anchors: List[Tuple[float, RGB]] = [
        (-1.0, (23, 43, 109)),
        (-0.45, (68, 143, 204)),
        (0.0, (246, 244, 239)),
        (0.45, (236, 146, 75)),
        (1.0, (148, 34, 49)),
    ]
    for idx, (left_n, left_c) in enumerate(anchors[:-1]):
        right_n, right_c = anchors[idx + 1]
        if left_n <= n <= right_n:
            alpha = (n - left_n) / max(1.0e-12, right_n - left_n)
            return blend(left_c, right_c, alpha)
    return anchors[0][1] if n < 0.0 else anchors[-1][1]


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png_rgb(path: Path, width: int, height: int, rows: List[bytearray]) -> None:
    raw = bytearray()
    for row in rows:
        raw.append(0)
        raw.extend(row)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = zlib.compress(bytes(raw), level=9)

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(_chunk(b"IHDR", ihdr))
    png.extend(_chunk(b"IDAT", data))
    png.extend(_chunk(b"IEND", b""))
    path.write_bytes(png)


def make_config(mode: "wf.SolverMode") -> "wf.SolverConfig":
    c = wf.SolverConfig()
    c.mode = mode
    c.precision = wf.PrecisionMode.FastFloat64
    c.cfl = 0.22
    c.max_steps = 0
    c.threads = 6
    c.deterministic = True
    c.spatial_order = 4
    c.split_pml = True
    c.reference_window = 32
    return c


def _boundary(boundary_type: "wf.BoundaryType", axis: int, upper_face: bool, parameter: str) -> "wf.BoundarySpec":
    b = wf.BoundarySpec()
    b.type = boundary_type
    b.axis = axis
    b.upper_face = upper_face
    b.parameter = wf.SymbolicExpr(parameter)
    return b


def pml_boundaries_nd(dims: int, sigma: str = "10.0") -> List["wf.BoundarySpec"]:
    out: List["wf.BoundarySpec"] = []
    for axis in range(dims):
        for upper in (False, True):
            out.append(_boundary(wf.BoundaryType.PML, axis, upper, sigma))
    return out


def periodic_boundaries_nd(dims: int) -> List["wf.BoundarySpec"]:
    out: List["wf.BoundarySpec"] = []
    for axis in range(dims):
        for upper in (False, True):
            out.append(_boundary(wf.BoundaryType.Periodic, axis, upper, "0.0"))
    return out


def pml_boundaries(sigma: str = "10.0") -> List["wf.BoundarySpec"]:
    return pml_boundaries_nd(2, sigma)


def periodic_boundaries() -> List["wf.BoundarySpec"]:
    return periodic_boundaries_nd(2)


def make_problem(
    nx: int,
    ny: int,
    density_expr: str,
    stiffness_expr: str,
    source_expr: str,
    damping_expr: str,
    dispersion_expr: str,
    boundaries: Sequence["wf.BoundarySpec"],
) -> "wf.ProblemSpec":
    p = wf.ProblemSpec()
    p.grid = wf.GridSpec()
    p.grid.dims = 2
    p.grid.shape = [nx, ny]
    p.grid.spacing = [1.0 / max(1, nx - 1), 1.0 / max(1, ny - 1)]
    p.grid.origin = [0.0, 0.0]
    p.field_components = 1

    p.medium = wf.MediumLaw()
    p.medium.density = wf.SymbolicExpr(density_expr)
    p.medium.stiffness = wf.SymbolicExpr(stiffness_expr)
    p.medium.damping = wf.SymbolicExpr(damping_expr)
    p.medium.dispersion = wf.SymbolicExpr(dispersion_expr)
    p.source_term = wf.SymbolicExpr(source_expr)
    p.boundaries = list(boundaries)
    return p


def make_problem_3d(
    nx: int,
    ny: int,
    nz: int,
    stiffness_expr: str,
    source_expr: str,
    damping_expr: str,
    dispersion_expr: str,
    boundaries: Sequence["wf.BoundarySpec"],
) -> "wf.ProblemSpec":
    p = wf.ProblemSpec()
    p.grid = wf.GridSpec()
    p.grid.dims = 3
    p.grid.shape = [nx, ny, nz]
    p.grid.spacing = [1.0 / max(1, nx - 1), 1.0 / max(1, ny - 1), 1.0 / max(1, nz - 1)]
    p.grid.origin = [0.0, 0.0, 0.0]
    p.field_components = 1

    p.medium = wf.MediumLaw()
    p.medium.density = wf.SymbolicExpr("1.0")
    p.medium.stiffness = wf.SymbolicExpr(stiffness_expr)
    p.medium.damping = wf.SymbolicExpr(damping_expr)
    p.medium.dispersion = wf.SymbolicExpr(dispersion_expr)
    p.source_term = wf.SymbolicExpr(source_expr)
    p.boundaries = list(boundaries)
    return p


def make_problem_4d(
    n: int,
    stiffness_expr: str,
    source_expr: str,
    damping_expr: str,
    boundaries: Sequence["wf.BoundarySpec"],
) -> "wf.ProblemSpec":
    p = wf.ProblemSpec()
    p.grid = wf.GridSpec()
    p.grid.dims = 4
    p.grid.shape = [n, n, n, n]
    h = 1.0 / max(1, n - 1)
    p.grid.spacing = [h, h, h, h]
    p.grid.origin = [0.0, 0.0, 0.0, 0.0]
    p.field_components = 1

    p.medium = wf.MediumLaw()
    p.medium.density = wf.SymbolicExpr("1.0")
    p.medium.stiffness = wf.SymbolicExpr(stiffness_expr)
    p.medium.damping = wf.SymbolicExpr(damping_expr)
    p.medium.dispersion = wf.SymbolicExpr("0.0")
    p.source_term = wf.SymbolicExpr(source_expr)
    p.boundaries = list(boundaries)
    return p


def sample_slice_4d(solver: "wf.Solver", n: int, x2_idx: int, x3_idx: int) -> Field2D:
    """Sample a 2-D (x0, x1) cross-section from a 4-D field at fixed x2_idx, x3_idx."""
    field: Field2D = []
    for i1 in range(n):
        row: List[float] = []
        for i0 in range(n):
            row.append(float(solver.sample([i0, i1, x2_idx, x3_idx])[0]))
        field.append(row)
    return field


def sample_field(solver: "wf.Solver", nx: int, ny: int) -> Field2D:
    field: Field2D = []
    for y in range(ny):
        row: List[float] = []
        for x in range(nx):
            row.append(float(solver.sample([x, y])[0]))
        field.append(row)
    return field


def load_csv_scalar_panel(path: Path, nx: int, ny: int, column: str, *, component: int = 0) -> Field2D:
    panel: Field2D = [[0.0 for _ in range(nx)] for _ in range(ny)]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["component"]) != component:
                continue
            flat = int(row["flat"])
            x = flat % nx
            y = flat // nx
            panel[y][x] = float(row.get(column, "0.0") or 0.0)
    return panel


def snapshot_panel(snapshot: "wf.FieldSnapshot", nx: int, ny: int, *, magnitude: bool = False) -> Field2D:
    panel: Field2D = []
    for y in range(ny):
        row: List[float] = []
        for x in range(nx):
            flat = y * nx + x
            if magnitude and getattr(snapshot, "complex_values", None):
                row.append(float(snapshot.complex_values[flat].magnitude()))
            elif magnitude:
                row.append(abs(float(snapshot.values[flat])))
            else:
                row.append(float(snapshot.values[flat]))
        panel.append(row)
    return panel


def resample_series(values: Sequence[float], count: int) -> List[float]:
    if count <= 0:
        return []
    if not values:
        return [0.0] * count
    if len(values) == count:
        return [float(v) for v in values]

    out: List[float] = []
    for i in range(count):
        src = int(round(i * (len(values) - 1) / max(1, count - 1)))
        out.append(float(values[src]))
    return out


def bar_panel(values: Sequence[float], nx: int, ny: int) -> Field2D:
    sampled = [float(v) if math.isfinite(float(v)) else 0.0 for v in resample_series(values, nx)]
    max_value = max((abs(v) for v in sampled), default=0.0)
    max_value = max(max_value, 1.0e-12)

    panel: Field2D = [[-0.08 for _ in range(nx)] for _ in range(ny)]
    for x, value in enumerate(sampled):
        height = int((abs(value) / max_value) * (ny - 1))
        for y in range(height + 1):
            panel[y][x] = abs(value) / max_value
    return panel


def sample_slice_xy(solver: "wf.Solver", nx: int, ny: int, z: int) -> Field2D:
    field: Field2D = []
    for y in range(ny):
        row: List[float] = []
        for x in range(nx):
            row.append(float(solver.sample([x, y, z])[0]))
        field.append(row)
    return field


def sample_slice_xz(solver: "wf.Solver", nx: int, nz: int, y: int) -> Field2D:
    field: Field2D = []
    for z in range(nz):
        row: List[float] = []
        for x in range(nx):
            row.append(float(solver.sample([x, y, z])[0]))
        field.append(row)
    return field


def sample_slice_yz(solver: "wf.Solver", ny: int, nz: int, x: int) -> Field2D:
    field: Field2D = []
    for z in range(nz):
        row: List[float] = []
        for y in range(ny):
            row.append(float(solver.sample([x, y, z])[0]))
        field.append(row)
    return field


def sample_volume(solver: "wf.Solver", nx: int, ny: int, nz: int) -> Volume3D:
    volume: Volume3D = []
    for z in range(nz):
        yz: List[List[float]] = []
        for y in range(ny):
            row: List[float] = []
            for x in range(nx):
                row.append(float(solver.sample([x, y, z])[0]))
            yz.append(row)
        volume.append(yz)
    return volume


def max_abs_in_sequence(sequence: Sequence[Sequence[Field2D]]) -> float:
    m = 0.0
    for frame_panels in sequence:
        for panel in frame_panels:
            for row in panel:
                for v in row:
                    av = abs(v)
                    if av > m:
                        m = av
    return max(m, 1.0e-12)


def max_abs_by_panel(sequence: Sequence[Sequence[Field2D]]) -> List[float]:
    if not sequence:
        return []

    scales = [0.0 for _ in sequence[0]]
    for frame_panels in sequence:
        for panel_idx, panel in enumerate(frame_panels):
            for row in panel:
                for value in row:
                    scales[panel_idx] = max(scales[panel_idx], abs(value))
    return [max(scale, 1.0e-12) for scale in scales]


def max_abs_in_volumes(volumes: Sequence[Volume3D]) -> float:
    m = 0.0
    for volume in volumes:
        for yz in volume:
            for row in yz:
                for value in row:
                    av = abs(value)
                    if av > m:
                        m = av
    return max(m, 1.0e-12)


def max_abs_in_hyperslice_frames(all_frames: Sequence[HypersliceGrid]) -> float:
    m = 0.0
    for grid in all_frames:
        for row_panels in grid:
            for panel in row_panels:
                for row in panel:
                    for v in row:
                        av = abs(v)
                        if av > m:
                            m = av
    return max(m, 1.0e-12)


def render_hyperslice_grid_frame(
    grid_panels: HypersliceGrid,
    scale: float,
    cell: int = 5,
    panel_gap: int = 5,
    margin: int = 14,
    stripe_h: int = 5,
    legend_h: int = 10,
    legend_margin: int = 8,
) -> Tuple[int, int, List[bytearray]]:
    """Render a 3×3 hyperslice grid for a 4-D wavefield.

    Each row of panels corresponds to a fixed x2 position (top=1/4, mid=1/2,
    bot=3/4); each column to a fixed x3 position (left=1/4, mid=1/2, right=3/4).
    The stripe colour on top of each row encodes the x2 level.
    """
    n_rows = len(grid_panels)
    n_cols = len(grid_panels[0])
    ny = len(grid_panels[0][0])
    nx = len(grid_panels[0][0][0])

    panel_w = nx * cell
    panel_h = ny * cell

    width = margin * 2 + n_cols * panel_w + (n_cols - 1) * panel_gap
    height = (
        margin * 2
        + n_rows * (stripe_h + panel_h)
        + (n_rows - 1) * panel_gap
        + legend_margin
        + legend_h
    )

    bg = (245, 248, 255)
    rows_out: List[bytearray] = [bytearray([bg[0], bg[1], bg[2]] * width) for _ in range(height)]

    def fill_rect(x: int, y: int, w: int, h: int, c: RGB) -> None:
        for yy in range(max(0, y), min(height, y + h)):
            row = rows_out[yy]
            for xx in range(max(0, x), min(width, x + w)):
                i = xx * 3
                row[i] = c[0]
                row[i + 1] = c[1]
                row[i + 2] = c[2]

    # Stripe colours indicate the x2 level for each row.
    row_stripe_colors: List[RGB] = [
        (21, 82, 161),   # top row: x2 = n/4 (near lower edge)
        (56, 131, 84),   # mid row: x2 = n/2 (center slice)
        (177, 82, 27),   # bot row: x2 = 3n/4 (near upper edge)
    ]

    cell_h = stripe_h + panel_h

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            panel = grid_panels[row_idx][col_idx]
            px = margin + col_idx * (panel_w + panel_gap)
            py = margin + row_idx * (cell_h + panel_gap)

            stripe_color = row_stripe_colors[row_idx % len(row_stripe_colors)]
            fill_rect(px, py, panel_w, stripe_h, stripe_color)

            for j in range(ny):
                src_row = panel[ny - 1 - j]
                for i in range(nx):
                    color = color_for(math.tanh(1.8 * src_row[i] / scale))
                    fill_rect(px + i * cell, py + stripe_h + j * cell, cell, cell, color)

    legend_w = min(300, width - 2 * margin)
    legend_x = (width - legend_w) // 2
    legend_y = height - margin - legend_h
    for i in range(legend_w):
        n_val = -1.0 + 2.0 * (i / max(1, legend_w - 1))
        fill_rect(legend_x + i, legend_y, 1, legend_h, color_for(n_val))

    return width, height, rows_out


def generate_gif_hyperslice_grid(
    name: str,
    all_frames: Sequence[HypersliceGrid],
    fps: int,
    cell: int = 5,
) -> Path:
    output = ASSETS / name
    ASSETS.mkdir(parents=True, exist_ok=True)

    scale = max_abs_in_hyperslice_frames(all_frames)

    with tempfile.TemporaryDirectory(prefix="wavefront_4d_gif_") as tmp:
        tmp_path = Path(tmp)
        for idx, grid in enumerate(all_frames):
            width, height, rows = render_hyperslice_grid_frame(grid, scale, cell=cell)
            frame_path = tmp_path / f"frame_{idx:04d}.png"
            write_png_rgb(frame_path, width, height, rows)

        encode_gif_from_frames(tmp_path, fps=fps, output=output)

    return output


def rms_region(field: Field2D, x0: int, x1: int, y0: int, y1: int) -> float:
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(len(field[0]), x1)
    y1 = min(len(field), y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    accum = 0.0
    count = 0
    for y in range(y0, y1):
        row = field[y]
        for x in range(x0, x1):
            value = row[x]
            accum += value * value
            count += 1
    return math.sqrt(accum / max(1, count))


def rms_rows_at_x(field: Field2D, x: int, rows: Sequence[int]) -> float:
    if not rows:
        return 0.0
    x = clamp(float(x), 0.0, float(len(field[0]) - 1))
    xi = int(x)
    accum = 0.0
    count = 0
    for row in rows:
        if row < 0 or row >= len(field):
            continue
        value = field[row][xi]
        accum += value * value
        count += 1
    return math.sqrt(accum / max(1, count))


def abs_profile_at_x(field: Field2D, x: int) -> List[float]:
    x = int(clamp(float(x), 0.0, float(len(field[0]) - 1)))
    return [abs(field[y][x]) for y in range(len(field))]


def count_prominent_peaks(values: Sequence[float], threshold_fraction: float) -> int:
    if len(values) < 3:
        return 0
    peak_threshold = max(values) * threshold_fraction if values else 0.0
    peaks = 0
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1] and values[i] > peak_threshold:
            peaks += 1
    return peaks


def l2_difference(lhs: Field2D, rhs: Field2D) -> float:
    if len(lhs) != len(rhs) or len(lhs[0]) != len(rhs[0]):
        raise ValueError("Field shape mismatch in l2_difference")
    accum = 0.0
    count = 0
    for y in range(len(lhs)):
        for x in range(len(lhs[0])):
            d = lhs[y][x] - rhs[y][x]
            accum += d * d
            count += 1
    return math.sqrt(accum / max(1, count))


def subtract_fields(lhs: Field2D, rhs: Field2D) -> Field2D:
    if len(lhs) != len(rhs):
        raise ValueError("Field shape mismatch in subtract_fields")
    if not lhs:
        return []

    width = len(lhs[0])
    if any(len(row) != width for row in lhs) or any(len(row) != width for row in rhs):
        raise ValueError("Non-rectangular field supplied to subtract_fields")
    return [[lhs[y][x] - rhs[y][x] for x in range(width)] for y in range(len(lhs))]


def demean_field(field: Field2D) -> Field2D:
    ny = len(field)
    nx = len(field[0]) if ny > 0 else 0
    if nx == 0 or ny == 0:
        return field
    mean = sum(value for row in field for value in row) / float(nx * ny)
    return [[value - mean for value in row] for row in field]


def accumulate_energy_field(
    accumulated: Field2D,
    field: Field2D,
    *,
    x_start: int = 0,
) -> Field2D:
    ny = len(field)
    nx = len(field[0]) if ny > 0 else 0
    if not accumulated:
        accumulated = [[0.0 for _ in range(nx)] for _ in range(ny)]

    out: Field2D = []
    for y in range(ny):
        row: List[float] = []
        for x in range(nx):
            prior = accumulated[y][x]
            if x >= x_start:
                value = field[y][x]
                prior += value * value
            row.append(prior)
        out.append(row)
    return out


def log_positive_field(field: Field2D) -> Field2D:
    peak = 0.0
    for row in field:
        for value in row:
            if value > peak:
                peak = value
    if peak <= 0.0:
        return [[0.0 for _ in row] for row in field]
    scale = math.log1p(peak)
    return [[math.log1p(max(0.0, value)) / scale for value in row] for row in field]


def text_width(text: str, scale: int = 1) -> int:
    width = 0
    for ch in text.upper():
        glyph = BITMAP_FONT.get(ch, BITMAP_FONT[" "])
        width += len(glyph[0]) * scale + scale
    return max(0, width - scale)


def text_color_for(background: RGB) -> RGB:
    # ITU-R BT.709 coefficients keep title text legible across the stripe palette.
    luminance = (
        BT709_LUMINANCE[0] * background[0]
        + BT709_LUMINANCE[1] * background[1]
        + BT709_LUMINANCE[2] * background[2]
    )
    return LIGHT_TEXT if luminance < TEXT_LUMINANCE_THRESHOLD else DARK_TEXT


def render_frame(
    panels: Sequence[Field2D],
    scales: Sequence[float],
    layout: Layout,
    stripe_colors: Sequence[RGB],
    overlays: Sequence[Overlay2D] | None = None,
    panel_titles: Sequence[str] | None = None,
) -> Tuple[int, int, List[bytearray]]:
    ny = len(panels[0])
    nx = len(panels[0][0])

    panel_w = nx * layout.cell
    panel_h = ny * layout.cell
    inner_h = layout.stripe_h + panel_h

    width = (
        layout.margin * 2
        + len(panels) * (panel_w + 2 * layout.panel_pad)
        + (len(panels) - 1) * layout.panel_gap
    )
    height = layout.margin * 2 + inner_h + layout.legend_margin + layout.legend_h

    bg = (245, 248, 255)
    panel_bg = (255, 255, 255)
    border = (184, 201, 230)
    rows: List[bytearray] = [bytearray([bg[0], bg[1], bg[2]] * width) for _ in range(height)]

    def set_px(x: int, y: int, c: RGB) -> None:
        if x < 0 or y < 0 or x >= width or y >= height:
            return
        i = x * 3
        rows[y][i] = c[0]
        rows[y][i + 1] = c[1]
        rows[y][i + 2] = c[2]

    def fill_rect(x: int, y: int, w: int, h: int, c: RGB) -> None:
        for yy in range(max(0, y), min(height, y + h)):
            row = rows[yy]
            for xx in range(max(0, x), min(width, x + w)):
                i = xx * 3
                row[i] = c[0]
                row[i + 1] = c[1]
                row[i + 2] = c[2]

    def stroke_rect(x: int, y: int, w: int, h: int, c: RGB) -> None:
        for xx in range(x, x + w):
            set_px(xx, y, c)
            set_px(xx, y + h - 1, c)
        for yy in range(y, y + h):
            set_px(x, yy, c)
            set_px(x + w - 1, yy, c)

    def draw_text(x: int, y: int, text: str, color: RGB, scale: int = 1) -> None:
        cursor = x
        for ch in text.upper():
            glyph = BITMAP_FONT.get(ch, BITMAP_FONT[" "])
            for gy, glyph_row in enumerate(glyph):
                for gx, bit in enumerate(glyph_row):
                    if bit != "1":
                        continue
                    for sy in range(scale):
                        for sx in range(scale):
                            set_px(cursor + gx * scale + sx, y + gy * scale + sy, color)
            cursor += len(glyph[0]) * scale + scale

    left = layout.margin
    top = layout.margin

    for panel_idx, panel in enumerate(panels):
        panel_scale = scales[panel_idx] if panel_idx < len(scales) else 1.0
        outer_x = left + panel_idx * (panel_w + 2 * layout.panel_pad + layout.panel_gap)
        outer_y = top
        outer_w = panel_w + 2 * layout.panel_pad
        outer_h = inner_h

        fill_rect(outer_x, outer_y, outer_w, outer_h, panel_bg)
        stroke_rect(outer_x, outer_y, outer_w, outer_h, border)

        stripe_color = stripe_colors[panel_idx % len(stripe_colors)]
        fill_rect(
            outer_x + 1,
            outer_y + 1,
            outer_w - 2,
            layout.stripe_h,
            stripe_color,
        )
        if panel_titles is not None and panel_idx < len(panel_titles):
            title = panel_titles[panel_idx]
            title_x = outer_x + max(3, (outer_w - text_width(title)) // 2)
            title_y = outer_y + max(1, (layout.stripe_h - 7) // 2)
            draw_text(title_x, title_y, title, text_color_for(stripe_color))

        panel_x = outer_x + layout.panel_pad
        panel_y = outer_y + layout.stripe_h
        overlay = overlays[panel_idx] if overlays is not None else None

        for y in range(ny):
            src_row = panel[ny - 1 - y]
            overlay_row = overlay[ny - 1 - y] if overlay is not None else None
            for x in range(nx):
                # Compress dynamic range so low-amplitude far-field structure remains visible.
                color = color_for(math.tanh(1.8 * src_row[x] / panel_scale))
                if overlay_row is not None:
                    mask = overlay_row[x]
                    if mask == 1:
                        color = blend(color, (34, 38, 50), 0.78)
                    elif mask == 2:
                        color = blend(color, (38, 208, 205), 0.72)
                fill_rect(
                    panel_x + x * layout.cell,
                    panel_y + y * layout.cell,
                    layout.cell,
                    layout.cell,
                    color,
                )

    legend_w = min(360, width - 2 * layout.margin)
    legend_x = (width - legend_w) // 2
    legend_y = height - layout.margin - layout.legend_h
    for i in range(legend_w):
        n = -1.0 + 2.0 * (i / max(1, legend_w - 1))
        fill_rect(legend_x + i, legend_y, 1, layout.legend_h, color_for(n))

    return width, height, rows


def render_volume_frame(
    volume: Volume3D,
    scale: float,
    yaw: float,
    pitch: float,
    width: int = 620,
    height: int = 430,
) -> Tuple[int, int, List[bytearray]]:
    bg = (245, 248, 255)
    rows: List[bytearray] = [bytearray([bg[0], bg[1], bg[2]] * width) for _ in range(height)]

    def set_px(x: int, y: int, c: RGB) -> None:
        if x < 0 or y < 0 or x >= width or y >= height:
            return
        i = x * 3
        rows[y][i] = c[0]
        rows[y][i + 1] = c[1]
        rows[y][i + 2] = c[2]

    def blend_px(x: int, y: int, c: RGB, a: float) -> None:
        if x < 0 or y < 0 or x >= width or y >= height:
            return
        i = x * 3
        base = (rows[y][i], rows[y][i + 1], rows[y][i + 2])
        mixed = blend(base, c, a)
        rows[y][i] = mixed[0]
        rows[y][i + 1] = mixed[1]
        rows[y][i + 2] = mixed[2]

    def fill_rect(x: int, y: int, w: int, h: int, c: RGB) -> None:
        for yy in range(max(0, y), min(height, y + h)):
            row = rows[yy]
            for xx in range(max(0, x), min(width, x + w)):
                i = xx * 3
                row[i] = c[0]
                row[i + 1] = c[1]
                row[i + 2] = c[2]

    def draw_line(x0: int, y0: int, x1: int, y1: int, c: RGB) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x = x0
        y = y0
        while True:
            set_px(x, y, c)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    cx = width // 2
    cy = height // 2 - 14
    dist = 3.3
    proj = 230.0
    c_yaw = math.cos(yaw)
    s_yaw = math.sin(yaw)
    c_pitch = math.cos(pitch)
    s_pitch = math.sin(pitch)

    def project(xn: float, yn: float, zn: float) -> Tuple[int, int, float]:
        x1 = c_yaw * xn - s_yaw * zn
        z1 = s_yaw * xn + c_yaw * zn
        y1 = c_pitch * yn - s_pitch * z1
        z2 = s_pitch * yn + c_pitch * z1

        inv = 1.0 / (z2 + dist)
        sx = int(cx + proj * x1 * inv)
        sy = int(cy - proj * y1 * inv)
        return sx, sy, z2

    # Draw a subtle projected cube to indicate 3D axes.
    cube = [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    projected = [project(x, y, z) for (x, y, z) in cube]
    for a, b in edges:
        x0, y0, _ = projected[a]
        x1, y1, _ = projected[b]
        draw_line(x0, y0, x1, y1, (176, 193, 223))

    nz = len(volume)
    ny = len(volume[0]) if nz > 0 else 0
    nx = len(volume[0][0]) if ny > 0 else 0

    particles: List[Tuple[float, int, int, RGB, float, int]] = []
    threshold = 0.06 * scale

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                value = volume[z][y][x]
                av = abs(value)
                if av < threshold:
                    continue

                xn = 2.0 * (x / max(1, nx - 1) - 0.5)
                yn = 2.0 * (y / max(1, ny - 1) - 0.5)
                zn = 2.0 * (z / max(1, nz - 1) - 0.5)
                sx, sy, depth = project(xn, yn, zn)
                color = color_for(math.tanh(2.0 * value / scale))

                strength = av / scale
                alpha = clamp(0.10 + 0.52 * strength, 0.10, 0.68)
                radius = 1 if strength < 0.35 else 2
                particles.append((depth, sx, sy, color, alpha, radius))

    particles.sort(key=lambda item: item[0])  # far to near painter order

    for _, sx, sy, color, alpha, radius in particles:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    blend_px(sx + dx, sy + dy, color, alpha)

    legend_w = 360
    legend_x = (width - legend_w) // 2
    legend_y = height - 26
    for i in range(legend_w):
        n = -1.0 + 2.0 * (i / max(1, legend_w - 1))
        fill_rect(legend_x + i, legend_y, 1, 10, color_for(n))

    return width, height, rows


def encode_gif_from_frames(frame_dir: Path, fps: int, output: Path) -> None:
    palette = frame_dir / "palette.png"
    frame_pattern = str(frame_dir / "frame_%04d.png")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-vf",
            "palettegen=stats_mode=diff",
            str(palette),
        ],
        check=True,
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-i",
            str(palette),
            "-lavfi",
            "paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(output),
        ],
        check=True,
    )


def generate_gif(
    name: str,
    frame_panels: Sequence[Sequence[Field2D]],
    fps: int,
    layout: Layout,
    stripe_colors: Sequence[RGB],
    overlays: Sequence[Overlay2D] | None = None,
    panel_titles: Sequence[str] | None = None,
    panel_scale_mode: str = "shared",
) -> Path:
    output = ASSETS / name
    ASSETS.mkdir(parents=True, exist_ok=True)

    if panel_scale_mode == "shared":
        scales = [max_abs_in_sequence(frame_panels) for _ in frame_panels[0]]
    elif panel_scale_mode == "panel":
        scales = max_abs_by_panel(frame_panels)
    else:
        raise ValueError(f"Unsupported panel_scale_mode: {panel_scale_mode}")

    with tempfile.TemporaryDirectory(prefix="wavefront_gif_") as tmp:
        tmp_path = Path(tmp)
        for idx, panels in enumerate(frame_panels):
            width, height, rows = render_frame(
                panels,
                scales,
                layout,
                stripe_colors,
                overlays=overlays,
                panel_titles=panel_titles,
            )
            frame_path = tmp_path / f"frame_{idx:04d}.png"
            write_png_rgb(frame_path, width, height, rows)

        encode_gif_from_frames(tmp_path, fps=fps, output=output)

    return output


def simulate_modes_frames() -> tuple[List[List[Field2D]], float, float]:
    nx = 44
    ny = 44
    frames = 68
    steps_per_frame = 6

    problem = make_problem(
        nx,
        ny,
        density_expr="1.0",
        stiffness_expr="1.2 + 0.8*tanh((x_0-0.72)/0.04)",
        source_expr="34.0*sin(25*t)*exp(-((x_0-0.30)*(x_0-0.30)+(x_1-0.72)*(x_1-0.72))/0.02)",
        damping_expr="0.0005",
        dispersion_expr="20.0",
        boundaries=pml_boundaries(),
    )

    solvers = [
        wf.Solver(problem, make_config(wf.SolverMode.LinearApprox)),
        wf.Solver(problem, make_config(wf.SolverMode.NonlinearContinuum)),
        wf.Solver(problem, make_config(wf.SolverMode.MicroSurrogate)),
    ]

    for solver in solvers:
        solver.run(110)

    frames_data: List[List[Field2D]] = []
    for _ in range(frames):
        for solver in solvers:
            solver.run(steps_per_frame)
        frames_data.append([sample_field(solver, nx, ny) for solver in solvers])

    final_linear, final_nonlinear, final_micro = frames_data[-1]
    linear_vs_nonlinear = l2_difference(final_linear, final_nonlinear)
    linear_vs_micro = l2_difference(final_linear, final_micro)
    return frames_data, linear_vs_nonlinear, linear_vs_micro


def scenario_modes_evolution() -> Path:
    frames_data, linear_vs_nonlinear, linear_vs_micro = simulate_modes_frames()
    if linear_vs_nonlinear < 5.0e-4 or linear_vs_micro < 5.0e-4:
        raise RuntimeError(
            "Mode demo invalid: interchangeable modes are not producing meaningfully distinct wavefields."
        )
    VALIDATION_SUMMARY["modes_evolution"] = {
        "linear_vs_nonlinear_l2": linear_vs_nonlinear,
        "linear_vs_micro_l2": linear_vs_micro,
    }

    return generate_gif(
        "wavefield-modes-evolution.gif",
        frame_panels=frames_data,
        fps=12,
        layout=Layout(cell=5),
        stripe_colors=[(21, 82, 161), (222, 121, 33), (56, 131, 84)],
        panel_titles=["LINEAR", "NONLINEAR", "MICRO"],
    )


def scenario_mode_residuals() -> Path:
    frames_data, linear_vs_nonlinear, linear_vs_micro = simulate_modes_frames()
    residual_frames: List[List[Field2D]] = []
    for linear, nonlinear, micro in frames_data:
        residual_frames.append(
            [
                subtract_fields(nonlinear, linear),
                subtract_fields(micro, linear),
            ]
        )

    final_nonlinear_residual, final_micro_residual = residual_frames[-1]
    nonlinear_residual_rms = math.sqrt(
        sum(value * value for row in final_nonlinear_residual for value in row)
        / max(1, len(final_nonlinear_residual) * len(final_nonlinear_residual[0]))
    )
    micro_residual_rms = math.sqrt(
        sum(value * value for row in final_micro_residual for value in row)
        / max(1, len(final_micro_residual) * len(final_micro_residual[0]))
    )
    if nonlinear_residual_rms < 5.0e-4 or micro_residual_rms < 5.0e-4:
        raise RuntimeError(
            f"Mode residual demo invalid: residual panels stayed too close to zero "
            f"(nonlinear={nonlinear_residual_rms:.3e}, micro={micro_residual_rms:.3e})."
        )
    VALIDATION_SUMMARY["mode_residuals"] = {
        "linear_vs_nonlinear_l2": linear_vs_nonlinear,
        "linear_vs_micro_l2": linear_vs_micro,
        "nonlinear_residual_rms": nonlinear_residual_rms,
        "micro_residual_rms": micro_residual_rms,
    }

    return generate_gif(
        "wavefield-mode-residuals.gif",
        frame_panels=residual_frames,
        fps=12,
        layout=Layout(cell=5),
        stripe_colors=[(222, 121, 33), (56, 131, 84)],
        panel_titles=["NONLINEAR", "MICRO"],
    )


def scenario_interface_reflection() -> Path:
    nx = 56
    ny = 44
    frames = 64
    steps_per_frame = 6

    problem = make_problem(
        nx,
        ny,
        density_expr="1.0",
        stiffness_expr="1.0 + 1.4*(0.5 + 0.5*tanh((x_0-0.62)/0.015))",
        source_expr="14.0*sin(22*t)*exp(-((x_0-0.22)*(x_0-0.22)+(x_1-0.50)*(x_1-0.50))/0.012)",
        damping_expr="0.0008",
        dispersion_expr="0.0",
        boundaries=pml_boundaries(),
    )

    solver = wf.Solver(problem, make_config(wf.SolverMode.LinearApprox))
    solver.run(110)

    frames_data: List[List[Field2D]] = []
    for _ in range(frames):
        solver.run(steps_per_frame)
        frames_data.append([sample_field(solver, nx, ny)])

    final_panel = frames_data[-1][0]
    reflected_rms = rms_region(final_panel, 0, int(0.45 * nx), int(0.20 * ny), int(0.80 * ny))
    transmitted_rms = rms_region(final_panel, int(0.75 * nx), nx, int(0.20 * ny), int(0.80 * ny))
    if reflected_rms < 1.0e-4 or transmitted_rms < 1.0e-4:
        raise RuntimeError(
            f"Interface demo invalid: reflection/transmission not both visible "
            f"(reflected={reflected_rms:.3e}, transmitted={transmitted_rms:.3e})."
        )
    VALIDATION_SUMMARY["interface_reflection"] = {
        "reflected_rms": reflected_rms,
        "transmitted_rms": transmitted_rms,
    }

    return generate_gif(
        "wavefield-interface-reflection.gif",
        frame_panels=frames_data,
        fps=12,
        layout=Layout(cell=6),
        stripe_colors=[(93, 79, 166)],
        panel_titles=["INTERFACE"],
    )


def scenario_boundary_comparison() -> Path:
    nx = 72
    ny = 56
    frames = 86
    steps_per_frame = 6

    problem_periodic = make_problem(
        nx,
        ny,
        density_expr="1.0",
        stiffness_expr="1.5",
        source_expr="22.0*sin(34*t)*exp(-32*t)*exp(-((x_0-0.18)*(x_0-0.18))/0.0007)",
        damping_expr="0.0",
        dispersion_expr="0.0",
        boundaries=periodic_boundaries(),
    )
    problem_pml = make_problem(
        nx,
        ny,
        density_expr="1.0",
        stiffness_expr="1.5",
        source_expr="22.0*sin(34*t)*exp(-32*t)*exp(-((x_0-0.18)*(x_0-0.18))/0.0007)",
        damping_expr="0.0",
        dispersion_expr="0.0",
        boundaries=pml_boundaries("10.0"),
    )

    solver_periodic = wf.Solver(problem_periodic, make_config(wf.SolverMode.LinearApprox))
    solver_pml = wf.Solver(problem_pml, make_config(wf.SolverMode.LinearApprox))
    solver_periodic.run(60)
    solver_pml.run(60)

    frames_data: List[List[Field2D]] = []
    early_periodic = 0.0
    early_pml = 0.0
    late_periodic = 0.0
    late_pml = 0.0
    for _ in range(frames):
        solver_periodic.run(steps_per_frame)
        solver_pml.run(steps_per_frame)
        panel_periodic_raw = sample_field(solver_periodic, nx, ny)
        panel_pml_raw = sample_field(solver_pml, nx, ny)
        frames_data.append([demean_field(panel_periodic_raw), demean_field(panel_pml_raw)])

        if len(frames_data) == 15:
            early_periodic = rms_region(panel_periodic_raw, int(0.90 * nx), nx, 0, ny)
            early_pml = rms_region(panel_pml_raw, int(0.90 * nx), nx, 0, ny)
        if len(frames_data) == frames:
            late_periodic = rms_region(panel_periodic_raw, int(0.90 * nx), nx, 0, ny)
            late_pml = rms_region(panel_pml_raw, int(0.90 * nx), nx, 0, ny)

    diag_periodic = json.loads(solver_periodic.diagnostics_json())
    diag_pml = json.loads(solver_pml.diagnostics_json())
    if early_periodic < 1.0e-4:
        raise RuntimeError(
            f"Boundary demo invalid: periodic early right-strip RMS too small ({early_periodic:.3e})."
        )
    if early_pml > early_periodic * 0.2:
        raise RuntimeError(
            f"Boundary demo invalid: PML does not suppress early wrap-around enough "
            f"({early_pml:.3e} vs periodic {early_periodic:.3e})."
        )
    if late_pml > late_periodic * 0.45:
        raise RuntimeError(
            f"Boundary demo invalid: late-time PML strip RMS too large "
            f"({late_pml:.3e} vs periodic {late_periodic:.3e})."
        )
    if float(diag_pml["absorbed_energy"]) <= 0.5:
        raise RuntimeError(
            f"Boundary demo invalid: absorbed_energy too small for PML ({diag_pml['absorbed_energy']})."
        )
    VALIDATION_SUMMARY["boundary_comparison"] = {
        "early_periodic_right_rms": early_periodic,
        "early_pml_right_rms": early_pml,
        "late_periodic_right_rms": late_periodic,
        "late_pml_right_rms": late_pml,
        "periodic_absorbed_energy": float(diag_periodic["absorbed_energy"]),
        "pml_absorbed_energy": float(diag_pml["absorbed_energy"]),
    }

    return generate_gif(
        "wavefield-boundary-comparison.gif",
        frame_panels=frames_data,
        fps=12,
        layout=Layout(cell=5),
        stripe_colors=[(177, 82, 27), (22, 111, 89)],
        panel_titles=["PERIODIC", "PML"],
    )


def scenario_double_slit() -> Path:
    nx = 96
    ny = 72
    frames = 96
    steps_per_frame = 6

    # Opaque wall in x_0 with two transmissive slits in x_1.
    wall_x0 = 0.485
    wall_x1 = 0.515
    slit_ranges = [(0.36, 0.44), (0.56, 0.64)]

    wall_expr = "0.5*(tanh((x_0-0.485)/0.003)-tanh((x_0-0.515)/0.003))"
    slit_1 = "0.5*(tanh((x_1-0.36)/0.006)-tanh((x_1-0.44)/0.006))"
    slit_2 = "0.5*(tanh((x_1-0.56)/0.006)-tanh((x_1-0.64)/0.006))"
    aperture = f"min(1.0,({slit_1})+({slit_2}))"
    blocker = f"({wall_expr})*(1.0-({aperture}))"

    problem = make_problem(
        nx,
        ny,
        density_expr=f"1.0 + 600.0*({blocker})",
        stiffness_expr=f"1.6 - 1.598*({blocker})",
        source_expr="34.0*sin(36*t)*exp(-18*t)*exp(-((x_0-0.12)*(x_0-0.12))/0.0005)",
        damping_expr=f"0.0004 + 45.0*({blocker})",
        dispersion_expr="0.0",
        boundaries=pml_boundaries("14.0"),
    )

    solver = wf.Solver(problem, make_config(wf.SolverMode.LinearApprox))
    solver.run(40)

    dx = 1.0 / max(1, nx - 1)
    dy = 1.0 / max(1, ny - 1)
    overlay: Overlay2D = [[0 for _ in range(nx)] for _ in range(ny)]
    for y in range(ny):
        yv = y * dy
        in_slit = any(lo <= yv <= hi for lo, hi in slit_ranges)
        for x in range(nx):
            xv = x * dx
            if wall_x0 <= xv <= wall_x1:
                if in_slit:
                    overlay[y][x] = 2
                else:
                    overlay[y][x] = 1

    frames_data: List[List[Field2D]] = []
    accumulated_intensity: Field2D = [[0.0 for _ in range(nx)] for _ in range(ny)]
    early_right_rms = 0.0
    slit_to_blocked_ratio = 0.0
    screen_peak_count = 0
    screen_max = 0.0

    near_wall_x = int(0.535 * nx)
    screen_x = int(0.88 * nx)
    slit_rows = list(range(int(0.36 * ny), int(0.44 * ny))) + list(range(int(0.56 * ny), int(0.64 * ny)))
    blocked_rows = (
        list(range(0, int(0.30 * ny)))
        + list(range(int(0.47 * ny), int(0.53 * ny)))
        + list(range(int(0.70 * ny), ny))
    )

    for _ in range(frames):
        solver.run(steps_per_frame)
        panel = sample_field(solver, nx, ny)
        accumulated_intensity = accumulate_energy_field(accumulated_intensity, panel, x_start=int(0.50 * nx))
        frames_data.append([panel, log_positive_field(accumulated_intensity)])

        if len(frames_data) == 15:
            early_right_rms = rms_region(panel, int(0.86 * nx), nx, 0, ny)
        if len(frames_data) == 46:
            slit_rms = rms_rows_at_x(panel, near_wall_x, slit_rows)
            blocked_rms = rms_rows_at_x(panel, near_wall_x, blocked_rows)
            slit_to_blocked_ratio = slit_rms / max(1.0e-12, blocked_rms)
        if len(frames_data) == frames:
            screen_profile = abs_profile_at_x(panel, screen_x)
            screen_peak_count = count_prominent_peaks(screen_profile, 0.22)
            screen_max = max(screen_profile)

    if early_right_rms > 1.0e-4:
        raise RuntimeError(
            f"Double-slit demo invalid: early far-field RMS too high ({early_right_rms:.3e}); "
            "possible phantom/right-side leakage."
        )
    if slit_to_blocked_ratio < 2.0:
        raise RuntimeError(
            f"Double-slit demo invalid: wall blocking too weak (slit/blocked ratio={slit_to_blocked_ratio:.2f})."
        )
    if screen_peak_count < 3 or screen_max < 1.0e-4:
        raise RuntimeError(
            f"Double-slit demo invalid: interference fringe structure too weak "
            f"(peaks={screen_peak_count}, screen_max={screen_max:.3e})."
        )
    VALIDATION_SUMMARY["double_slit"] = {
        "early_far_right_rms": early_right_rms,
        "slit_to_blocked_ratio": slit_to_blocked_ratio,
        "screen_peak_count": screen_peak_count,
        "screen_max_abs": screen_max,
    }

    return generate_gif(
        "wavefield-double-slit.gif",
        frame_panels=frames_data,
        fps=12,
        layout=Layout(cell=5),
        stripe_colors=[(119, 42, 146), (181, 96, 31)],
        overlays=[overlay, overlay],
        panel_titles=["FIELD AND SLITS", "LOG INTENSITY"],
        panel_scale_mode="panel",
    )


def scenario_3d_volume() -> Path:
    nx = 31
    ny = 31
    nz = 31
    frames = 72
    steps_per_frame = 4

    problem = make_problem_3d(
        nx,
        ny,
        nz,
        stiffness_expr="1.5",
        source_expr=(
            "32.0*sin(46*t)*exp(-24*t)*exp(-((x_0-0.50)*(x_0-0.50)+"
            "(x_1-0.50)*(x_1-0.50)+(x_2-0.50)*(x_2-0.50))/0.0018)"
        ),
        damping_expr="0.0001",
        dispersion_expr="0.0",
        boundaries=pml_boundaries_nd(3, "10.0"),
    )

    solver = wf.Solver(problem, make_config(wf.SolverMode.LinearApprox))
    solver.run(70)

    volumes: List[Volume3D] = []
    for _ in range(frames):
        solver.run(steps_per_frame)
        volumes.append(sample_volume(solver, nx, ny, nz))

    scale = max_abs_in_volumes(volumes)
    mid = volumes[len(volumes) // 2]
    cx = nx // 2
    cy = ny // 2
    cz = nz // 2
    r = max(3, min(nx, ny, nz) // 6)
    axis_samples = [
        abs(mid[cz][cy][cx + r]),
        abs(mid[cz][cy][cx - r]),
        abs(mid[cz][cy + r][cx]),
        abs(mid[cz][cy - r][cx]),
        abs(mid[cz + r][cy][cx]),
        abs(mid[cz - r][cy][cx]),
    ]
    axis_max = max(axis_samples)
    axis_min = min(axis_samples)
    isotropy_spread = (axis_max - axis_min) / max(axis_max, 1.0e-12)
    active_voxels = 0
    threshold = 0.08 * scale
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if abs(mid[z][y][x]) > threshold:
                    active_voxels += 1

    if isotropy_spread > 0.35:
        raise RuntimeError(
            f"3D demo invalid: isotropy spread too high ({isotropy_spread:.3f}) for homogeneous medium."
        )
    if active_voxels < 120:
        raise RuntimeError(
            f"3D demo invalid: volumetric support too sparse ({active_voxels} active voxels)."
        )
    VALIDATION_SUMMARY["volume_3d"] = {
        "isotropy_spread": isotropy_spread,
        "active_voxels_midframe": active_voxels,
        "scale": scale,
    }

    output = ASSETS / "wavefield-3d-volume.gif"
    ASSETS.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="wavefront_3d_gif_") as tmp:
        tmp_path = Path(tmp)
        for idx, volume in enumerate(volumes):
            phase = idx / max(1, frames - 1)
            yaw = 0.63 + 0.70 * phase
            pitch = 0.56
            width, height, rows = render_volume_frame(volume, scale, yaw=yaw, pitch=pitch)
            frame_path = tmp_path / f"frame_{idx:04d}.png"
            write_png_rgb(frame_path, width, height, rows)

        encode_gif_from_frames(tmp_path, fps=10, output=output)

    return output


def scenario_4d_hyperslice() -> Path:
    """4-D wave simulation rendered as a 3×3 hyperslice grid.

    A point-like Gaussian pulse is placed at the centre of a unit 4-cube
    [0,1]^4.  Because a wave front in 4-D expands as a 3-sphere (the surface
    of a 4-ball), each 2-D cross-section at a fixed (x2, x3) position shows a
    *circular* arc whose radius depends on how far the slice is from the
    4-D source.  The 3×3 grid encodes:
      rows  – three x2 levels (1/4, 1/2, 3/4)
      cols  – three x3 levels (1/4, 1/2, 3/4)
    The centre panel (x2=1/2, x3=1/2) cuts through the equator of the
    3-sphere and shows the largest ring; corner panels reveal progressively
    smaller cross-sections, directly visualising the 4-D geometry.
    """
    n = 20
    frames = 64
    steps_per_frame = 5

    r4sq = (
        "(x_0-0.5)*(x_0-0.5)"
        "+(x_1-0.5)*(x_1-0.5)"
        "+(x_2-0.5)*(x_2-0.5)"
        "+(x_3-0.5)*(x_3-0.5)"
    )
    problem = make_problem_4d(
        n,
        stiffness_expr="1.0",
        source_expr=f"26.0*sin(24*t)*exp(-12*t)*exp(-({r4sq})/0.005)",
        damping_expr="0.0001",
        boundaries=pml_boundaries_nd(4, "10.0"),
    )

    solver = wf.Solver(problem, make_config(wf.SolverMode.LinearApprox))
    solver.run(50)

    # Sample at quarter, centre, three-quarter along x2 and x3.
    slice_positions = [n // 4, n // 2, 3 * n // 4]

    all_frames: List[HypersliceGrid] = []
    for _ in range(frames):
        solver.run(steps_per_frame)
        grid: HypersliceGrid = []
        for x2_idx in slice_positions:
            row_panels: List[Field2D] = []
            for x3_idx in slice_positions:
                row_panels.append(sample_slice_4d(solver, n, x2_idx, x3_idx))
            grid.append(row_panels)
        all_frames.append(grid)

    # Validation: centre panel should carry the dominant amplitude.
    mid_grid = all_frames[len(all_frames) // 2]
    center_panel = mid_grid[1][1]  # x2=n/2, x3=n/2
    corner_panel = mid_grid[0][0]  # x2=n/4, x3=n/4

    center_peak = max(abs(v) for row in center_panel for v in row)
    corner_peak = max(abs(v) for row in corner_panel for v in row)

    if center_peak < 1.0e-4:
        raise RuntimeError(
            f"4D demo invalid: centre panel peak amplitude too small ({center_peak:.3e})."
        )
    if center_peak < corner_peak:
        raise RuntimeError(
            f"4D demo invalid: centre panel ({center_peak:.3e}) not stronger than corner ({corner_peak:.3e})."
        )

    # Isotropy in the central (x2, x3) = (n/2, n/2) slice along all four
    # in-plane axes at a fixed radius.
    c = n // 2
    r = max(2, n // 5)
    axis_vals = [
        abs(center_panel[c][c + r]),
        abs(center_panel[c][c - r]),
        abs(center_panel[c + r][c]),
        abs(center_panel[c - r][c]),
    ]
    ax_max = max(axis_vals)
    ax_min = min(axis_vals)
    isotropy_spread_4d = (ax_max - ax_min) / max(ax_max, 1.0e-12)

    if isotropy_spread_4d > 0.45:
        raise RuntimeError(
            f"4D demo invalid: in-plane isotropy spread too high ({isotropy_spread_4d:.3f})."
        )

    VALIDATION_SUMMARY["hyperslice_4d"] = {
        "center_panel_peak": center_peak,
        "corner_panel_peak": corner_peak,
        "center_to_corner_ratio": center_peak / max(corner_peak, 1.0e-12),
        "isotropy_spread_4d": isotropy_spread_4d,
    }

    return generate_gif_hyperslice_grid(
        "wavefield-4d-hyperslice.gif",
        all_frames=all_frames,
        fps=12,
        cell=5,
    )


def scenario_longitudinal_wave() -> Path:
    """Longitudinal (compressional / P-wave) simulation.

    A 2-component vector displacement field is driven by a localised Gaussian
    source.  The grad-div spatial operator couples the x- and y-displacement
    components, producing radially expanding compression/rarefaction rings.

    Left panel : divergence of the displacement field  (∇·u)  showing
                 compression (red) and rarefaction (blue) bands.
    Right panel: transverse scalar wave with the same source for comparison.
    """
    nx = 48
    ny = 48
    frames = 72
    steps_per_frame = 6

    source_expr = (
        "16.0*sin(28*t)*exp(-16*t)*exp(-((x_0-0.50)*(x_0-0.50)"
        "+(x_1-0.50)*(x_1-0.50))/0.012)"
    )

    # Longitudinal problem: 2-component vector field with grad-div coupling
    p_long = wf.ProblemSpec()
    p_long.grid = wf.GridSpec()
    p_long.grid.dims = 2
    p_long.grid.shape = [nx, ny]
    p_long.grid.spacing = [1.0 / max(1, nx - 1), 1.0 / max(1, ny - 1)]
    p_long.grid.origin = [0.0, 0.0]
    p_long.field_components = 2
    p_long.wave_type = wf.WaveType.Longitudinal
    p_long.medium = wf.MediumLaw()
    p_long.medium.density = wf.SymbolicExpr("1.0")
    p_long.medium.stiffness = wf.SymbolicExpr("1.3")
    p_long.medium.damping = wf.SymbolicExpr("0.0004")
    p_long.medium.dispersion = wf.SymbolicExpr("0.0")
    p_long.source_term = wf.SymbolicExpr(source_expr)
    p_long.boundaries = pml_boundaries("12.0")

    # Transverse comparison: scalar field with same source/medium
    p_trans = make_problem(
        nx, ny,
        density_expr="1.0",
        stiffness_expr="1.3",
        source_expr=source_expr,
        damping_expr="0.0004",
        dispersion_expr="0.0",
        boundaries=pml_boundaries("12.0"),
    )

    solver_long = wf.Solver(p_long, make_config(wf.SolverMode.LinearApprox))
    solver_trans = wf.Solver(p_trans, make_config(wf.SolverMode.LinearApprox))

    # Warm-up
    solver_long.run(80)
    solver_trans.run(80)

    dx = 1.0 / max(1, nx - 1)
    dy = 1.0 / max(1, ny - 1)

    def sample_divergence(solver: "wf.Solver") -> Field2D:
        """Compute ∇·u = ∂u_x/∂x + ∂u_y/∂y via central differences."""
        # Sample both displacement components on the full grid.
        ux: List[List[float]] = []
        uy: List[List[float]] = []
        for j in range(ny):
            row_x: List[float] = []
            row_y: List[float] = []
            for i in range(nx):
                vals = solver.sample([i, j])
                row_x.append(float(vals[0]))
                row_y.append(float(vals[1]))
            ux.append(row_x)
            uy.append(row_y)

        div: Field2D = []
        for j in range(ny):
            row: List[float] = []
            for i in range(nx):
                # ∂u_x / ∂x  (central)
                if 0 < i < nx - 1:
                    dux_dx = (ux[j][i + 1] - ux[j][i - 1]) / (2.0 * dx)
                elif i == 0:
                    dux_dx = (ux[j][1] - ux[j][0]) / dx
                else:
                    dux_dx = (ux[j][-1] - ux[j][-2]) / dx

                # ∂u_y / ∂y  (central)
                if 0 < j < ny - 1:
                    duy_dy = (uy[j + 1][i] - uy[j - 1][i]) / (2.0 * dy)
                elif j == 0:
                    duy_dy = (uy[1][i] - uy[0][i]) / dy
                else:
                    duy_dy = (uy[-1][i] - uy[-2][i]) / dy

                row.append(dux_dx + duy_dy)
            div.append(row)
        return div

    frames_data: List[List[Field2D]] = []
    for _ in range(frames):
        solver_long.run(steps_per_frame)
        solver_trans.run(steps_per_frame)
        divergence_panel = sample_divergence(solver_long)
        transverse_panel = sample_field(solver_trans, nx, ny)
        frames_data.append([divergence_panel, transverse_panel])

    # Validation: divergence field should carry visible structure.
    final_div = frames_data[-1][0]
    final_trans = frames_data[-1][1]
    div_rms = math.sqrt(
        sum(v * v for row in final_div for v in row) / max(1, nx * ny)
    )
    trans_rms = math.sqrt(
        sum(v * v for row in final_trans for v in row) / max(1, nx * ny)
    )
    diff_l2 = l2_difference(final_div, final_trans)

    if div_rms < 1.0e-6:
        raise RuntimeError(
            f"Longitudinal demo invalid: divergence field RMS too small ({div_rms:.3e})."
        )
    if diff_l2 < 1.0e-6:
        raise RuntimeError(
            f"Longitudinal demo invalid: divergence indistinguishable from transverse "
            f"(l2 diff = {diff_l2:.3e})."
        )

    VALIDATION_SUMMARY["longitudinal_wave"] = {
        "divergence_rms": div_rms,
        "transverse_rms": trans_rms,
        "longitudinal_vs_transverse_l2": diff_l2,
    }

    return generate_gif(
        "wavefield-longitudinal.gif",
        frame_panels=frames_data,
        fps=12,
        layout=Layout(cell=5),
        stripe_colors=[(42, 118, 162), (162, 72, 42)],
        panel_titles=["DIVERGENCE", "TRANSVERSE"],
    )


def scenario_monitor_analysis() -> Path:
    nx = 36
    ny = 28
    frames = 20
    steps_per_frame = 3

    problem = make_problem(
        nx,
        ny,
        density_expr="1.0",
        stiffness_expr="1.15",
        source_expr="16.0*sin(18*t)*exp(-((x_0-0.18)*(x_0-0.18)+(x_1-0.50)*(x_1-0.50))/0.012)",
        damping_expr="0.0010",
        dispersion_expr="0.0",
        boundaries=pml_boundaries(),
    )

    layer = wf.GeometryRegion()
    layer.name = "glass-slab"
    layer.shape = wf.GeometryShape.Layer
    layer.axis = 0
    layer.lower = 0.54
    layer.upper = 0.76
    layer.medium = wf.MediumLaw()
    layer.medium.density = wf.SymbolicExpr("1.8")
    layer.medium.stiffness = wf.SymbolicExpr("3.4")
    layer.medium.damping = wf.SymbolicExpr("0.002")
    layer.medium.dispersion = wf.SymbolicExpr("0.0")
    problem.geometry = [layer]

    center_probe = wf.ProbeMonitorSpec()
    center_probe.name = "center"
    center_probe.index = [nx // 2, ny // 2]
    center_probe.component = 0
    center_probe.capture_complex = True

    output_surface = wf.SurfaceMonitorSpec()
    output_surface.name = "output"
    output_surface.axis = 0
    output_surface.upper_face = True
    output_surface.component = 0

    problem.monitors.probes = [center_probe]
    problem.monitors.surfaces = [output_surface]
    problem.monitors.snapshot_interval = 1
    problem.monitors.spectrum_bins = nx
    problem.monitors.enable_far_field = True

    config = make_config(wf.SolverMode.LinearApprox)
    config.backend = wf.ExecutionBackend.ThreadedCPU
    config.far_field_samples = nx

    solver = wf.Solver(problem, config)
    solver.run(24)

    frames_data: List[List[Field2D]] = []
    for _ in range(frames):
        solver.run(steps_per_frame)
        snapshot = solver.field_snapshot()
        spectrum = solver.probe_spectrum("center", nx)
        far_field = solver.far_field_pattern(nx)
        frames_data.append(
            [
                snapshot_panel(snapshot, nx, ny, magnitude=True),
                bar_panel([sample.magnitude for sample in spectrum], nx, ny),
                bar_panel(far_field.amplitudes, nx, ny),
            ]
        )

    probe_history = solver.probe_history("center")
    transmitted_flux = solver.surface_flux("output")
    final_snapshot = solver.field_snapshot()
    final_spectrum = solver.probe_spectrum("center", nx)
    final_far_field = solver.far_field_pattern(nx)

    max_spectrum = max((sample.magnitude for sample in final_spectrum if math.isfinite(sample.magnitude)), default=0.0)
    max_far_field = max((value for value in final_far_field.amplitudes if math.isfinite(value)), default=0.0)
    if getattr(final_snapshot, "complex_values", None):
        max_snapshot = max(
            (value.magnitude() for value in final_snapshot.complex_values if math.isfinite(value.magnitude())),
            default=0.0,
        )
    else:
        max_snapshot = max((abs(value) for value in final_snapshot.values if math.isfinite(value)), default=0.0)

    if len(probe_history) < frames:
        raise RuntimeError("Monitor demo invalid: insufficient probe history collected.")
    if max_spectrum < 1.0e-6:
        raise RuntimeError("Monitor demo invalid: probe spectrum remained trivial.")
    if max_far_field < 1.0e-6:
        raise RuntimeError("Monitor demo invalid: far-field amplitudes remained trivial.")
    if transmitted_flux.transmitted_proxy <= 0.0:
        raise RuntimeError("Monitor demo invalid: transmitted surface flux was not accumulated.")
    if max_snapshot < 1.0e-6:
        raise RuntimeError("Monitor demo invalid: snapshot magnitude remained trivial.")

    VALIDATION_SUMMARY["monitor_analysis"] = {
        "probe_samples": len(probe_history),
        "peak_spectrum_magnitude": max_spectrum,
        "peak_far_field_amplitude": max_far_field,
        "transmitted_flux": transmitted_flux.transmitted_proxy,
        "peak_snapshot_magnitude": max_snapshot,
    }

    return generate_gif(
        "wavefield-monitor-analysis.gif",
        frame_panels=frames_data,
        fps=12,
        layout=Layout(cell=5),
        stripe_colors=[(54, 112, 179), (160, 101, 28), (82, 142, 96)],
        panel_titles=["SNAPSHOT", "SPECTRUM", "FAR FIELD"],
        panel_scale_mode="panel",
    )


def scenario_collision_wavefronts() -> Path:
    nx = 84
    ny = 56
    frames = 72
    steps_per_frame = 4

    problem = make_problem(
        nx,
        ny,
        density_expr="1.0",
        stiffness_expr="1.25",
        source_expr="0.0",
        damping_expr="0.0005",
        dispersion_expr="0.0",
        boundaries=pml_boundaries("12.0"),
    )
    left_source = wf.WaveSourceSpec()
    left_source.name = "left-wave"
    left_source.wave_id = "left"
    left_source.wave_class = "left"
    left_source.term = wf.SymbolicExpr(
        "18.0*sin(24*t)*exp(-16*t)*exp(-((x_0-0.18)*(x_0-0.18))/0.0012)*exp(-((x_1-0.50)*(x_1-0.50))/0.040)"
    )
    right_source = wf.WaveSourceSpec()
    right_source.name = "right-wave"
    right_source.wave_id = "right"
    right_source.wave_class = "right"
    right_source.term = wf.SymbolicExpr(
        "18.0*sin(24*t)*exp(-16*t)*exp(-((x_0-0.82)*(x_0-0.82))/0.0012)*exp(-((x_1-0.50)*(x_1-0.50))/0.040)"
    )
    problem.sources = [left_source, right_source]

    centre_layer = wf.GeometryRegion()
    centre_layer.name = "collision-band"
    centre_layer.shape = wf.GeometryShape.Layer
    centre_layer.axis = 0
    centre_layer.lower = 0.48
    centre_layer.upper = 0.52
    centre_layer.medium = problem.medium
    problem.geometry = [centre_layer]

    collision_monitor = wf.CollisionMonitorSpec()
    collision_monitor.name = "collision-band"
    collision_monitor.component = 0
    collision_monitor.geometry_region = "collision-band"
    collision_monitor.shell_thickness = 0.04
    collision_monitor.threshold = 1.0e-7
    problem.monitors.collisions = [collision_monitor]

    solver = wf.Solver(problem, make_config(wf.SolverMode.LinearApprox))
    solver.run(36)

    overlay: Overlay2D = [[0 for _ in range(nx)] for _ in range(ny)]
    x_lo = int(0.48 * (nx - 1))
    x_hi = int(0.52 * (nx - 1))
    for y in range(ny):
        for x in range(max(0, x_lo - 1), min(nx, x_hi + 2)):
            overlay[y][x] = 1

    frames_data: List[List[Field2D]] = []
    peak_collision = 0.0
    peak_self = 0.0
    with tempfile.TemporaryDirectory(prefix="wavefront_collision_gif_") as tmp:
        tmp_path = Path(tmp)
        for frame_idx in range(frames):
            solver.run(steps_per_frame)
            field_panel = demean_field(sample_field(solver, nx, ny))
            csv_path = tmp_path / f"collision_{frame_idx:04d}.csv"
            solver.export_field_csv(str(csv_path))
            collision_panel = log_positive_field(load_csv_scalar_panel(csv_path, nx, ny, "collision_activity"))
            self_panel = log_positive_field(load_csv_scalar_panel(csv_path, nx, ny, "self_activity"))
            peak_collision = max(peak_collision, max((max(row) for row in collision_panel), default=0.0))
            peak_self = max(peak_self, max((max(row) for row in self_panel), default=0.0))
            frames_data.append([field_panel, collision_panel, self_panel])

    collision = solver.collision_surface("collision-band")
    if collision.integrated_collision <= 0.0:
        raise RuntimeError("Collision demo invalid: collision monitor stayed at zero.")
    if peak_collision < 1.0e-5:
        raise RuntimeError(f"Collision demo invalid: exported collision activity stayed trivial ({peak_collision:.3e}).")
    if peak_self < peak_collision:
        raise RuntimeError(
            f"Collision demo invalid: self activity unexpectedly weaker than collision activity "
            f"({peak_self:.3e} vs {peak_collision:.3e})."
        )
    if len(collision.wave_pairs) != 1 or len(collision.class_pairs) != 1:
        raise RuntimeError("Collision demo invalid: expected one wave pair and one class pair for the head-on setup.")

    VALIDATION_SUMMARY["collision_wavefronts"] = {
        "integrated_collision": collision.integrated_collision,
        "peak_collision_monitor": collision.peak_collision,
        "peak_collision_activity_frame": peak_collision,
        "peak_self_activity_frame": peak_self,
        "wave_pair_count": len(collision.wave_pairs),
        "class_pair_count": len(collision.class_pairs),
    }

    return generate_gif(
        "wavefield-collision-wavefronts.gif",
        frame_panels=frames_data,
        fps=12,
        layout=Layout(cell=5),
        stripe_colors=[(89, 104, 173), (170, 78, 38), (83, 138, 96)],
        overlays=[overlay, overlay, overlay],
        panel_titles=["FIELD", "COLLISION", "SELF"],
        panel_scale_mode="panel",
    )


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to build GIFs; please install it and retry.")


def main() -> None:
    require_ffmpeg()
    ASSETS.mkdir(parents=True, exist_ok=True)
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        scenario_modes_evolution(),
        scenario_mode_residuals(),
        scenario_interface_reflection(),
        scenario_boundary_comparison(),
        scenario_double_slit(),
        scenario_longitudinal_wave(),
        scenario_collision_wavefronts(),
        scenario_monitor_analysis(),
        scenario_3d_volume(),
        scenario_4d_hyperslice(),
    ]

    metrics_path = CHECKS_DIR / "validation-metrics.json"
    metrics_path.write_text(json.dumps(VALIDATION_SUMMARY, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for output in outputs:
        size_kb = output.stat().st_size / 1024.0
        print(f"wrote {output} ({size_kb:.1f} KiB)")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
