#!/usr/bin/env python3
"""Generate animated GIF examples from real wavefront simulations.

Requirements:
- Built Python extension available via PYTHONPATH=build/python
- ffmpeg available in PATH
"""

from __future__ import annotations

import binascii
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


@dataclass
class Layout:
    cell: int = 4
    margin: int = 16
    panel_gap: int = 12
    panel_pad: int = 6
    stripe_h: int = 6
    legend_h: int = 10
    legend_margin: int = 10


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
    """Diverging map: blue (negative) -> white -> red (positive)."""
    n = clamp(v, -1.0, 1.0)
    if n < 0.0:
        t = n + 1.0
        r = int(255 * t)
        g = int(255 * t)
        b = 255
    else:
        t = 1.0 - n
        r = 255
        g = int(255 * t)
        b = int(255 * t)
    return r, g, b


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


def sample_field(solver: "wf.Solver", nx: int, ny: int) -> Field2D:
    field: Field2D = []
    for y in range(ny):
        row: List[float] = []
        for x in range(nx):
            row.append(float(solver.sample([x, y])[0]))
        field.append(row)
    return field


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


def demean_field(field: Field2D) -> Field2D:
    ny = len(field)
    nx = len(field[0]) if ny > 0 else 0
    if nx == 0 or ny == 0:
        return field
    mean = sum(value for row in field for value in row) / float(nx * ny)
    return [[value - mean for value in row] for row in field]


def render_frame(
    panels: Sequence[Field2D],
    scale: float,
    layout: Layout,
    stripe_colors: Sequence[RGB],
    overlays: Sequence[Overlay2D] | None = None,
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

    left = layout.margin
    top = layout.margin

    for panel_idx, panel in enumerate(panels):
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

        panel_x = outer_x + layout.panel_pad
        panel_y = outer_y + layout.stripe_h
        overlay = overlays[panel_idx] if overlays is not None else None

        for y in range(ny):
            src_row = panel[ny - 1 - y]
            overlay_row = overlay[ny - 1 - y] if overlay is not None else None
            for x in range(nx):
                # Compress dynamic range so low-amplitude far-field structure remains visible.
                color = color_for(math.tanh(1.8 * src_row[x] / scale))
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
) -> Path:
    output = ASSETS / name
    ASSETS.mkdir(parents=True, exist_ok=True)

    scale = max_abs_in_sequence(frame_panels)

    with tempfile.TemporaryDirectory(prefix="wavefront_gif_") as tmp:
        tmp_path = Path(tmp)
        for idx, panels in enumerate(frame_panels):
            width, height, rows = render_frame(
                panels,
                scale,
                layout,
                stripe_colors,
                overlays=overlays,
            )
            frame_path = tmp_path / f"frame_{idx:04d}.png"
            write_png_rgb(frame_path, width, height, rows)

        encode_gif_from_frames(tmp_path, fps=fps, output=output)

    return output


def scenario_modes_evolution() -> Path:
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
        frames_data.append([panel])

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
        stripe_colors=[(119, 42, 146)],
        overlays=[overlay],
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


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to build GIFs; please install it and retry.")


def main() -> None:
    require_ffmpeg()
    ASSETS.mkdir(parents=True, exist_ok=True)
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        scenario_modes_evolution(),
        scenario_interface_reflection(),
        scenario_boundary_comparison(),
        scenario_double_slit(),
        scenario_3d_volume(),
    ]

    metrics_path = CHECKS_DIR / "validation-metrics.json"
    metrics_path.write_text(json.dumps(VALIDATION_SUMMARY, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for output in outputs:
        size_kb = output.stat().st_size / 1024.0
        print(f"wrote {output} ({size_kb:.1f} KiB)")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
