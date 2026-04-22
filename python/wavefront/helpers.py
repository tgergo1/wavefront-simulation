"""High-level helper builders for wavefront-simulation."""

from ._wavefront import (
    CollisionMonitorSpec,
    GeometryRegion,
    GeometryShape,
    MediumLaw,
    ProbeMonitorSpec,
    SurfaceMonitorSpec,
    SymbolicExpr,
    WaveSourceSpec,
)


def builtin_material(name: str) -> MediumLaw:
    presets = {
        "air": ("1.225", "0.014", "0.0001", "0.0"),
        "water": ("1000.0", "2.25e9", "0.001", "0.0"),
        "glass": ("2500.0", "7.0e10", "0.0005", "0.0"),
        "steel": ("7850.0", "2.0e11", "0.0002", "0.0"),
        "honey": ("1420.0", "3.2e9", "0.15", "0.02"),
        "hyperhoney": ("1550.0", "3.8e9", "0.22", "0.08"),
        "oobleck": ("1650.0", "4.6e9", "0.35", "0.05"),
        "aerogel": ("120.0", "1.5e7", "0.03", "0.01"),
        "ferrofluid": ("1210.0", "1.8e9", "0.09", "0.12"),
        "plasma": ("0.18", "8.0e4", "0.005", "0.25"),
        "metamaterial": ("950.0", "7.5e8", "0.12", "0.4"),
        "neutron_star_crust": ("4.0e17", "1.0e29", "1.0e-6", "0.001"),
        "strange_matter": ("8.0e17", "3.2e29", "5.0e-7", "0.02"),
    }
    density, stiffness, damping, dispersion = presets[name]
    medium = MediumLaw()
    medium.density = SymbolicExpr(density)
    medium.stiffness = SymbolicExpr(stiffness)
    medium.damping = SymbolicExpr(damping)
    medium.dispersion = SymbolicExpr(dispersion)
    return medium


def make_layer_region(name: str, axis: int, lower: float, upper: float, material: MediumLaw) -> GeometryRegion:
    region = GeometryRegion()
    region.name = name
    region.shape = GeometryShape.Layer
    region.axis = axis
    region.lower = lower
    region.upper = upper
    region.medium = material
    return region


def make_polygon_region(name: str, vertices: list[float], material: MediumLaw) -> GeometryRegion:
    region = GeometryRegion()
    region.name = name
    region.shape = GeometryShape.Polygon
    region.vertices = vertices
    region.medium = material
    return region


def make_sdf_region(name: str, signed_distance: str, material: MediumLaw) -> GeometryRegion:
    region = GeometryRegion()
    region.name = name
    region.shape = GeometryShape.SignedDistanceField
    region.signed_distance = SymbolicExpr(signed_distance)
    region.medium = material
    return region


def make_koch_snowflake_region(
    name: str,
    center: list[float],
    radius: float,
    iterations: int,
    material: MediumLaw,
    scale: float = 1.0,
) -> GeometryRegion:
    region = GeometryRegion()
    region.name = name
    region.shape = GeometryShape.Fractal
    region.center = center
    region.radius = radius
    region.fractal_generator = "koch_snowflake"
    region.fractal_iterations = iterations
    region.fractal_scale = scale
    region.medium = material
    return region


def make_probe_monitor(name: str, index: list[int], component: int = 0, capture_complex: bool = False) -> ProbeMonitorSpec:
    monitor = ProbeMonitorSpec()
    monitor.name = name
    monitor.index = index
    monitor.component = component
    monitor.capture_complex = capture_complex
    return monitor


def make_surface_monitor(name: str, axis: int, upper_face: bool, component: int = 0) -> SurfaceMonitorSpec:
    monitor = SurfaceMonitorSpec()
    monitor.name = name
    monitor.axis = axis
    monitor.upper_face = upper_face
    monitor.component = component
    return monitor


def make_geometry_surface_monitor(name: str, geometry_region: str, component: int = 0, shell_thickness: float = 0.0) -> SurfaceMonitorSpec:
    monitor = SurfaceMonitorSpec()
    monitor.name = name
    monitor.component = component
    monitor.geometry_region = geometry_region
    monitor.shell_thickness = shell_thickness
    return monitor


def make_wave_source(name: str, expression: str, wave_id: str | None = None, wave_class: str | None = None) -> WaveSourceSpec:
    source = WaveSourceSpec()
    source.name = name
    source.wave_id = wave_id or name
    source.wave_class = wave_class or source.wave_id
    source.term = SymbolicExpr(expression)
    return source


def make_collision_monitor(
    name: str,
    axis: int,
    upper_face: bool,
    component: int = 0,
    threshold: float = 0.0,
) -> CollisionMonitorSpec:
    monitor = CollisionMonitorSpec()
    monitor.name = name
    monitor.axis = axis
    monitor.upper_face = upper_face
    monitor.component = component
    monitor.threshold = threshold
    return monitor


def make_geometry_collision_monitor(
    name: str,
    geometry_region: str,
    component: int = 0,
    shell_thickness: float = 0.0,
    threshold: float = 0.0,
) -> CollisionMonitorSpec:
    monitor = CollisionMonitorSpec()
    monitor.name = name
    monitor.component = component
    monitor.geometry_region = geometry_region
    monitor.shell_thickness = shell_thickness
    monitor.threshold = threshold
    return monitor
