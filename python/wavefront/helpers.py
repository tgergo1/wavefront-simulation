"""High-level helper builders for wavefront-simulation."""

from ._wavefront import GeometryRegion, GeometryShape, MediumLaw, ProbeMonitorSpec, SurfaceMonitorSpec, SymbolicExpr


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
