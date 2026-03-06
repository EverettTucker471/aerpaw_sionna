import os
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from gpu_load_balancer import GpuLoadBalancerService
from sionna_wrapper import Sionna
from utils import (
    AntennaType,
    CoordinateConverter,
    PolarizationType,
    RadiationPattern,
)


class SceneNotFoundError(ValueError):
    """Raised when a scene id is not managed by the factory."""


class SionnaFactory:
    """Factory and registry for independent Sionna scene instances."""

    def __init__(self) -> None:
        self._instances: Dict[str, Sionna] = {}

    def create_scene(self) -> str:
        scene_id = str(uuid4())
        self._instances[scene_id] = Sionna()
        return scene_id

    def get_scene(self, scene_id: str) -> Sionna:
        engine = self._instances.get(scene_id)
        if engine is None:
            raise SceneNotFoundError(f"Scene '{scene_id}' not found")
        return engine

    def delete_scene(self, scene_id: str) -> None:
        self._instances.pop(scene_id, None)

    def shutdown(self) -> None:
        for engine in self._instances.values():
            engine.reset()
        self._instances.clear()


factory = SionnaFactory()
gpu_dispatcher: Optional[GpuLoadBalancerService] = None
coordinate_converter = CoordinateConverter.from_env()


def _configured_gpu_ids() -> List[str]:
    raw_gpu_ids = os.getenv("SIONNA_GPU_IDS", "0")
    parsed = [gpu.strip() for gpu in raw_gpu_ids.split(",") if gpu.strip()]
    return parsed or ["0"]


def _require_dispatcher() -> GpuLoadBalancerService:
    if gpu_dispatcher is None:
        raise RuntimeError("GPU load balancer is not initialized")
    return gpu_dispatcher


async def _dispatch(scene_id: str, fn, *args, **kwargs):
    return await _require_dispatcher().dispatch(scene_id, fn, *args, **kwargs)


def _geo_to_local(position: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return coordinate_converter.lat_lon_alt_to_local(*position)


def _local_to_geo(position: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return coordinate_converter.local_to_lat_lon_alt(*position)


async def initialize() -> None:
    """Initialize backend resources and GPU queue workers."""
    global gpu_dispatcher
    if gpu_dispatcher is not None:
        return

    gpu_dispatcher = GpuLoadBalancerService(gpu_ids=_configured_gpu_ids())
    await gpu_dispatcher.start()


async def shutdown() -> None:
    """Shutdown GPU workers and clean up all scene instances."""
    global gpu_dispatcher

    if gpu_dispatcher is not None:
        await gpu_dispatcher.shutdown()
        gpu_dispatcher = None

    factory.shutdown()


async def create_scene(scene_path: Optional[str] = None) -> str:
    """Create, load, and register a new scene instance."""
    scene_id = factory.create_scene()
    engine = factory.get_scene(scene_id)
    try:
        await _dispatch(scene_id, engine.load_simulation_scene, scene_path)
    except Exception:
        factory.delete_scene(scene_id)
        raise
    return scene_id


async def get_scene_info(scene_id: str) -> Dict:
    """Get information about a specific scene."""
    engine = factory.get_scene(scene_id)
    scene_info = await _dispatch(scene_id, engine.get_scene_info)
    scene_info["coordinate_reference"] = {
        "lat": coordinate_converter.reference.lat,
        "lon": coordinate_converter.reference.lon,
        "alt": coordinate_converter.reference.alt,
    }
    return scene_info


async def reset_scene(scene_id: str) -> None:
    """Reset a specific scene to initial state."""
    engine = factory.get_scene(scene_id)
    await _dispatch(scene_id, engine.reset)


async def add_transmitter(
    scene_id: str,
    name: str,
    position: Tuple[float, float, float],
    signal_power: float,
    velocity: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0),
    orientation: Optional[Tuple[float, float, float]] = None,
) -> Dict:
    """Add a transmitter to a specific scene."""
    engine = factory.get_scene(scene_id)
    local_position = _geo_to_local(position)

    def _add() -> Dict:
        engine.add_transmitter(name, local_position, orientation)
        result = {"name": name, "position": _local_to_geo(local_position)}
        if orientation:
            result["orientation"] = orientation
        return result

    return await _dispatch(scene_id, _add)


async def update_transmitter_position(
    scene_id: str, name: str, position: Tuple[float, float, float]
) -> Dict:
    """Update the position of an existing transmitter in a scene."""
    engine = factory.get_scene(scene_id)
    local_position = _geo_to_local(position)

    def _update() -> Dict:
        engine.update_ant_position(AntennaType.Transmitter, name, local_position)
        return {"name": name, "position": _local_to_geo(local_position)}

    return await _dispatch(scene_id, _update)


async def get_transmitters(scene_id: str) -> List[str]:
    """Get list of all transmitter names in a scene."""
    engine = factory.get_scene(scene_id)
    return await _dispatch(scene_id, lambda: list(engine.transmitters.keys()))


async def add_receiver(
    scene_id: str,
    name: str,
    position: Tuple[float, float, float],
    velocity: Tuple[float, float, float],
    orientation: Optional[Tuple[float, float, float]] = None,
) -> Dict:
    """Add a receiver to a specific scene."""
    engine = factory.get_scene(scene_id)
    local_position = _geo_to_local(position)

    def _add() -> Dict:
        engine.add_receiver(name, local_position, orientation)
        result = {"name": name, "position": _local_to_geo(local_position)}
        if orientation:
            result["orientation"] = orientation
        return result

    return await _dispatch(scene_id, _add)


async def update_receiver_position(
    scene_id: str, name: str, position: Tuple[float, float, float]
) -> Dict:
    """Update the position of an existing receiver in a scene."""
    engine = factory.get_scene(scene_id)
    local_position = _geo_to_local(position)

    def _update() -> Dict:
        engine.update_ant_position(AntennaType.Receiver, name, local_position)
        return {"name": name, "position": _local_to_geo(local_position)}

    return await _dispatch(scene_id, _update)


async def get_receivers(scene_id: str) -> List[str]:
    """Get list of all receiver names in a scene."""
    engine = factory.get_scene(scene_id)
    return await _dispatch(scene_id, lambda: list(engine.receivers.keys()))


async def set_array(
    scene_id: str,
    ant_type: str,
    num_rows_cols: Tuple[int, int],
    vertical_horizontal_spacing: Tuple[float, float],
    pattern: str,
    polarization: str,
) -> Dict:
    """
    Set antenna array configuration for transmitter or receiver.

    Args:
        scene_id: Unique scene identifier.
        ant_type: Antenna type ('tx' or 'rx')
        num_rows_cols: Tuple of (num_rows, num_cols)
        vertical_horizontal_spacing: Tuple of (vertical_spacing, horizontal_spacing)
        pattern: Radiation pattern ('iso', 'dipole', or 'tr38901')
        polarization: Polarization type ('V', 'H', or 'cross')

    Returns:
        Dictionary with array configuration details
    """
    try:
        antenna_enum = AntennaType(ant_type)
    except ValueError:
        raise ValueError(f"Invalid antenna type: {ant_type}. Must be 'tx' or 'rx'")

    try:
        RadiationPattern(pattern)
    except ValueError:
        raise ValueError(
            f"Invalid pattern: {pattern}. Must be 'iso', 'dipole', or 'tr38901'"
        )

    try:
        PolarizationType(polarization)
    except ValueError:
        raise ValueError(
            f"Invalid polarization: {polarization}. Must be 'V', 'H', or 'cross'"
        )

    num_rows, num_cols = num_rows_cols
    vertical_spacing, horizontal_spacing = vertical_horizontal_spacing
    engine = factory.get_scene(scene_id)

    await _dispatch(
        scene_id,
        engine.set_array,
        antenna_enum,
        num_rows,
        num_cols,
        vertical_spacing,
        horizontal_spacing,
        pattern,
        polarization,
    )

    return {
        "antenna_type": ant_type,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "vertical_spacing": vertical_spacing,
        "horizontal_spacing": horizontal_spacing,
        "pattern": pattern,
        "polarization": polarization,
    }


async def compute_paths(scene_id: str, max_depth: int = 3) -> Dict:
    """Compute propagation paths between transmitters and receivers in a scene."""
    engine = factory.get_scene(scene_id)
    return await _dispatch(scene_id, engine.compute_paths, max_depth)


async def get_cir(scene_id: str) -> Dict:
    """Get the Channel Impulse Response for a scene."""
    engine = factory.get_scene(scene_id)
    return await _dispatch(scene_id, engine.get_channel_impulse_response)
