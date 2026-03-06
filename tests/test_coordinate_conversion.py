import asyncio
from pathlib import Path
import sys

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gpu_load_balancer import GpuLoadBalancerService  # noqa: E402
import main  # noqa: E402
from utils import CoordinateConverter, CoordinateReference  # noqa: E402


class FakeSionna:
    def __init__(self):
        self.scene_loaded = False
        self.transmitters = {}
        self.receivers = {}
        self._computed_paths = None

    def load_simulation_scene(self, scene_path=None):
        self.scene_loaded = True

    def reset(self):
        self.transmitters.clear()
        self.receivers.clear()
        self._computed_paths = None

    def add_transmitter(self, name, position, orientation=None):
        self.transmitters[name] = {"position": position, "orientation": orientation}

    def add_receiver(self, name, position, orientation=None):
        self.receivers[name] = {"position": position, "orientation": orientation}

    def update_ant_position(self, ant_type, name, position):
        if name in self.transmitters:
            self.transmitters[name]["position"] = position
            return
        if name in self.receivers:
            self.receivers[name]["position"] = position
            return
        raise ValueError("Antenna not found")

    def get_scene_info(self):
        return {
            "object_count": 0,
            "objects": [],
            "transmitter_count": len(self.transmitters),
            "receiver_count": len(self.receivers),
        }

    def compute_paths(self, max_depth=3):
        return {"path_count": 1, "max_depth": max_depth}

    def get_channel_impulse_response(self):
        return {
            "delays": [],
            "gains": {"real": [], "imag": [], "magnitude": [], "phase": []},
            "shape": {
                "num_rx": 0,
                "num_rx_ant": 0,
                "num_tx": 0,
                "num_tx_ant": 0,
                "num_paths": 0,
                "num_time_steps": 0,
            },
        }


def _assert_tuple_close(lhs, rhs, tol=1e-6):
    assert len(lhs) == len(rhs)
    for left, right in zip(lhs, rhs):
        assert left == pytest.approx(right, abs=tol)


def test_round_trip_geo_local_geo():
    converter = CoordinateConverter(
        CoordinateReference(lat=35.727, lon=-78.696, alt=110.0)
    )
    source_geo = (35.727321, -78.695612, 125.25)
    local = converter.lat_lon_alt_to_local(*source_geo)
    recovered_geo = converter.local_to_lat_lon_alt(*local)
    _assert_tuple_close(recovered_geo, source_geo, tol=1e-6)


def test_main_uses_geo_to_local_conversion(monkeypatch):
    converter = CoordinateConverter(
        CoordinateReference(lat=35.727, lon=-78.696, alt=110.0)
    )
    monkeypatch.setattr(main, "Sionna", FakeSionna)
    monkeypatch.setattr(main, "factory", main.SionnaFactory())
    monkeypatch.setattr(main, "coordinate_converter", converter)
    monkeypatch.setattr(main, "gpu_dispatcher", None)

    async def _run():
        dispatcher = GpuLoadBalancerService(gpu_ids=["0"])
        monkeypatch.setattr(main, "gpu_dispatcher", dispatcher)
        await dispatcher.start()
        try:
            scene_id = await main.create_scene()
            geo_position = (35.727321, -78.695612, 125.25)
            response = await main.add_transmitter(
                scene_id, "tx-check", geo_position, None
            )

            engine = main.factory.get_scene(scene_id)
            local_position = engine.transmitters["tx-check"]["position"]
            expected_local = converter.lat_lon_alt_to_local(*geo_position)

            _assert_tuple_close(local_position, expected_local, tol=1e-6)
            _assert_tuple_close(response["position"], geo_position, tol=1e-6)
        finally:
            await main.shutdown()

    asyncio.run(_run())
