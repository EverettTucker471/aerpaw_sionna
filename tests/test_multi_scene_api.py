from pathlib import Path
import sys

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import app as app_module  # noqa: E402
import main  # noqa: E402
from utils import AntennaType, CoordinateConverter, CoordinateReference  # noqa: E402


class FakeSionna:
    def __init__(self):
        self.scene_loaded = False
        self.transmitters = {}
        self.receivers = {}
        self._computed_paths = None

    def load_simulation_scene(self, scene_path=None):
        self.scene_loaded = True
        self.scene_path = scene_path

    def get_scene_info(self):
        if not self.scene_loaded:
            raise RuntimeError("No scene loaded")
        return {
            "object_count": 0,
            "objects": [],
            "transmitter_count": len(self.transmitters),
            "receiver_count": len(self.receivers),
        }

    def reset(self):
        self.transmitters.clear()
        self.receivers.clear()
        self._computed_paths = None

    def add_transmitter(self, name, position, orientation=None):
        if not self.scene_loaded:
            raise RuntimeError("Scene not loaded")
        self.transmitters[name] = {"position": position, "orientation": orientation}

    def add_receiver(self, name, position, orientation=None):
        if not self.scene_loaded:
            raise RuntimeError("Scene not loaded")
        self.receivers[name] = {"position": position, "orientation": orientation}

    def set_array(self, *args, **kwargs):
        return None

    def update_ant_position(self, ant_type, name, position):
        if ant_type == AntennaType.Transmitter:
            if name not in self.transmitters:
                raise ValueError(f"Transmitter '{name}' not found")
            self.transmitters[name]["position"] = position
            return

        if ant_type == AntennaType.Receiver:
            if name not in self.receivers:
                raise ValueError(f"Receiver '{name}' not found")
            self.receivers[name]["position"] = position
            return

        raise RuntimeError("Invalid Antenna Type")

    def compute_paths(self, max_depth=3):
        if not self.transmitters or not self.receivers:
            raise RuntimeError("No transmitters or receivers in scene")
        self._computed_paths = {
            "num_rx": len(self.receivers),
            "num_tx": len(self.transmitters),
            "num_paths": max(1, len(self.receivers) * len(self.transmitters)),
        }
        return {"path_count": self._computed_paths["num_paths"], "max_depth": max_depth}

    def get_channel_impulse_response(self):
        if self._computed_paths is None:
            raise RuntimeError("Paths not computed")

        num_rx = self._computed_paths["num_rx"]
        num_tx = self._computed_paths["num_tx"]
        num_paths = self._computed_paths["num_paths"]
        zeros = [[[[[0.0 for _ in range(num_paths)] for _ in range(num_tx)] for _ in range(num_rx)]]]

        return {
            "delays": zeros,
            "gains": {
                "real": zeros,
                "imag": zeros,
                "magnitude": zeros,
                "phase": zeros,
            },
            "shape": {
                "num_rx": num_rx,
                "num_rx_ant": 1,
                "num_tx": num_tx,
                "num_tx_ant": 1,
                "num_paths": num_paths,
                "num_time_steps": 1,
            },
        }


@pytest.fixture
def patched_backend(monkeypatch):
    monkeypatch.setattr(main, "Sionna", FakeSionna)
    monkeypatch.setattr(main, "factory", main.SionnaFactory())
    monkeypatch.setattr(main, "gpu_dispatcher", None)
    monkeypatch.setattr(
        main,
        "coordinate_converter",
        CoordinateConverter(CoordinateReference(lat=35.727, lon=-78.696, alt=110.0)),
    )
    return main.factory


@pytest.fixture
def client(patched_backend):
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_routes_are_scene_scoped():
    routes = [route for route in app_module.app.routes if isinstance(route, APIRoute)]
    paths = {route.path for route in routes}

    allowed_non_scene_paths = {"/", "/scenes"}
    assert allowed_non_scene_paths.issubset(paths)

    scoped_paths = paths - allowed_non_scene_paths
    assert scoped_paths
    assert all("{scene_id}" in path for path in scoped_paths)


def test_create_scene_returns_unique_scene_ids(client):
    response_a = client.post("/scenes", json={})
    response_b = client.post("/scenes", json={})

    assert response_a.status_code == 201
    assert response_b.status_code == 201

    scene_a = response_a.json()["scene_id"]
    scene_b = response_b.json()["scene_id"]

    assert scene_a
    assert scene_b
    assert scene_a != scene_b

    scene_info = client.get(f"/scenes/{scene_a}")
    assert scene_info.status_code == 200
    assert scene_info.json()["coordinate_reference"] == {
        "lat": 35.727,
        "lon": -78.696,
        "alt": 110.0,
    }


def test_scene_updates_are_isolated_by_scene_id(client):
    scene_a = client.post("/scenes", json={}).json()["scene_id"]
    scene_b = client.post("/scenes", json={}).json()["scene_id"]

    add_tx_response = client.post(
        f"/scenes/{scene_a}/transmitters",
        json={"name": "tx1", "position": {"lat": 35.0, "lon": -78.0, "alt": 110.0}},
    )
    assert add_tx_response.status_code == 201

    tx_scene_a = client.get(f"/scenes/{scene_a}/transmitters")
    tx_scene_b = client.get(f"/scenes/{scene_b}/transmitters")

    assert tx_scene_a.status_code == 200
    assert tx_scene_b.status_code == 200
    assert tx_scene_a.json() == ["tx1"]
    assert tx_scene_b.json() == []


def test_factory_creates_multiple_independent_instances(monkeypatch):
    monkeypatch.setattr(main, "Sionna", FakeSionna)
    factory = main.SionnaFactory()

    scene_a = factory.create_scene()
    scene_b = factory.create_scene()

    assert scene_a != scene_b

    engine_a = factory.get_scene(scene_a)
    engine_b = factory.get_scene(scene_b)
    assert engine_a is not engine_b

    engine_a.load_simulation_scene()
    engine_b.load_simulation_scene()
    engine_a.add_transmitter("tx-a", (0.0, 0.0, 10.0))
    assert list(engine_a.transmitters.keys()) == ["tx-a"]
    assert list(engine_b.transmitters.keys()) == []
