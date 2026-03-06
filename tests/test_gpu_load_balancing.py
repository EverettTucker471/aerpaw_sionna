import asyncio
import time
from pathlib import Path
import sys

from fastapi.testclient import TestClient


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import app as app_module  # noqa: E402
from gpu_load_balancer import GpuLoadBalancerService  # noqa: E402
import main  # noqa: E402


def test_gpu_selector_is_stable_for_scene_id():
    service = GpuLoadBalancerService(gpu_ids=["0", "1", "2"])
    first = service.select_gpu_id("scene-a")
    second = service.select_gpu_id("scene-a")
    third = service.select_gpu_id("scene-b")

    assert first == second
    assert first in {"0", "1", "2"}
    assert third in {"0", "1", "2"}


def test_dispatch_awaits_and_returns_result():
    async def _run():
        service = GpuLoadBalancerService(gpu_ids=["0", "1"])
        await service.start()
        try:
            start = time.perf_counter()
            result = await service.dispatch(
                "scene-a", lambda: (time.sleep(0.08), "done")[1]
            )
            elapsed = time.perf_counter() - start
        finally:
            await service.shutdown()

        assert result == "done"
        assert elapsed >= 0.07

    asyncio.run(_run())


def test_simulation_route_blocks_until_gpu_job_finishes(monkeypatch):
    async def delayed_compute(scene_id: str, max_depth: int = 3):
        await asyncio.sleep(0.1)
        return {"path_count": 7, "max_depth": max_depth}

    async def create_scene_stub(scene_path=None):
        return "scene-delay"

    monkeypatch.setattr(main, "compute_paths", delayed_compute)
    monkeypatch.setattr(main, "create_scene", create_scene_stub)

    with TestClient(app_module.app) as client:
        create_response = client.post("/scenes", json={})
        assert create_response.status_code == 201

        start = time.perf_counter()
        response = client.post(
            "/scenes/scene-delay/simulation/paths", json={"max_depth": 2}
        )
        elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert response.json()["path_count"] == 7
    assert elapsed >= 0.09
