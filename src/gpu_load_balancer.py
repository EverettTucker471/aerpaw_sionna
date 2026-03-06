import asyncio
import hashlib
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional


@dataclass
class _QueuedJob:
    call: Callable[[], Any]
    future: asyncio.Future


class GpuLoadBalancerService:
    """
    Dispatch scene jobs onto GPU-specific queues.

    Jobs for the same scene id always resolve to the same queue (scene affinity).
    """

    def __init__(self, gpu_ids: Optional[List[str]] = None):
        self._gpu_ids = gpu_ids or ["0"]
        if not self._gpu_ids:
            self._gpu_ids = ["0"]

        self._queues: List[asyncio.Queue] = [
            asyncio.Queue() for _ in range(len(self._gpu_ids))
        ]
        self._workers: List[asyncio.Task] = []
        self._started = False
        self._sentinel = object()

    @property
    def gpu_ids(self) -> List[str]:
        return list(self._gpu_ids)

    def _queue_index_for_scene(self, scene_id: str) -> int:
        digest = hashlib.sha256(scene_id.encode("utf-8")).digest()
        hash_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return hash_int % len(self._queues)

    def select_gpu_id(self, scene_id: str) -> str:
        return self._gpu_ids[self._queue_index_for_scene(scene_id)]

    async def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._workers = [
            asyncio.create_task(
                self._worker(queue_index),
                name=f"gpu-queue-worker-{self._gpu_ids[queue_index]}",
            )
            for queue_index in range(len(self._queues))
        ]

    async def shutdown(self) -> None:
        if not self._started:
            return

        for queue in self._queues:
            await queue.put(self._sentinel)

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def dispatch(
        self, scene_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if not self._started:
            raise RuntimeError("GPU load balancer service is not running")

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        job = _QueuedJob(call=partial(fn, *args, **kwargs), future=future)
        queue_index = self._queue_index_for_scene(scene_id)
        await self._queues[queue_index].put(job)
        return await future

    async def _worker(self, queue_index: int) -> None:
        queue = self._queues[queue_index]
        while True:
            item = await queue.get()
            if item is self._sentinel:
                queue.task_done()
                break

            job: _QueuedJob = item
            try:
                result = await asyncio.to_thread(job.call)
            except Exception as exc:
                if not job.future.cancelled():
                    job.future.set_exception(exc)
            else:
                if not job.future.cancelled():
                    job.future.set_result(result)
            finally:
                queue.task_done()
