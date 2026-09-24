"""Streaming Dask executor: accumulate each chunk as it completes.

Alternative to coffea's DaskExecutor for the merge-OOM problem: DaskExecutor
tree-reduces chunk outputs *on the workers*, so with many-axis histograms the
partial sums grow until small condor workers die at the end of the run. Here
workers never merge anything -- each finished chunk's output is pulled back to
the client, added in place to the running total, and its distributed memory is
released. Peak merge memory is (running total + one chunk) on the client,
independent of worker memory.

Pattern borrowed from Pepper's ResumableExecutor (pepper/executor.py); this is
the accumulation part only (no state file / resume).

Works with coffea.processor.Runner exactly like DaskExecutor:

    executor=StreamingDaskExecutor(client=client, retries=12, status=True)
"""

from dataclasses import dataclass
from typing import Optional

from coffea.processor.accumulator import iadd
from coffea.processor.executor import (
    ExecutorBase,
    _compression_wrapper,
    _decompress,
)


@dataclass
class StreamingDaskExecutor(ExecutorBase):
    client: Optional["distributed.Client"] = None  # noqa: F821
    retries: int = 3
    priority: int = 0

    def __getstate__(self):
        return dict(self.__dict__, client=None)

    def __call__(self, items, function, accumulator):
        from distributed import as_completed
        from distributed.scheduler import KilledWorker

        items = list(items)
        if len(items) == 0:
            return accumulator, 0

        if self.compression is not None:
            function = _compression_wrapper(
                self.compression, function, name=self.function_name)

        futures = self.client.map(
            function, items,
            pure=False,
            priority=self.priority,
            retries=self.retries,
        )
        key_to_item = {f.key: item for f, item in zip(futures, items)}

        progress = None
        if self.status:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(futures), desc=self.desc, unit=self.unit)

        try:
            for future in as_completed(futures):
                try:
                    result = future.result()
                except KilledWorker as ex:
                    raise RuntimeError(
                        f"Work item {key_to_item.get(ex.task, ex.task)} caused "
                        "a KilledWorker exception (likely a segfault or "
                        "out-of-memory issue)") from ex
                future.release()
                if self.compression is not None:
                    result = _decompress(result)
                if accumulator is None:
                    accumulator = result
                else:
                    iadd(accumulator, result)
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        return accumulator, 0
