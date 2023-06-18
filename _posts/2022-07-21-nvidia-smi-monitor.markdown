---
layout: post
title: Monitoring Background GPU Usage
category: 💻
excerpt: Python code snippit for doing background GPU monitoring using `nvidia-smi`
---

I was recently trying to log my GPU usage to Tensorboard for a training job and found that calling `nvidia-smi` on every step, or at some interval, was eating up my performance. I rewrote it into a clean daemon process and thought I should share it here for other people who might be in the same situation.

```python
import logging
import multiprocessing as mp
import os
import re
import subprocess as sp
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)

NUMBER_REGEX = re.compile(r"[\d\.]+")

FIELDS = ["index", "memory.total", "memory.used", "temperature.gpu", "utilization.gpu"]


@dataclass(frozen=True)
class GPUStats:
    index: int
    memory_used: float
    temperature: float
    gpu_utilization: float


def parse_number(s: str) -> float:
    match = NUMBER_REGEX.search(s)
    if match is None:
        raise ValueError(s)
    return float(match.group())


def parse_gpu_stats(row: str) -> GPUStats:
    cols = row.split(",")
    index = int(cols[0].strip())
    (
        memory_total,
        memory_used,
        temperature,
        gpu_utilization,
    ) = (parse_number(col) for col in cols[1:])

    return GPUStats(
        index=index,
        memory_used=100 * memory_used / memory_total,
        temperature=temperature,
        gpu_utilization=gpu_utilization,
    )


def gen_gpu_stats(loop_secs: int = 5) -> Iterable[GPUStats]:
    fields = ",".join(FIELDS)

    # This command loops `nvidia-smi` forever, yielding rows of the requested
    # fields, which are then parsed to the dataclass above.
    command = f"nvidia-smi --query-gpu={fields} --format=csv --loop={loop_secs}"

    with sp.Popen(command.split(), stdout=sp.PIPE, universal_newlines=True) as proc:
        stdout = proc.stdout
        assert stdout is not None
        rows = iter(stdout.readline, "")
        for row in rows:
            try:
                yield parse_gpu_stats(row)
            except ValueError:  # Handles the first row and misc errors.
                continue


def gpu_stats_worker(ping_interval: int, queue: "mp.Queue[GPUStats]") -> None:
    # Simply iterates from the coroutine and puts items into the queue.
    for gpu_stat in gen_gpu_stats(ping_interval):
        queue.put(gpu_stat)
```

Then, from your training loop, you can do something like this:

```python
import atexit
from typing import Dict

gpu_stats: Dict[int, GPUStats] = {}

# Starts the `nvidia-smi` process.
queue: "mp.Queue[GPUStats]" = mp.Queue()
proc = mp.Process(target=gpu_stats_worker, args=(config, queue), daemon=True)
proc.start()

# This shouldn't be necessary, but just for fun.
atexit.register(proc.kill)

for batch in dataloader:

    ...

    # Gets any new GPU stats
    while not queue.empty():
        gpu_stat: GPUStats = queue.get()
        gpu_stats[gpu_stat.index] = gpu_stat

    # Logs statistics to Tensorboard
    for gpu_stat in self.gpu_stats.values():
        tb_writer.add_scalar(f"gpu-{gpu_stat.index}/mem_used", gpu_stat.memory_used)
        tb_writer.add_scalar(f"gpu-{gpu_stat.index}/temp", gpu_stat.temperature)
        tb_writer.add_scalar(f"gpu-{gpu_stat.index}/gpu_util", gpu_stat.gpu_utilization)
```

Hope this helps!