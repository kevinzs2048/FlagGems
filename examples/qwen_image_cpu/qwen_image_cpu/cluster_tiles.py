"""Experimental CPU-cluster-dependent GEMM work claims; no default dispatch."""

import plistlib
import subprocess
from functools import lru_cache

import torch


def read_topology():
    raw = subprocess.check_output(
        ["/usr/sbin/ioreg", "-a", "-l", "-p", "IODeviceTree", "-r", "-n", "cpus"]
    )
    roots = plistlib.loads(raw)
    nodes = [node for node in roots if node.get("IORegistryEntryName") == "cpus"]
    if len(nodes) != 1:
        raise RuntimeError("Expected one cpus node in IORegistry")
    cpus = []
    for child in nodes[0].get("IORegistryEntryChildren", []):
        if "logical-cpu-id" not in child:
            continue
        cpu, cluster = child["logical-cpu-id"], child.get("logical-cluster-id")
        if type(cpu) is not int or type(cluster) is not int or not 0 <= cluster < 32:
            raise RuntimeError("Invalid logical CPU/cluster metadata")
        cpus.append({"cpu": cpu, "cluster": cluster})
    count = int(
        subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.ncpu"], text=True)
    )
    cpus.sort(key=lambda row: row["cpu"])
    if [row["cpu"] for row in cpus] != list(range(count)):
        raise RuntimeError("CPU topology does not cover all logical CPUs exactly once")
    mapping = torch.tensor([row["cluster"] for row in cpus], dtype=torch.long)
    return mapping, {"source": "ioreg cpus + sysctl hw.ncpu", "cpus": cpus}


@lru_cache(None)
def ops():
    from flag_gems.runtime.backend._arm.quantized_linear.image_sme import load_ops

    return load_ops("cluster_tiles")
