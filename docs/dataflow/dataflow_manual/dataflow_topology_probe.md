# Dataflow GPU Topology Probe

This document describes the host-side GPU topology probe exposed by
`tilelang.dataflow`.

## Scope

The migrated code provides host-side GPU topology measurement utilities for
Dataflow experiments:

- `tilelang.dataflow.get_gpu_gpc_sm_groups(...)` measures GPC SM groups.
- `tilelang.dataflow.get_gpu_die_sm_groups(...)` measures die-level SM groups.
- `tilelang.dataflow.probe_gpu_topology(...)` combines both views.
- `python -m tilelang.dataflow.hardware_topology --gpc-only --json` prints a
  GPC-only report without running the die probe.

The Dataflow scheduler treats `GPUTopology.cluster_size` as
the CUDA thread-block cluster size, and in-kernel coordination should continue
to use CUDA's block rank inside a cluster. Hardware GPC groups are analysis
metadata; a GPC is not a CUDA cluster.

## Usage

GPC-only analysis:

```bash
python -m tilelang.dataflow.hardware_topology --gpc-only --json
```

Combined die and GPC analysis:

```bash
python -m tilelang.dataflow.hardware_topology --json
```

The GPC probe requires `nvcc` and a CUDA runtime with thread-block-cluster launch
support. The die probe requires CuPy because it JIT-compiles the latency probes.
