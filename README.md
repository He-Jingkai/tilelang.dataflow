# tilelang.dataflow: executing dynamic inputs with fine-grained on-chip communication

`tilelang.dataflow` is a research prototype built on TileLang,
providing a CUDA-side programming and runtime path for dynamic, skewed workloads
with cross-SM dependencies. Its public APIs include
`T.dataflow_program`, while the implementation lives under
`tilelang/dataflow/`.

This guide explains how a declarative program becomes a CUDA kernel, how dynamic
work is scheduled, and how cross-SM communication is implemented on NVIDIA
GPUs.

Refer to [`examples/dataflow/mla/mla_decode_non_paged.py`](./examples/dataflow/mla/mla_decode_non_paged.py) and [`examples/dataflow/fusedmoe/example_fusedmoe_dataflow.py`](./examples/dataflow/fusedmoe/example_fusedmoe_dataflow.py) for usage examples.

## 0. Improvements

The following results mirror the latest benchmark snapshot in
`examples/dataflow/current_results.md`, last updated on 2026-08-18.

Lower latency is better. "tilelang.dataflow speedup" is defined uniformly as:

```text
competitor_p50 / tilelang_dataflow_p50 - 1
```

All measurements were collected on the same NVIDIA H100 80GB HBM3 system.
The latest tilelang.dataflow points used one fresh Python process per point, one
measured round, 5 warmups, and 30 samples. The observed clocks were 1980 MHz SM
and 2619 MHz HBM.

### MLA vs FlashMLA

tilelang.dataflow used the generic automatically selected `sm_count=112`,
`cluster_size=16` global-joint schedule at every point.

- tilelang.dataflow metric: `%globaltimer` concurrent global-span p50.
- FlashMLA metric: same-host CUDA-event kernel-span p50.
- FlashMLA baseline revision: `9241ae3ef9bac614dd25e45e507e089f888280e0`.

| Trace | Trace family | Batch | tilelang.dataflow p50 (us) | FlashMLA p50 (us) | tilelang.dataflow speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Dolphin-R1 | 16 | 43.296 | 53.488 | +23.540% |
| 1 | Dolphin-R1 | 32 | 64.304 | 72.016 | +11.993% |
| 2 | Dolphin-R1 | 64 | 105.424 | 114.048 | +8.180% |
| 3 | Dolphin-R1 | 128 | 185.680 | 201.840 | +8.703% |
| 4 | ShareGPT90K | 16 | 88.448 | 94.400 | +6.729% |
| 5 | ShareGPT90K | 32 | 44.048 | 54.032 | +22.666% |
| 6 | ShareGPT90K | 64 | 54.096 | 63.568 | +17.510% |
| 7 | ShareGPT90K | 128 | 92.240 | 103.728 | +12.454% |
| 8 | OpenR1-Math-220k | 16 | 71.408 | 78.368 | +9.747% |
| 9 | OpenR1-Math-220k | 32 | 120.592 | 128.048 | +6.183% |
| 10 | OpenR1-Math-220k | 64 | 293.312 | 306.896 | +4.631% |
| 11 | OpenR1-Math-220k | 128 | 573.504 | 596.032 | +3.928% |
| 12 | OpenThoughts-114k-Code | 16 | 59.696 | 68.720 | +15.117% |
| 13 | OpenThoughts-114k-Code | 32 | 99.392 | 108.160 | +8.822% |
| 14 | OpenThoughts-114k-Code | 64 | 213.104 | 226.864 | +6.457% |
| 15 | OpenThoughts-114k-Code | 128 | 409.520 | 430.848 | +5.208% |

Every tilelang.dataflow point passed reference correctness and two-launch bitwise
determinism.

### Fused MoE vs MegaMoE

The fixed matrix uses FP8, 32 experts, and `top_k=2`:

- DeepSeek: `d_hidden=7168`, `d_expert=2048`;
- Qwen: `d_hidden=2048`, `d_expert=768`.

The tilelang.dataflow scheduler selected topology, transport, and handoff
through generic contracts for each workload. DeepSeek-1024 selected streamed
producer push; the other seven points selected all-gather.

- MegaMoE metric: p50 of 30 historical Kineto/CUPTI kernel durations.
- MegaMoE revision: `23f46aa68c892a349bb7ce331a325e36acceb57e`.

| Model | Tokens | tilelang.dataflow p50 (us) | MegaMoE p50 (us) | tilelang.dataflow speedup |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek | 128 | 483.808 | 491.6010 | +1.611% |
| DeepSeek | 256 | 482.464 | 514.0010 | +6.537% |
| DeepSeek | 512 | 501.664 | 550.8175 | +9.798% |
| DeepSeek | 1024 | 558.992 | 573.9050 | +2.668% |
| Qwen | 128 | 77.616 | 87.8725 | +13.214% |
| Qwen | 256 | 79.696 | 87.1360 | +9.335% |
| Qwen | 512 | 76.224 | 95.6800 | +25.525% |
| Qwen | 1024 | 81.424 | 109.2805 | +34.212% |

Separate fresh-process numerical replay passed all eight workloads;
concurrent and sequential tilelang.dataflow execution were bitwise identical at
every point.

## 1. Project Goals

Modern inference operators receive work whose size and distribution depend on
the current request. Decoding attention sees changing sequence lengths and
batch composition. MoE sees changing token-to-expert routing, including hot
experts and nearly empty experts. Skew within one batch makes both workloads
hard to partition statically.

Conventional implementations split these operators into multiple kernels. MLA
uses split-KV attention followed by reduction/finalization; MoE uses GEMM-1,
GEMM-2, and output aggregation. Kernel boundaries provide correctness, but also
turn imbalance into idle SM time, leave low-parallelism tail phases, move
intermediate results through HBM, and add launch overhead.

tilelang.dataflow expresses the computation as a stage graph and executes its
dependent subtasks through fine-grained SM cooperation.

### 1.1 Execute dynamic and skewed work

The developer declares the logical task domain, dynamic range axes, operator
bodies, and data dependencies. Concrete partitioning and placement are supplied
by an input-dependent schedule rather than embedded in the algorithm.

The separation is visible in the execution path:

- **Handlers** contain statically compiled computation for map, reduce, and
  finalize operations.
- **The plan** describes the current task ranges, CTA placement, communication,
  storage, and handler variants.
- **The wrapper kernel** is a persistent dispatcher. Each CTA consumes its own
  instruction queue and invokes the requested handlers.

This model lets the same operator definition serve different sequence-length
distributions or expert-routing results. The host-side kernel factory builds or
retrieves an artifact specialized for the resulting plan.

### 1.2 Move intermediate data through on-chip paths

When a producer and consumer CTA are in the same thread-block cluster, tilelang.dataflow can
transfer an intermediate directly between their shared-memory address spaces.
The intermediate does not need an HBM round trip. When the dependency crosses a
cluster boundary, the planner uses an HBM-resident exchange slot with explicit
device-scope synchronization.

The transport decision is part of the compiled plan, not the operator body.
Operator code consumes a typed intermediate regardless of its physical route.

### 1.3 Replace kernel barriers with fine-grained synchronization

tilelang.dataflow refines a phase-level dependency into producer-consumer dependencies
between individual subtasks. A consumer waits for the slot it needs, while
other CTAs can continue independent work. Cluster transfers use hardware
`mbarrier` state; HBM transfers use flags and device-scope memory ordering.

Slot lifetime is planned together with synchronization. A producer cannot reuse
a shared-memory source while a remote asynchronous read is still live, and a
destination cannot be overwritten before its previous consumer releases it.

### 1.4 Reuse global input data with multicast loads

When several CTAs in a cluster need the same global-memory tile, a TMA multicast
load can populate the selected shared-memory destinations with one issued
operation. tilelang.dataflow carries multicast and cooperative-input choices through typed
copy, layout, pipeline, and target-capability contracts so that common TileLang
lowering can select a legal implementation.

The design targets NVIDIA GPUs with thread-block clusters and TMA. The cluster
communication templates are guarded for `__CUDA_ARCH__ >= 900`, so the mechanism
is not SM90-only: SM90 and SM100 are selected through target capability
resolution and architecture-specific lowering.

## 2. Execution and Compilation Model

### 2.1 Core entities

| Entity | Meaning |
| --- | --- |
| Task | One coordinate in the declared task domain, such as `(batch, head_block)` or `group_block`. |
| Dynamic range | Input-dependent work attached to a task, such as KV length, expert width, or output hidden width. |
| Stage | A node in the logical stage graph: map/partial, reshared, reduce, or finalize. |
| Handler | A statically lowered CUDA-callable implementation of a stage for a physical specialization. |
| Instruction | One scheduled handler invocation plus task, range, slot, and communication metadata. |
| Queue | The ordered instruction stream assigned to one logical CTA rank. |
| Slot | Planned storage for an intermediate in shared memory, scratch memory, or global staging memory. |
| Communication plan | A transfer and synchronization action connecting source and destination slots. |
| Wrapper | The cluster-launched kernel that drains queues, performs communication, and dispatches handlers. |

The logical and physical layers are deliberately separate:

```text
declarative program
  task domain + dynamic ranges + typed stages
                    |
                    v
input-dependent scheduling and specialization
  CTA queues + ranges + handler variants + transport + lifetimes
                    |
                    v
ordinary TileLang PrimFuncs and operation contracts
                    |
                    v
common TileLang lowering and target selection
                    |
                    v
cluster wrapper + packed runtime plan + lowered handlers
```

The tilelang.dataflow layer owns scheduler facts such as instruction placement, handler
arity, range specialization, and slot ownership. Common TileLang lowering owns
the selection of concrete copy, GEMM, pipeline, and target instructions. This
boundary prevents scheduler-specific facts from leaking into generic CUDA
codegen.

### 2.2 Compilation steps

1. **Capture the declarative graph.** Decorated Python functions become operator
   metadata and `OperatorCall` objects. Calling an operator while constructing a
   graph does not execute its body.
2. **Validate logical contracts.** The frontend checks stage dependencies,
   intermediate types, task arguments, dynamic range axes, and operation
   contracts.
3. **Resolve the target and topology.** Compilation records a capability
   snapshot for the selected GPU and a `GPUTopology` describing SM and cluster
   organization.
4. **Resolve dynamic work.** The kernel factory turns current sequence lengths,
   routed expert groups, or other range metadata into `range_lengths`, optional
   offsets, task coordinates, and task weights.
5. **Schedule subtasks.** The scheduler partitions ranges, assigns work to CTA
   queues, chooses reduction structure or stage-graph placement, and creates
   `InstructionPlan` objects.
6. **Select handler variants.** Range length, physical arity, reduction arity,
   precision, and execution contracts identify the required static handler
   variants.
7. **Plan storage and communication.** The compiler assigns shared/HBM slots,
   proves lifetimes, chooses cluster or HBM transport, inserts barriers and
   release actions, and checks the target's shared-memory budget.
8. **Lower handlers.** Operator bodies become ordinary PrimFuncs, then use the
   common TileLang transformation and CUDA lowering path. Typed operation
   contracts drive GEMM, copy, layout, and pipeline selection.
9. **Pack the runtime plan.** Instructions, per-CTA queue offsets, handler
   arguments, slots, communication records, tensor arguments, and TMA
   descriptors are serialized into the runtime ABI.
10. **Generate and launch the wrapper.** The wrapper initializes communication
    state, drains one queue per CTA, receives required inputs, invokes the
    selected handler, sends produced values, and observes slot-lifetime actions.

The public `@df.jit` path combines these steps with an artifact cache. Factory
arguments and the compile configuration contribute to the cache fingerprint.
A cache hit reuses the compiled plan and handlers; a new dynamic configuration
produces another plan/artifact. Once an artifact is launched, its queue layout
is immutable and the GPU wrapper follows that plan without making host-style
scheduling decisions inside the kernel.

### 2.3 Wrapper execution

Each launched CTA has a logical rank. The rank selects an offset and length in
the flattened queue. The wrapper then repeatedly:

1. loads the next `DataflowInstruction`;
2. decodes task coordinates, dynamic range bounds, input slots, and output slot;
3. issues or waits for receive-side communication;
4. dispatches the instruction's handler variant;
5. issues send-side communication and release actions;
6. advances until `EXIT`.

Receive-side work precedes the handler because its slots are handler inputs.
Send-side work follows handler completion because its source slot is a handler
output. Source and destination reuse waits are inserted according to the
compiler's lifetime proof rather than after every operation.

### 2.4 Dynamic ranges and static handler shapes

Dynamic work does not require generating arbitrary CUDA code at device runtime.
tilelang.dataflow uses a family of statically compiled variants:

- an **exact-length variant** for explicitly selected common lengths;
- a **tile-count variant** for common numbers of range tiles;
- a **generic variant** for all remaining lengths and tails.

`iter_range_buckets="auto"` currently denotes tile counts `1`, `2`, `4`, plus
the generic case. `iter_range_bucket_size` defines the logical size of one tile.
The scheduler records the selected `handler_variant_key` in each instruction,
and the wrapper dispatches the matching handler. Generic handlers receive
`range_begin`, `range_end`, and `dataflow_range_tiles_per_handler()` and mask
partial tails.

This is why a short dynamic range can execute different code from a long range:
the plan can bind a smaller unrolled/specialized handler instead of always
running the maximum-range body. The generic handler remains the correctness
fallback.

### 2.5 Threads, tiles, CTAs, and clusters

The following quantities must not be conflated:

- `block_size` in `DataflowKernelSpec` is a scheduler range-partitioning unit.
  It is not necessarily the number of CUDA threads.
- `threads=` on `@T.dataflow.map`, `@T.dataflow.iter`, `@T.dataflow.reduce`, or
  `@T.dataflow.finalize` declares the operator's physical thread requirement.
- Execution planning may choose `compute_threads` and `consumer_threads` for a
  legal GEMM/pipeline implementation.
- PrimFunc lowering resolves the physical thread count for every handler.
- The wrapper launch width is raised to the maximum resident handler thread
  requirement. A smaller handler runs with its declared thread limit inside
  that wrapper.
- One scheduled CTA occupies one cluster rank. `topology.cluster_size` controls
  the number of cooperating CTA ranks in a thread-block cluster.

Therefore, the programming abstraction does not infer CUDA thread count from a
logical `map` range alone. The operator's physical contract and selected target
implementation determine it, while the wrapper must be wide enough for every
handler it may dispatch.

### 2.6 Source locations for the execution model

| Concern | Source location |
| --- | --- |
| Program/stage graph and validation | [`tilelang/dataflow/program.py`](./tilelang/dataflow/program.py), `DataflowProgram` |
| Operator metadata and decorators | [`tilelang/dataflow/operators.py`](./tilelang/dataflow/operators.py) |
| Dynamic range buckets | [`tilelang/dataflow/iter_range_buckets.py`](./tilelang/dataflow/iter_range_buckets.py) |
| Scheduler IR and queue construction | [`tilelang/dataflow/scheduler.py`](./tilelang/dataflow/scheduler.py), `InstructionPlan` |
| Automatic policy | [`tilelang/dataflow/scheduler_auto_policy.py`](./tilelang/dataflow/scheduler_auto_policy.py) |
| Execution-plan selection | [`tilelang/dataflow/execution_planning.py`](./tilelang/dataflow/execution_planning.py) |
| Compiler orchestration | [`tilelang/dataflow/compiler.py`](./tilelang/dataflow/compiler.py), `_compile_snapshot` |
| Handler Body IR and PrimFunc lowering | [`tilelang/dataflow/body_ir.py`](./tilelang/dataflow/body_ir.py), [`tilelang/dataflow/primfunc_lowering.py`](./tilelang/dataflow/primfunc_lowering.py) |
| Common lowering ownership contract | [`docs/dataflow/dataflow_refactor/mla_refactor/dataflow_lowering_architecture_contract.md`](./docs/dataflow/dataflow_refactor/mla_refactor/dataflow_lowering_architecture_contract.md) |
| Runtime ABI packing | [`tilelang/dataflow/runtime.py`](./tilelang/dataflow/runtime.py), [`tilelang/dataflow/launch.py`](./tilelang/dataflow/launch.py) |
| Wrapper generation | [`tilelang/dataflow/wrapper.py`](./tilelang/dataflow/wrapper.py) |
| CUDA launch and target validation | [`tilelang/dataflow/executor.py`](./tilelang/dataflow/executor.py) |

## 3. Programming Framework and Operators

User code normally imports both the TileLang language surface and tilelang.dataflow
contracts:

```python
import tilelang.language as T
import tilelang.dataflow as df
```

### 3.1 Typed intermediates

`@T.dataflow_intermediate` declares the values crossing stage boundaries. A
field can be a scalar or a fixed physical tensor. Layout requests can be
attached when the common lowering path needs a matrix/swizzle contract.

```python
@T.dataflow_intermediate
class Partial:
    score_max: T.float32
    score_sum: T.float32
    output: T.Tensor((block_h, value_dim), T.float32)
```

The type is used by graph validation, slot-size computation, handler ABI
construction, memory planning, and communication planning. It is not only a
Python annotation.

### 3.2 Declarative operators

| Operator | Purpose |
| --- | --- |
| `@T.dataflow.iter` | Compute partial results over an input-dependent range. |
| `@T.dataflow.map` | Apply a map stage, optionally consuming and producing typed intermediates. |
| `DataflowProgram.reshared` | Preserve a logical intermediate while changing its physical partitioning, ownership, arity, or consumer order. |
| `@T.dataflow.reduce` | Associatively combine partial intermediates. |
| `@T.dataflow.finalize` | Convert the final intermediate into user-visible output. |

The former `T.dataflow_iter`, `T.dataflow_map`, `T.dataflow_reduce`, and
`T.dataflow_finalize` names remain compatibility aliases. The namespace also
leaves the existing `T.reduce(buffer, out, ...)` buffer-reduction API untouched.

`partial(...)` is the graph builder for a leading range-producing map stage.
It is used by the MLA `partial -> reduce -> finalize` pattern. General stage
graphs use `map(...)` and may insert `reshared(...)` between producer and
consumer maps.

### 3.3 Physical operation contracts

Logical operators can carry typed requests that let the compiler choose a
physical implementation without embedding scheduler or instruction details in
the operator body.

| Contract | Controls |
| --- | --- |
| `DataflowRangeCoarseningRequest` | Logical range extent, handler extent, and output tile arity. |
| `DataflowPipelineRequest` | Producer/consumer synchronization and pipeline staging. |
| `DataflowOperatorPhysicalContract` | Input/output slot form, direct outputs, and physical slot expectations. |
| `DataflowResharedTransportRequest` | Reshared family, logical/physical arity, field mapping, and consumer order. |
| `DataflowCrossHandlerHandoffRequest` | Explicit resident buffering between successive handlers. |
| `DataflowTensorLayoutRequest` | Logical tensor rank and physical layout family. |
| `DataflowPrecisionPolicy` | Accumulator and specialization precision choices. |
| `DataflowMemoryPolicy` | Shared, scratch-backed, or direct-global memory planning policy. |

Contracts are legality and selection inputs. Hardware instruction selection
still belongs to common TileLang lowering.

### 3.4 MLA decode: `partial -> reduce -> finalize`

The non-paged MLA example declares:

```python
program = (
    T.dataflow_program(
        task_domain=("batch", "head_block"),
        dynamic_ranges={"kv": "seq_lens"},
    )
    .partial(split_tile(...), task_args=("batch", "head_block"), range_axis="kv")
    .reduce(reduce_partial())
    .finalize(finalize(Output="Output"), fused_reduce=reduce_finalizers)
)
```

Execution proceeds as follows:

1. Each `(batch, head_block)` task obtains its own KV range length from
   `seq_lens` and an optional range offset.
2. The scheduler divides the range into split-KV subtasks and balances them
   across CTA queues.
3. Partial handlers produce online-softmax state and output accumulators.
4. Reduction handlers combine only the available partials. Fine-grained
   producer-consumer edges replace a global phase boundary.
5. The finalizer normalizes the accumulator and writes the output. Fused reduce
   finalizers remove avoidable tail dispatch when a legal reduction arity is
   known.

Short and long sequences can therefore use different subtask counts and handler
variants within the same declarative operator.

### 3.5 Fused MoE: `map -> reshared -> map`

The fused MoE example consumes grouped routing metadata and builds one logical
task per active token chunk (`group_block`):

```python
program = (
    T.dataflow_program(
        task_domain=("group_block",),
        dynamic_ranges={
            "expert_tile": "expert_tiles",
            "hidden_tile": "hidden_tiles",
        },
    )
    .map(gemm1(), name="gemm1", range_axis="expert_tile", ...)
    .reshared(input="gemm1", name="routed_reshared", transport_contract=...)
    .map(gemm2(), input="routed_reshared", range_axis="hidden_tile", ...)
)
```

The physical flow is:

1. Routing metadata (`group_sizes`, offsets, padded offsets, and the expert for
   each group block) determines active token chunks. A hot expert naturally
   contributes more `group_block` tasks than a cold expert.
2. GEMM-1 maps the expert/intermediate dimension. Each handler loads the token
   chunk and a shard of gate/up weights, computes the gated intermediate, and
   produces `DataflowRoutedUpShard` in a planned output slot.
3. `reshared` changes ownership from GEMM-1 expert shards to the input collection
   required by GEMM-2. The transport plan records physical arity, field-axis
   mapping, and GEMM-2 access order.
4. Within a cluster, the intermediate shards can be pushed directly to consumer
   shared-memory slots or kept resident when the execution contract permits it.
   Cross-cluster dependencies use planned HBM staging.
5. GEMM-2 maps the output hidden dimension. For each hidden tile it consumes the
   logical GEMM-1 intermediate across expert shards, loads down-projection
   weights, accumulates, applies routing weights, and writes the output rows.

The scheduler can assign different amounts of work to CTA queues according to
the active expert/token chunks. GEMM-1 and GEMM-2 remain separate handlers but
execute under one wrapper launch, so their dependency does not require a kernel
barrier or mandatory intermediate HBM round trip.

### 3.6 Source locations for programming APIs and operators

| Concern | Source location |
| --- | --- |
| `T.dataflow_*` language exports | [`tilelang/language/dataflow.py`](./tilelang/language/dataflow.py) |
| Decorator implementation | [`tilelang/dataflow/operators.py`](./tilelang/dataflow/operators.py) |
| Intermediate schemas and calls | [`tilelang/dataflow/ir.py`](./tilelang/dataflow/ir.py) |
| Program graph (`map`, `reshared`, `reduce`, `finalize`) | [`tilelang/dataflow/program.py`](./tilelang/dataflow/program.py) |
| Typed operation contracts | [`tilelang/dataflow/operation_contracts.py`](./tilelang/dataflow/operation_contracts.py) |
| Range coarsening | [`tilelang/dataflow/range_coarsening.py`](./tilelang/dataflow/range_coarsening.py) |
| Pipeline planning | [`tilelang/dataflow/pipeline_planning.py`](./tilelang/dataflow/pipeline_planning.py) |
| Reshared transport planning | [`tilelang/dataflow/reshared_transport.py`](./tilelang/dataflow/reshared_transport.py) |
| Cross-handler handoff | [`tilelang/dataflow/handoff_planning.py`](./tilelang/dataflow/handoff_planning.py), [`tilelang/dataflow/primfunc_lowering.py`](./tilelang/dataflow/primfunc_lowering.py) |
| GEMM implementation selection | [`tilelang/dataflow/gemm_lowering.py`](./tilelang/dataflow/gemm_lowering.py) |
| MLA example | [`examples/dataflow/mla/mla_decode_non_paged.py`](./examples/dataflow/mla/mla_decode_non_paged.py) |
| Fused MoE example | [`examples/dataflow/fusedmoe/example_fusedmoe_dataflow.py`](./examples/dataflow/fusedmoe/example_fusedmoe_dataflow.py) |

## 4. Communication Primitives and Hardware Mechanisms

### 4.1 Runtime ABI

The host planner and CUDA wrapper share four central records:

| Record | Selected contents |
| --- | --- |
| `DataflowInstruction` | Opcode, handler ID, task ID, argument offset, communication range, primary slot, flags. |
| `DataflowQueue` | Flat instruction array plus per-CTA offsets and lengths. |
| `DataflowSlot` | Shared/global offsets, byte size, storage flags, and lifetime metadata. |
| `DataflowCommPlan` | Transport kind, source/destination slots, peer CTA rank, bytes, flag epoch, and barrier metadata. |

Fixed-width layouts are mirrored by Python packing in
`tilelang/dataflow/runtime.py` and CUDA structs in
`src/tl_templates/cuda/dataflow_runtime.h`. ABI size assertions in the CUDA
header protect host/device compatibility.

### 4.2 Cluster-shared transfer

For a same-cluster edge, the transfer protocol is:

1. The destination CTA initializes a local `mbarrier` for the destination slot.
2. The producer maps the destination shared-memory pointer and barrier into the
   cluster address space with `mapa.shared::cluster`.
3. The producer performs remote `mbarrier.arrive.expect_tx` with the expected
   byte count, establishes the asynchronous shared-memory proxy ordering, and
   issues `cp.async.bulk.shared::cluster.shared::cta` with
   `mbarrier::complete_tx::bytes`.
4. The `complete_tx::bytes` form binds copy completion to the destination
   barrier. Hardware retires the pending transaction bytes as the asynchronous
   copy completes.
5. The consumer waits with the hardware `mbarrier` parity-wait helper and then
   executes the required proxy fence before ordinary shared-memory reads.

Destination completion and source lifetime are related but distinct. The
destination `mbarrier` tells the consumer when the received bytes are visible.
The wrapper also tracks completion/acknowledgement for outstanding remote reads
so that the producer cannot reuse its source slot or exit while a cluster send
still references it.

This protocol is asynchronous at the copy engine. It does not mean that the
producer may immediately destroy the source or that the consumer may read before
the barrier completes.

### 4.3 Cross-cluster HBM transfer

Cluster shared memory is addressable only within a thread-block cluster. For a
cross-cluster dependency, tilelang.dataflow uses a global exchange slot:

1. the producer writes the slot through a TMA or scalar global path;
2. a device-scope fence orders the payload before a release flag;
3. the consumer waits for the corresponding flag epoch with acquire semantics;
4. the consumer issues a global-to-shared TMA load and waits on its local
   `mbarrier`, or uses the scalar fallback selected by the plan;
5. flag epochs and slot lifetimes allow safe reuse across queue generations.

The logical handler ABI is unchanged. Only the slot and communication records
differ from the cluster-shared route.

### 4.4 TMA multicast input

TMA multicast is a global-to-cluster-shared operation. The issued instruction
contains a CTA mask identifying cluster ranks that receive the tensor tile. Each
participating destination associates the transfer with its `mbarrier`, and
consumers wait before using the shared-memory tile.

At the TileLang surface this appears as a copy with multicast/cooperative input
annotations and a compatible tensor-map descriptor. Lowering validates layout,
alignment, cluster shape, and target capabilities before emitting the SM90+ TMA
form. This path is useful when MLA CTAs share query-side data or when several
MoE GEMM shards consume the same token tile or weight tile.

Multicast is different from unicast: unicast has one destination CTA per issued
transfer, while multicast names multiple destination ranks in the cluster.

### 4.5 Topology-aware placement

CUDA exposes cluster launch controls but does not directly provide the complete
physical GPC grouping needed by the scheduler. tilelang.dataflow measures cluster occupancy
behavior and observed SM IDs, then infers a usable topology model. The scheduler
uses that model to form legal clusters, keep communicating CTAs close, and avoid
fragmenting the available SM set.

The resolved topology and target capability snapshot are included in compile
and artifact fingerprints. A launch validates the artifact against the actual
device before loading the generated kernel.

### 4.6 Source locations for communication and hardware support

| Concern | Source location |
| --- | --- |
| Host-side ABI records and packing | [`tilelang/dataflow/runtime.py`](./tilelang/dataflow/runtime.py) |
| Launch package and uploaded buffers | [`tilelang/dataflow/launch.py`](./tilelang/dataflow/launch.py), [`tilelang/dataflow/executor.py`](./tilelang/dataflow/executor.py) |
| Device ABI and queue helpers | [`src/tl_templates/cuda/dataflow_runtime.h`](./src/tl_templates/cuda/dataflow_runtime.h) |
| Cluster/HBM communication dispatch | [`src/tl_templates/cuda/dataflow_comm.h`](./src/tl_templates/cuda/dataflow_comm.h) |
| Cluster TMA store and TMA multicast templates | [`src/tl_templates/cuda/copy_sm90.h`](./src/tl_templates/cuda/copy_sm90.h) |
| Tile copy lowering | [`src/backend/cuda/op/copy.cc`](./src/backend/cuda/op/copy.cc), [`src/transform/lower_tile_op.cc`](./src/transform/lower_tile_op.cc) |
| CUDA wrapper dispatch generation | [`tilelang/dataflow/wrapper.py`](./tilelang/dataflow/wrapper.py), [`tilelang/dataflow/primfunc_linking.py`](./tilelang/dataflow/primfunc_linking.py) |
| Barrier and slot lifetime planning | [`tilelang/dataflow/barrier_planning.py`](./tilelang/dataflow/barrier_planning.py), [`tilelang/dataflow/memory_planner.py`](./tilelang/dataflow/memory_planner.py) |
| GPC/SM topology probe | [`tilelang/dataflow/gpc_sm_probe.py`](./tilelang/dataflow/gpc_sm_probe.py), [`tilelang/dataflow/die_sm_probe.py`](./tilelang/dataflow/die_sm_probe.py) |
| Topology model | [`tilelang/dataflow/hardware_topology.py`](./tilelang/dataflow/hardware_topology.py), [`tilelang/dataflow/topology.py`](./tilelang/dataflow/topology.py) |
| Target capability checks | [`tilelang/utils/target_capabilities.py`](./tilelang/utils/target_capabilities.py) |

## 5. Building and Inspecting an Operator

### 5.1 Authoring workflow

1. Identify the logical task domain and the input-dependent range axes.
2. Declare every cross-stage value as a typed `dataflow_intermediate`.
3. Write map/iter, reduce, and finalize bodies as ordinary TileLang operator
   bodies.
4. Compose the stage graph and name dependencies explicitly.
5. Add typed range, pipeline, layout, transport, handoff, precision, and memory
   requests only where the operator requires them.
6. Build a `DataflowKernelSpec` with topology, current range lengths, task
   extents/coordinates, `block_size`, and scheduler configuration.
7. Compile through `@df.jit` or `df.compile_kernel_spec` and inspect the resulting
   plan before benchmarking.

### 5.2 Inspection interfaces

`DataflowCompiledProgram` exposes structured information intended for tooling
and maintenance:

```python
compiled = kernel_factory(...)

plan = compiled.dump_plan()
decisions = compiled.decision_artifact()
layout_report = compiled.validate_memory_layout()
cuda_source = compiled.wrapper_source
```

- `dump_plan()` shows queue lengths, instructions, range specialization,
  reshared transport, handoff, slots, communication, and launch resources.
- `decision_artifact()` records implementation selection and target provenance.
- `validate_memory_layout()` checks handler scratch, communication slots, and
  dynamic shared-memory layout.
- `wrapper_source` is useful for debugging generated code; structured artifacts
  are the stable inspection boundary.

### 5.3 Where to modify the system

| Change | Primary files |
| --- | --- |
| Add or validate a frontend construct | `tilelang/language/dataflow.py`, `tilelang/dataflow/operators.py`, `tilelang/dataflow/program.py` |
| Add a scheduling policy | `tilelang/dataflow/scheduler.py`, `tilelang/dataflow/scheduler_auto_policy.py`, `tilelang/dataflow/scheduler_config.py` |
| Add a handler specialization | `tilelang/dataflow/handler_identity.py`, `tilelang/dataflow/handler_registry.py`, `tilelang/dataflow/primfunc_lowering.py` |
| Add a physical operation contract | `tilelang/dataflow/operation_contracts.py` and the matching planner/lowering module |
| Add a transport realization | `tilelang/dataflow/reshared_transport.py`, `tilelang/dataflow/runtime.py`, `src/tl_templates/cuda/dataflow_comm.h` |
| Change wrapper dispatch or lifetime handling | `tilelang/dataflow/wrapper.py`, `tilelang/dataflow/primfunc_linking.py`, `tilelang/dataflow/barrier_planning.py` |
| Add a generic TileLang hardware implementation | `src/backend/cuda/op/`, `src/transform/`, and `src/tl_templates/cuda/` |
| Add an end-to-end operator | `examples/dataflow/` plus focused tests under `testing/python/dataflow/` |

### 5.4 Relevant tests

The following groups cover the main contracts:

- Frontend and graph: `test_dataflow_frontend.py`, `test_dataflow_program.py`,
  `test_dataflow_reshared_graph.py`.
- Scheduling and specialization: `test_dataflow_scheduler.py`,
  `test_dataflow_scheduler_auto_policy.py`, `test_dataflow_range_coarsening.py`.
- Lowering and wrapper: `test_dataflow_primfunc_lowering.py`,
  `test_dataflow_primfunc_linking.py`, `test_dataflow_wrapper_compile.py`.
- Runtime and communication: `test_dataflow_runtime.py`,
  `test_dataflow_barrier_planning.py`, `test_dataflow_cluster_mbarrier_debug.py`.
- Operators: `test_dataflow_mla_decode_nonpaged.py` and
  `test_dataflow_fusedmoe_example.py`.
- Target and topology: `test_dataflow_target_capabilities.py` and
  `test_dataflow_topology_probe.py`.

All test files above are under [`testing/python/dataflow/`](./testing/python/dataflow/).

## 6. Design Summary

tilelang.dataflow turns a multi-phase inference operator into a typed graph of dynamic
subtasks. Static handlers implement computation; an input-dependent plan assigns
those handlers to per-CTA queues, selects shape variants, and plans storage and
communication. A cluster wrapper executes the queues in one launch.

For MLA, this model balances split-KV work and performs fine-grained reduction
and finalization. For MoE, it connects GEMM-1 and GEMM-2 through a reshared
intermediate whose physical ownership is chosen for the routed token workload.
In both cases, same-cluster dependencies can use TMA and hardware `mbarrier`
instead of mandatory HBM round trips and kernel barriers, while target-aware HBM
staging preserves the same logical execution model across cluster boundaries.

For the detailed design, please refer to our SOSP '26 paper:

**Taming Dynamism on GPUs: Cross-SM Kernel Fusion via SM Cooperation and Just-in-Time Reduction**

*Jingkai He, Guangda Sun, Tianjian Li, Dong Du, Yubin Xia, Haibo Chen (Shanghai Jiao Tong University)*