# Dataflow CUDA Runtime

The Dataflow CUDA runtime is the device-side contract between the generic
scheduler, PrimFunc lowering, and the persistent wrapper kernel. Operator
semantics do not belong in this layer.

## Source layout

| File | Responsibility |
| --- | --- |
| `tilelang/dataflow/abi_schema.py` | Canonical host/device ABI schema. |
| `src/tl_templates/cuda/dataflow_abi_generated.h` | Generated CUDA ABI declarations. |
| `src/tl_templates/cuda/dataflow_runtime.h` | Queue, slot, argument, and indexing helpers. |
| `src/tl_templates/cuda/dataflow_comm.h` | Cluster and HBM transport helpers. |
| `tilelang/dataflow/wrapper.py` | Queue-driven wrapper generation. |
| `tilelang/dataflow/runtime.py` | Host-side packing of scheduler plans. |

`dataflow_abi_generated.h` is generated from the Python schema. Do not edit it
independently.

## ABI records

All scheduler-owned scalar fields use fixed-width integer types. The generated
header contains size, alignment, and field-offset assertions for every record.

### `DataflowInstruction`

One instruction is one queue entry for one CTA. It identifies the opcode,
handler, task, packed handler arguments, communication slice, and primary slot.
Its flag word also encodes the cluster receive/send counts required by batched
wrapper dispatch.

### `DataflowHandlerArgs`

The handler record carries task coordinates, dynamic range bounds, input and
output slot slices, and optional cross-handler handoff metadata. Device handlers
must decode this record through `tl::dataflow_handler_args`; they must not infer
offsets from a workload shape.

### `DataflowTensorArg`

Runtime tensors are passed in a separate record table containing the device
pointer and compact rank/dtype/flag metadata. Shape, stride, layout, and target
validation occur on the host before launch.

### `DataflowSlot`

A slot describes shared and global offsets, capacity, synchronization indices,
owner CTA, and placement flags. Offsets are relative to wrapper-owned bases so
the scheduler never embeds kernel-local pointers.

The current flags distinguish normal shared storage, scratch-backed storage,
direct-global HBM storage, cluster-gated push, and communicate slots.

### `DataflowCommPlan`

A communication record identifies the transport operation, source and
destination slots, peer CTA, barrier phase, flag epoch, byte subrange, and
segment position. Supported operations are:

- cluster send and receive;
- cluster release;
- blocking HBM send and receive;
- split HBM receive issue and wait.

The split receive form permits a scheduler to issue a ready TMA load before the
consumer reaches its wait point. If it was not issued speculatively, the wait
path issues the transfer itself, preserving correctness without blocking an
unrelated handler.

## Wrapper execution

The generated persistent wrapper performs the following work for each CTA:

1. Initialize wrapper-owned barriers and receive state.
2. Resolve the queue from the configured CTA/SM mapping.
3. Load the next instruction and its typed argument record.
4. Dispatch receive-side communication.
5. Call the linked PrimFunc handler selected by `handler_id`.
6. Dispatch send/release communication.
7. Continue until an exit instruction or queue exhaustion.

The wrapper owns queue traversal, communication order, and synchronization.
Linked handlers own operator computation and their declared scratch region.
Neither side may silently reuse storage owned by the other.

## Cluster transport

On SM90 and newer, cluster sends use `tl::tma_store_cluster` to write the
consumer CTA's shared-memory inbox. The producer synchronizes before issuing a
batch so handler stores are visible. The consumer waits on its local mbarrier,
performs the async-proxy fence, and synchronizes before invoking the handler.

Barrier phase and lifetime are scheduler-owned. A barrier may be reused only
after the plan proves that no previous transfer can still arrive at it.
Unsupported targets reject cluster transport instead of silently changing its
semantics.

## HBM transport

HBM transport uses a global staging range plus a release/acquire flag epoch:

- the producer publishes payload stores before advancing the flag;
- the consumer waits for the required epoch before reading the payload;
- monotonically increasing epochs make explicit flag reuse safe.

On supported targets the runtime can use TMA for shared-to-global stores and
global-to-shared loads. The scalar striped-copy implementation remains the
portable fallback. Direct-global slots publish and consume only the flag because
the handler already accesses the final global location.

Split HBM receive state is indexed by barrier and lives in the wrapper control
region. A non-blocking issue checks flag readiness; a wait either completes the
existing TMA operation or performs the original blocking issue-and-wait path.

## Memory ownership

Dynamic shared memory is partitioned into explicit regions for control state,
handler scratch, permanent communicate slots, and transient prefetch storage.
The compiler validates alignment, capacity, live ranges, and any intentional
aliasing before source generation.

Slots may share physical storage only when the lifetime validator proves the
ranges disjoint. Scratch-backed and direct-global placements are explicit plan
decisions and are reflected in the launch package.

## Scheduler obligations

Before packing a plan, the scheduler must guarantee:

- every queue offset and length is in bounds;
- handler, task, argument, slot, and communication references are valid;
- peer CTAs for cluster transfers belong to the same CUDA cluster;
- send/receive records agree on bytes, barrier phase, flag epoch, and storage;
- slot and barrier reuse is lifetime-safe;
- split HBM issue precedes its matching wait without introducing a wait cycle;
- terminal side effects and cross-handler handoffs are ordered;
- resource use fits the selected target and launch topology.

The runtime validates structural bounds again while packing. It does not repair
an illegal schedule.

## Maintenance rules

- Change the ABI in `tilelang/dataflow/abi_schema.py`, regenerate the header,
  and update host/device ABI tests together.
- Keep `arg_offset` byte-addressed and communication offsets record-addressed.
- Keep synchronization in transport helpers; do not duplicate it in
  operator-specific lowering.
- Add transport behavior through typed communication kinds and update packing,
  dispatch, lifetime validation, and tests as one change.
- Keep generated source private. Public tooling should consume typed plans and
  decision artifacts.
- Keep debug handlers opt-in under `tilelang.dataflow.experimental`; production
  compilation links general PrimFunc handlers.

Relevant generic coverage lives in `testing/python/dataflow`, including ABI,
barrier planning, scheduler, communication, handoff, PrimFunc lowering/linking,
wrapper generation, launch packing, and target-capability tests.
