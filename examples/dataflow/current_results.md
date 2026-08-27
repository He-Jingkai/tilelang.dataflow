# Current Dataflow Competitor Results

Last updated: 2026-08-18.

This page records the current published TileLang Dataflow results. Lower
latency is better. “Dataflow speedup” means:

```text
competitor_p50 / dataflow_p50 - 1
```

The published Dataflow baseline snapshot is commit `3eaa9575`, rebased directly
onto `cd02a30c`. This refresh supersedes the `184ff763` publication. Every
point ran in a fresh Python process on the same NVIDIA H100 80GB HBM3, with one
measured round, 5 warmups, and 30 samples. The observed clocks were 1980 MHz SM
and 2619 MHz HBM. Before promotion, every point passed the 3% per-point gate
against the preceding publication.

The FlashMLA and MegaMoE columns are retained historical baselines; they were
not rerun during this publication refresh.

## MLA vs FlashMLA

The public `published-h100` preset restores the complete measured contract:
`sm_count=112`, `cluster_size=16`, `threads=128`, `block_n=64`,
`block_h=32`, FP16 accumulation, `--maxrregcount=168`, and the generic
ordered global-joint scheduler. The Dataflow metric is `%globaltimer`
concurrent global-span p50. FlashMLA used same-host CUDA-event kernel-span p50
at revision `9241ae3ef9bac614dd25e45e507e089f888280e0`.

| Trace | Family | Batch | Published Dataflow (us) | FlashMLA (us) | Published speedup |
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

All 16 points passed the preceding publication gate; the largest regression
was +1.558%. The published geometric-mean speedup over the retained FlashMLA
baseline is 10.589%.

Run the complete matrix with one fresh process per trace:

```bash
for trace in $(seq 0 15); do
  PYTHONPATH=. TILELANG_CACHE_DIR="tmp/dataflow-mla-repro/cache-$trace" \
    python examples/dataflow/mla/benchmark.py \
      --preset published-h100 --trace-index "$trace" \
      --no-schedule-pic --quiet \
      --profile-walltime --profile-walltime-span-only \
      --profile-walltime-warmup 5 --profile-walltime-repeat 30 \
      --profile-walltime-output-dir "tmp/dataflow-mla-repro/trace-$trace"
done
```

Each `RESULT` line includes the published reference, regression percentage,
and gate verdict; the process exits nonzero when a preset point exceeds 3%.

## Fused MoE vs MegaMoE

The fixed matrix uses FP8, 32 experts, and `top_k=2`. DeepSeek uses
`d_hidden=7168, d_expert=2048`; Qwen uses
`d_hidden=2048, d_expert=768`. The benchmark presets preserve only the input
and typed profile. The production compiler still derives topology, transport,
pipeline ownership, and handoff from generic contracts. DeepSeek-1024 selects
streamed producer push; the other seven points select all-gather.

MegaMoE used the p50 of 30 historical Kineto/CUPTI kernel durations at revision
`23f46aa68c892a349bb7ce331a325e36acceb57e`.

| Model | Tokens | Published Dataflow (us) | MegaMoE (us) | Published speedup |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek | 128 | 483.808 | 491.6010 | +1.611% |
| DeepSeek | 256 | 482.464 | 514.0010 | +6.537% |
| DeepSeek | 512 | 501.664 | 550.8175 | +9.798% |
| DeepSeek | 1024 | 558.992 | 573.9050 | +2.668% |
| Qwen | 128 | 77.616 | 87.8725 | +13.214% |
| Qwen | 256 | 79.696 | 87.1360 | +9.335% |
| Qwen | 512 | 76.224 | 95.6800 | +25.525% |
| Qwen | 1024 | 81.424 | 109.2805 | +34.212% |

All eight points passed the preceding publication gate; the largest regression
was +1.158%. The published geometric-mean speedup over the retained MegaMoE
baseline is 12.384%. DeepSeek-1024 reproduces the generic hot-cluster-4
streamed-push path at 558.992 us rather than falling back to the old all-gather
path.

Run the complete matrix with one fresh process per workload:

```bash
for workload in \
  deepseek-128 deepseek-256 deepseek-512 deepseek-1024 \
  qwen-128 qwen-256 qwen-512 qwen-1024; do
  PYTHONPATH=. TILELANG_CACHE_DIR="tmp/dataflow-moe-repro/cache-$workload" \
    python examples/dataflow/fusedmoe/benchmark.py \
      --published-h100-workload "$workload" \
      --dataflow-walltime-dir "tmp/dataflow-moe-repro/$workload"
done
```

The workload preset enforces the exact single-point 5-warmup/30-sample
protocol, disables the TileLang cache, clears `DATAFLOW_*` overrides, prints a
`PUBLISHED_RESULT` gate verdict, and exits nonzero above 3%.

Correctness is tracked separately from this performance-only refresh. The
established publication audit ran every MLA point against the reference with
two-launch bitwise determinism and matched every MoE point between sequential
and concurrent execution. Its largest-point checks reported MLA trace 15
`max_abs_error=0.00016022` and DeepSeek-1024 `max_abs_error=0.780433`,
`relative_l2=0.013764` against the FP32 reference. The `3eaa9575` rebase
validation additionally passed the complete generic Dataflow test suite before
the performance matrix was measured.
