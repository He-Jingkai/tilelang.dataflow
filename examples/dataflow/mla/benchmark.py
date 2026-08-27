from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from pathlib import Path
import statistics
from typing import Any, Sequence

import tilelang.dataflow as df

from examples.dataflow.mla import mla_decode_non_paged as mla_decode


PUBLISHED_H100_PRESET = "published-h100"
PUBLISHED_H100_CONFIG_OVERRIDES = {
    "sm_count": 112,
    "cluster_size": 16,
    "block_n": 64,
    "range_block_n": 64,
    "block_h": 32,
    "threads": 128,
}
PUBLISHED_H100_P50_US = (
    43.296,
    64.304,
    105.424,
    185.680,
    88.448,
    44.048,
    54.096,
    92.240,
    71.408,
    120.592,
    293.312,
    573.504,
    59.696,
    99.392,
    213.104,
    409.520,
)
PUBLISHED_PER_POINT_GATE_PERCENT = 3.0

REAL_TRACE_NONPAGED_MLA_SEQ_LENS = [
    [
        233,
        453,
        672,
        895,
        1137,
        1405,
        1722,
        2104,
        2561,
        3109,
        3770,
        4579,
        5602,
        6962,
        8916,
        12411,
    ],  # cognitivecomputations_dolphin-r1_batch_16.csv
    [
        146,
        264,
        374,
        482,
        591,
        698,
        809,
        920,
        1039,
        1164,
        1293,
        1434,
        1586,
        1749,
        1931,
        2135,
        2360,
        2605,
        2869,
        3161,
        3501,
        3881,
        4297,
        4764,
        5278,
        5856,
        6537,
        7357,
        8295,
        9704,
        11573,
        14839,
    ],  # cognitivecomputations_dolphin-r1_batch_32.csv
    [
        400,
        529,
        614,
        697,
        740,
        797,
        877,
        942,
        972,
        1047,
        1079,
        1114,
        1147,
        1206,
        1238,
        1282,
        1311,
        1356,
        1386,
        1424,
        1478,
        1499,
        1557,
        1598,
        1643,
        1684,
        1719,
        1759,
        1826,
        1873,
        1941,
        1977,
        2037,
        2097,
        2181,
        2260,
        2342,
        2407,
        2535,
        2673,
        2782,
        2897,
        3060,
        3223,
        3392,
        3594,
        3725,
        3951,
        4136,
        4422,
        4679,
        4985,
        5269,
        5657,
        6169,
        6560,
        7126,
        7782,
        8556,
        9672,
        10736,
        11951,
        13715,
        17616,
    ],  # cognitivecomputations_dolphin-r1_batch_64.csv
    [
        351,
        435,
        485,
        552,
        593,
        633,
        659,
        700,
        737,
        767,
        794,
        823,
        850,
        873,
        909,
        946,
        968,
        997,
        1021,
        1043,
        1062,
        1092,
        1116,
        1136,
        1151,
        1167,
        1197,
        1222,
        1238,
        1258,
        1272,
        1290,
        1308,
        1327,
        1334,
        1355,
        1379,
        1395,
        1402,
        1423,
        1460,
        1479,
        1495,
        1521,
        1551,
        1571,
        1591,
        1620,
        1636,
        1659,
        1683,
        1695,
        1709,
        1732,
        1762,
        1791,
        1824,
        1834,
        1857,
        1883,
        1912,
        1931,
        1966,
        1988,
        2017,
        2070,
        2106,
        2121,
        2169,
        2185,
        2238,
        2298,
        2325,
        2379,
        2401,
        2469,
        2529,
        2558,
        2661,
        2690,
        2728,
        2799,
        2886,
        2966,
        3009,
        3086,
        3141,
        3178,
        3261,
        3364,
        3488,
        3571,
        3636,
        3751,
        3819,
        3919,
        4090,
        4156,
        4351,
        4480,
        4613,
        4797,
        4961,
        5033,
        5286,
        5400,
        5615,
        5849,
        6001,
        6164,
        6420,
        6743,
        7034,
        7240,
        7498,
        7819,
        8218,
        8567,
        9059,
        9624,
        10203,
        11003,
        11623,
        12537,
        13429,
        14728,
        16630,
        19737,
    ],  # cognitivecomputations_dolphin-r1_batch_128.csv
    [
        266,
        627,
        1059,
        1578,
        2203,
        2951,
        3897,
        5071,
        6573,
        8497,
        11030,
        14537,
        19866,
        28775,
        30152,
        32767,
    ],  # liyucheng_ShareGPT90K_batch_16.csv
    [
        143,
        271,
        376,
        465,
        548,
        624,
        696,
        765,
        832,
        898,
        966,
        1035,
        1111,
        1191,
        1282,
        1381,
        1491,
        1615,
        1754,
        1909,
        2086,
        2272,
        2472,
        2700,
        2947,
        3221,
        3539,
        3922,
        4354,
        4931,
        5780,
        7768,
    ],  # liyucheng_ShareGPT90K_batch_32.csv
    [
        82,
        136,
        180,
        216,
        248,
        278,
        309,
        337,
        363,
        392,
        417,
        442,
        469,
        497,
        520,
        546,
        574,
        600,
        625,
        650,
        680,
        707,
        729,
        758,
        783,
        811,
        842,
        868,
        895,
        924,
        950,
        976,
        1005,
        1033,
        1059,
        1092,
        1122,
        1154,
        1189,
        1222,
        1260,
        1296,
        1336,
        1385,
        1442,
        1491,
        1541,
        1613,
        1689,
        1772,
        1869,
        1963,
        2071,
        2207,
        2351,
        2526,
        2723,
        2972,
        3233,
        3539,
        3931,
        4443,
        5190,
        6494,
    ],  # liyucheng_ShareGPT90K_batch_64.csv
    [
        59,
        93,
        121,
        144,
        167,
        186,
        206,
        224,
        242,
        257,
        270,
        284,
        300,
        313,
        327,
        341,
        355,
        371,
        383,
        396,
        409,
        422,
        438,
        449,
        463,
        476,
        489,
        501,
        515,
        525,
        540,
        554,
        567,
        579,
        592,
        605,
        617,
        632,
        644,
        659,
        674,
        686,
        701,
        715,
        727,
        739,
        750,
        764,
        778,
        792,
        804,
        819,
        831,
        841,
        856,
        874,
        890,
        905,
        918,
        931,
        944,
        956,
        969,
        983,
        998,
        1012,
        1027,
        1039,
        1053,
        1067,
        1084,
        1100,
        1113,
        1127,
        1144,
        1161,
        1182,
        1202,
        1217,
        1233,
        1249,
        1269,
        1287,
        1309,
        1330,
        1352,
        1376,
        1403,
        1426,
        1454,
        1480,
        1503,
        1534,
        1562,
        1581,
        1625,
        1660,
        1701,
        1753,
        1790,
        1832,
        1887,
        1930,
        1989,
        2050,
        2105,
        2169,
        2236,
        2324,
        2387,
        2486,
        2580,
        2672,
        2787,
        2906,
        3035,
        3160,
        3319,
        3480,
        3653,
        3829,
        4060,
        4304,
        4680,
        4990,
        5535,
        6353,
        7703,
    ],  # liyucheng_ShareGPT90K_batch_128.csv
    [
        566,
        1251,
        1951,
        2662,
        3403,
        4188,
        5041,
        5961,
        6975,
        8112,
        9405,
        10902,
        12696,
        14919,
        17885,
        22527,
    ],  # open-r1_OpenR1-Math-220k_batch_16.csv
    [
        328,
        677,
        1024,
        1376,
        1728,
        2079,
        2437,
        2804,
        3175,
        3560,
        3954,
        4358,
        4778,
        5226,
        5688,
        6172,
        6687,
        7222,
        7789,
        8402,
        9062,
        9773,
        10544,
        11365,
        12269,
        13279,
        14396,
        15725,
        17258,
        19167,
        21637,
        25373,
    ],  # open-r1_OpenR1-Math-220k_batch_32.csv
    [
        1970,
        2399,
        2697,
        2956,
        3226,
        3433,
        3639,
        3857,
        4043,
        4266,
        4467,
        4648,
        4870,
        5084,
        5278,
        5461,
        5652,
        5855,
        6049,
        6298,
        6506,
        6723,
        6988,
        7208,
        7447,
        7650,
        7913,
        8156,
        8404,
        8679,
        8915,
        9192,
        9502,
        9753,
        10086,
        10406,
        10747,
        11125,
        11498,
        11859,
        12267,
        12703,
        13182,
        13554,
        13994,
        14381,
        14940,
        15473,
        15891,
        16401,
        17060,
        17745,
        18393,
        19067,
        19962,
        20708,
        21582,
        22612,
        23712,
        24980,
        26245,
        27831,
        29665,
        31561,
    ],  # open-r1_OpenR1-Math-220k_batch_64.csv
    [
        1660,
        2052,
        2305,
        2462,
        2623,
        2770,
        2893,
        3008,
        3136,
        3252,
        3373,
        3475,
        3573,
        3681,
        3813,
        3906,
        4003,
        4108,
        4208,
        4305,
        4413,
        4516,
        4609,
        4702,
        4812,
        4894,
        4997,
        5115,
        5213,
        5306,
        5407,
        5504,
        5609,
        5709,
        5811,
        5924,
        6047,
        6149,
        6272,
        6374,
        6469,
        6585,
        6709,
        6815,
        6921,
        7028,
        7146,
        7266,
        7358,
        7512,
        7617,
        7749,
        7867,
        8010,
        8137,
        8277,
        8400,
        8535,
        8664,
        8748,
        8888,
        9028,
        9166,
        9318,
        9451,
        9589,
        9754,
        9889,
        10046,
        10229,
        10370,
        10552,
        10708,
        10887,
        11061,
        11265,
        11472,
        11615,
        11829,
        12009,
        12186,
        12377,
        12618,
        12774,
        12974,
        13221,
        13462,
        13647,
        13911,
        14108,
        14301,
        14616,
        14845,
        15061,
        15332,
        15605,
        15843,
        16096,
        16256,
        16555,
        16879,
        17209,
        17566,
        17904,
        18258,
        18609,
        19023,
        19414,
        19884,
        20219,
        20647,
        21050,
        21563,
        22021,
        22499,
        23130,
        23595,
        24148,
        24726,
        25311,
        26013,
        26813,
        27509,
        28494,
        29282,
        30213,
        31294,
        32596,
    ],  # open-r1_OpenR1-Math-220k_batch_128.csv
    [
        710,
        1186,
        1665,
        2170,
        2705,
        3297,
        3944,
        4650,
        5422,
        6294,
        7281,
        8443,
        9787,
        11422,
        13614,
        17046,
    ],  # open-r1_OpenThoughts-114k-Code_decontaminated_batch_16.csv
    [
        548,
        827,
        1081,
        1338,
        1595,
        1851,
        2113,
        2386,
        2672,
        2969,
        3280,
        3597,
        3930,
        4282,
        4654,
        5037,
        5437,
        5854,
        6313,
        6800,
        7324,
        7871,
        8482,
        9146,
        9839,
        10621,
        11475,
        12445,
        13579,
        14941,
        16637,
        19529,
    ],  # open-r1_OpenThoughts-114k-Code_decontaminated_batch_32.csv
    [
        1197,
        1457,
        1640,
        1794,
        1945,
        2052,
        2194,
        2350,
        2446,
        2589,
        2686,
        2822,
        2941,
        3113,
        3242,
        3411,
        3516,
        3646,
        3845,
        4031,
        4162,
        4325,
        4475,
        4703,
        4907,
        5032,
        5226,
        5383,
        5585,
        5812,
        5966,
        6178,
        6411,
        6593,
        6804,
        7035,
        7233,
        7534,
        7870,
        8178,
        8535,
        8842,
        9103,
        9523,
        9798,
        10206,
        10651,
        11040,
        11600,
        11974,
        12455,
        13078,
        13633,
        14029,
        14787,
        15342,
        16120,
        16810,
        17441,
        18340,
        19371,
        20290,
        21958,
        24058,
    ],  # open-r1_OpenThoughts-114k-Code_decontaminated_batch_64.csv
    [
        1073,
        1294,
        1413,
        1511,
        1600,
        1665,
        1753,
        1834,
        1903,
        1966,
        2021,
        2107,
        2153,
        2236,
        2294,
        2363,
        2433,
        2477,
        2540,
        2614,
        2688,
        2760,
        2802,
        2864,
        2943,
        3032,
        3113,
        3146,
        3194,
        3270,
        3401,
        3484,
        3508,
        3563,
        3656,
        3776,
        3851,
        3956,
        4028,
        4092,
        4173,
        4229,
        4315,
        4363,
        4480,
        4567,
        4684,
        4750,
        4826,
        4921,
        4972,
        5040,
        5202,
        5245,
        5368,
        5448,
        5523,
        5662,
        5742,
        5795,
        5908,
        6016,
        6117,
        6150,
        6241,
        6412,
        6565,
        6680,
        6769,
        6804,
        6904,
        7066,
        7131,
        7265,
        7333,
        7503,
        7708,
        7824,
        7990,
        8176,
        8279,
        8399,
        8594,
        8773,
        8896,
        9124,
        9298,
        9541,
        9766,
        9851,
        9956,
        10194,
        10664,
        10889,
        10983,
        11143,
        11485,
        11684,
        11831,
        12005,
        12154,
        12307,
        12634,
        12981,
        13211,
        13678,
        14056,
        14254,
        14393,
        14793,
        15294,
        15772,
        16086,
        16300,
        16634,
        17012,
        17321,
        17727,
        18260,
        18640,
        19271,
        19742,
        20226,
        20814,
        21414,
        22435,
        23867,
        25743,
    ],  # open-r1_OpenThoughts-114k-Code_decontaminated_batch_128.csv
]
DEFAULT_REAL_TRACE_NONPAGED_MLA_SEQ_LENS = REAL_TRACE_NONPAGED_MLA_SEQ_LENS[-1]


@dataclass(frozen=True)
class NonPagedTiledMLATensors:
    q: Any
    q_pe: Any
    kv: Any
    k_pe: Any


def make_random_nonpaged_tiled_tensors(
    config: Any,
    *,
    torch_module: Any,
    seed: int = 0,
    device: str = "cuda",
) -> NonPagedTiledMLATensors:
    torch_module.manual_seed(seed)

    def randn(shape: tuple[int, ...]):
        return torch_module.randn(shape, device=device, dtype=torch_module.float16)

    return NonPagedTiledMLATensors(
        q=randn((config.batch, config.heads, config.dim)),
        q_pe=randn((config.batch, config.heads, config.pe_dim)),
        kv=randn((config.batch, config.kv_ctx, config.kv_heads, config.dim)),
        k_pe=randn((config.batch, config.kv_ctx, config.kv_heads, config.pe_dim)),
    )


def allocate_nonpaged_tiled_output(config: Any, *, torch_module: Any, device: str = "cuda"):
    return torch_module.full(
        (config.batch, config.heads, config.dim),
        float("nan"),
        device=device,
        dtype=getattr(torch_module, config.output_dtype),
    )


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Dataflow real-trace tiled non-paged MLA decode example.")
    parser.add_argument(
        "--preset",
        choices=(PUBLISHED_H100_PRESET,),
        default=None,
        help=("Apply the exact public H100 result configuration. Explicit shape and launch arguments still override the preset."),
    )
    parser.add_argument(
        "--trace-index",
        type=int,
        default=-1,
        help="Index into REAL_TRACE_NONPAGED_MLA_SEQ_LENS. Use -1 for the last trace.",
    )
    parser.add_argument("--heads", type=int, default=None, help="Override query head count.")
    parser.add_argument("--kv-heads", type=int, default=None, help="Override KV head count.")
    parser.add_argument("--dim", type=int, default=None, help="Override MLA latent value dimension.")
    parser.add_argument("--pe-dim", type=int, default=None, help="Override MLA positional dimension.")
    parser.add_argument("--sm-count", type=int, default=None, help="Override Dataflow queue/CTA count.")
    parser.add_argument("--cluster-size", type=int, default=None, help="Override CUDA cluster size.")
    parser.add_argument("--block-n", type=int, default=None, help="Override MLA KV tile size.")
    parser.add_argument(
        "--range-block-n",
        type=int,
        default=None,
        help="Override scheduler KV range block size.",
    )
    parser.add_argument("--block-h", type=int, default=None, help="Override query-head tile size.")
    parser.add_argument("--threads", type=int, default=None, help="Override TileLang handler thread count.")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile without allocating tensors or launching.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the compiled kernel with DataflowProfiler.",
    )
    parser.add_argument(
        "--profile-walltime",
        action="store_true",
        default=os.environ.get("DATAFLOW_PROFILE_WALLTIME") == "1",
        help="Run Dataflow per-SM walltime instrumentation and print a balance report.",
    )
    parser.add_argument(
        "--profile-walltime-span-only",
        action="store_true",
        help="Record only CTA begin/end timestamps for an unperturbed concurrent global span.",
    )
    parser.add_argument(
        "--profile-walltime-repeat",
        type=int,
        default=1,
        help="Measured walltime samples.",
    )
    parser.add_argument(
        "--profile-walltime-warmup",
        type=int,
        default=0,
        help="Warmup walltime samples.",
    )
    parser.add_argument(
        "--profile-walltime-top-k",
        type=int,
        default=12,
        help="Slow SMs to include in report.",
    )
    parser.add_argument(
        "--profile-walltime-output-dir",
        type=Path,
        default=Path("schedule_res/dataflow_mla_walltime"),
        help="Directory for Dataflow per-SM walltime detail/summary CSVs.",
    )
    parser.add_argument(
        "--profile-walltime-prefix",
        default=None,
        help="CSV filename prefix for Dataflow per-SM walltime output.",
    )
    parser.add_argument(
        "--no-schedule-pic",
        action="store_true",
        help="Disable scheduler visualization.",
    )
    parser.add_argument(
        "--pic-dir",
        default="schedule_res",
        help="Directory for scheduler visualization output.",
    )
    parser.add_argument(
        "--force-hbm-comms",
        action="store_true",
        help="Use HBM send/recv for inter-CTA reductions.",
    )
    parser.add_argument(
        "--iter-range-buckets",
        default=None,
        help=("Comma-separated exact ITER tile-count buckets, or 'auto'. Unmatched ranges use the generic handler."),
    )
    parser.add_argument(
        "--iter-range-exact-lengths",
        default=None,
        help=("Comma-separated exact ITER range lengths. Matched ranges use a fixed-length handler before tile-count buckets."),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable Dataflow compile and launch progress logs.",
    )
    return parser


def parse_args_from_list(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = make_arg_parser()
    args = parser.parse_args(argv)
    try:
        args.trace_index = normalize_real_trace_index(args.trace_index)
    except ValueError as err:
        parser.error(str(err))
    return args


def parse_args() -> argparse.Namespace:
    return parse_args_from_list()


def normalize_real_trace_index(trace_index: int) -> int:
    if trace_index == -1:
        return len(REAL_TRACE_NONPAGED_MLA_SEQ_LENS) - 1
    if 0 <= trace_index < len(REAL_TRACE_NONPAGED_MLA_SEQ_LENS):
        return trace_index
    raise ValueError(f"trace-index must be -1 or between 0 and {len(REAL_TRACE_NONPAGED_MLA_SEQ_LENS) - 1}, got {trace_index}")


def make_config_from_args(
    args: argparse.Namespace,
    seq_lens: Sequence[int],
) -> mla_decode.NonPagedTiledMLAConfig:
    base_config = mla_decode.NonPagedTiledMLAConfig()
    if args.preset == PUBLISHED_H100_PRESET:
        base_config = replace(base_config, **PUBLISHED_H100_CONFIG_OVERRIDES)
    config = replace(
        base_config,
        batch=len(seq_lens),
        heads=base_config.heads if args.heads is None else args.heads,
        kv_heads=base_config.kv_heads if args.kv_heads is None else args.kv_heads,
        kv_ctx=max(seq_lens),
        dim=base_config.dim if args.dim is None else args.dim,
        pe_dim=base_config.pe_dim if args.pe_dim is None else args.pe_dim,
    )
    return replace(
        config,
        sm_count=config.sm_count if args.sm_count is None else args.sm_count,
        cluster_size=(config.cluster_size if args.cluster_size is None else args.cluster_size),
        block_n=config.block_n if args.block_n is None else args.block_n,
        range_block_n=(config.range_block_n if args.range_block_n is None else args.range_block_n),
        block_h=config.block_h if args.block_h is None else args.block_h,
        threads=config.threads if args.threads is None else args.threads,
    )


def default_dataflow_mla_compile_flags(
    config: mla_decode.NonPagedTiledMLAConfig,
) -> tuple[str, ...]:
    if config.block_n >= 64 and config.block_h >= 32 and config.dim >= 64 and config.pe_dim >= 64:
        return ("--maxrregcount=168",)
    return ()


def default_dataflow_mla_semantic_config(
    config: mla_decode.NonPagedTiledMLAConfig,
) -> df.DataflowSemanticConfig:
    return df.DataflowSemanticConfig(
        precision=df.DataflowPrecisionPolicy(
            mode="explicit",
            accumulator_dtype=config.output_dtype,
        )
    )


def published_h100_compile_options() -> dict[str, Any]:
    return {
        "target_override": df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
            max_dynamic_shared_memory=232448,
        ),
        "scheduler_config": df.DataflowSchedulerConfig.from_options(
            ordered_interval_tree=True,
            ordered_tree_max_reduce_arity=2,
        ),
    }


def concurrent_global_span_p50_us(
    summary_rows: Sequence[dict[str, Any]],
) -> float:
    finish_by_sample: dict[int, list[float]] = {}
    for row in summary_rows:
        finish_by_sample.setdefault(int(row["sample"]), []).append(float(row["finish_time_us"]))
    if not finish_by_sample:
        raise RuntimeError("walltime profile returned no samples")
    return statistics.median(max(values) for values in finish_by_sample.values())


def main() -> None:
    args = parse_args()
    trace_index = args.trace_index
    seq_lens = REAL_TRACE_NONPAGED_MLA_SEQ_LENS[trace_index]
    config = make_config_from_args(args, seq_lens)
    compile_options = {
        "pic": not args.no_schedule_pic,
        "pic_dir": args.pic_dir,
        "force_hbm_comms": args.force_hbm_comms,
        "progress": not args.quiet,
        "compile_flags": default_dataflow_mla_compile_flags(config),
        "semantic_config": default_dataflow_mla_semantic_config(config),
    }
    if args.preset == PUBLISHED_H100_PRESET:
        compile_options.update(published_h100_compile_options())
    if args.iter_range_buckets is not None:
        compile_options["iter_range_buckets"] = args.iter_range_buckets
    if args.iter_range_exact_lengths is not None:
        compile_options["iter_range_exact_lengths"] = args.iter_range_exact_lengths
    compiled = mla_decode.nonpaged_tiled_split_mla(
        config,
        seq_lens=seq_lens,
        compile_options=compile_options,
    )

    if args.compile_only:
        print(
            "Dataflow tiled non-paged MLA decode compiled "
            f"for trace_index={trace_index}; "
            f"seq_lens={list(seq_lens)}; "
            f"shared_memory_bytes={compiled.launch_package.shared_memory_bytes}; "
            f"schedule_pic_dir={args.pic_dir if not args.no_schedule_pic else None}."
        )
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("examples/dataflow/mla/benchmark.py requires CUDA for execution")

    tensors = make_random_nonpaged_tiled_tensors(
        config,
        torch_module=torch,
        seed=0,
        device="cuda",
    )
    output = allocate_nonpaged_tiled_output(
        config,
        torch_module=torch,
        device="cuda",
    )

    if args.profile:
        profile_result = compiled.get_profiler().profile(
            Q=tensors.q,
            QPe=tensors.q_pe,
            KV=tensors.kv,
            KPe=tensors.k_pe,
            Output=output,
        )
        queue_lengths = [len(queue) for queue in compiled.plan.queues.values()]
        print(
            "Dataflow tiled non-paged MLA decode profile "
            f"for trace_index={trace_index}; "
            f"{profile_result.metric} mean={profile_result.mean_ms:.6f} ms, "
            f"median={profile_result.median_ms:.6f} ms, "
            f"min={profile_result.min_ms:.6f} ms, "
            f"max={profile_result.max_ms:.6f} ms; "
            f"shared_memory_bytes={compiled.launch_package.shared_memory_bytes}; "
            f"instructions={len(compiled.plan.instructions)}; "
            f"slots={len(compiled.plan.slots)}; "
            f"comms={len(compiled.plan.comms)}; "
            f"queues={len(compiled.plan.queues)}; "
            f"max_queue_len={max(queue_lengths, default=0)}."
        )
        if not args.profile_walltime:
            return

    if args.profile_walltime:
        prefix = args.profile_walltime_prefix or (f"trace{trace_index}_sm{config.sm_count}_cluster{config.cluster_size}_walltime")
        walltime_result = compiled.profile_walltime(
            Q=tensors.q,
            QPe=tensors.q_pe,
            KV=tensors.kv,
            KPe=tensors.k_pe,
            Output=output,
            repeat=args.profile_walltime_repeat,
            warmup=args.profile_walltime_warmup,
            span_only=args.profile_walltime_span_only,
            output_dir=args.profile_walltime_output_dir,
            prefix=prefix,
            top_k=args.profile_walltime_top_k,
        )
        p50_us = concurrent_global_span_p50_us(walltime_result.summary_rows)
        gate_fields = ""
        regression_percent = None
        if args.preset == PUBLISHED_H100_PRESET:
            reference_p50_us = PUBLISHED_H100_P50_US[trace_index]
            regression_percent = (p50_us / reference_p50_us - 1.0) * 100.0
            gate_fields = (
                f"reference_p50_us={reference_p50_us:.3f} "
                f"regression_percent={regression_percent:+.3f} "
                f"gate_percent={PUBLISHED_PER_POINT_GATE_PERCENT:.1f} "
                f"passes={regression_percent <= PUBLISHED_PER_POINT_GATE_PERCENT} "
            )
        print(
            "Dataflow tiled non-paged MLA decode walltime profile "
            f"for trace_index={trace_index}; "
            f"detail_csv={walltime_result.detail_csv}; "
            f"summary_csv={walltime_result.summary_csv}."
        )
        print(
            "RESULT "
            f"preset={args.preset or 'none'} trace_index={trace_index} "
            "metric=percent_globaltimer_concurrent_global_span_us "
            f"p50_us={p50_us:.3f} "
            f"{gate_fields}"
            f"warmups={args.profile_walltime_warmup} samples={args.profile_walltime_repeat} "
            f"sm_count={config.sm_count} cluster_size={config.cluster_size} "
            f"threads={config.threads}"
        )
        print(walltime_result.report)
        if regression_percent is not None and regression_percent > PUBLISHED_PER_POINT_GATE_PERCENT:
            raise RuntimeError(
                f"trace {trace_index} regressed {regression_percent:.3f}% against the published {reference_p50_us:.3f} us result"
            )
        return

    compiled(
        Q=tensors.q,
        QPe=tensors.q_pe,
        KV=tensors.kv,
        KPe=tensors.k_pe,
        Output=output,
    )
    torch.cuda.synchronize()

    print(f"Dataflow tiled non-paged MLA decode example launched for trace_index={trace_index}; seq_lens={list(seq_lens)}.")


if __name__ == "__main__":
    main()
