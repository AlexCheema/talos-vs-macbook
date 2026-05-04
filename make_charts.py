"""Generate the two bar charts in the README from a representative benchmark run.

Usage: python3 make_charts.py

M4 Max numbers are upstream's recorded run on a MacBook Pro (clang 17,
Python 3.12, numpy 2.3, mlx 0.30). M3 Ultra is a 2025 Mac Studio (Mac15,14,
20P+8E, macOS 26.4.1). M1 Max is a 2022 Mac Studio (Mac13,1, macOS 26.3.1).
DGX Spark is NVIDIA GB10 (Grace ARM Cortex-X925/A725 + Blackwell, gcc 13.3,
glibc libmvec, nvcc 13.0 -arch=sm_121, Ubuntu 24.04 kernel 6.17). Re-run and
edit RESULTS to refresh.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# (label, tok/sec, watts).
# Power: Apple single P-core ~5 W under SIMD load (Apple's published per-core
# figures); Grace single core ~3 W (estimated; not measured); Blackwell 19.96 W
# measured via nvidia-smi --query-gpu=power.draw -lms 100 averaged across 257
# samples during a 10M-token bench_cuda_persistent run; Cyclone V FPGA fabric
# on the DE1-SoC ~2 W (TALOS-V2 measurement, upstream).
RESULTS = [
    ("M4 Max · pure-python",             7_430,    5.0),
    ("M4 Max · numpy fp32",             40_244,    5.0),
    ("M4 Max · mlx fp32 (cpu)",          9_350,    5.0),
    ("M4 Max · mlx fp32 (gpu)",          3_337,    5.0),
    ("M4 Max · c fp32+NEON",         3_756_165,    5.0),
    ("M4 Max · c Q4.12",             3_143_586,    5.0),

    ("M3 Ultra · pure-python",           8_039,    5.0),
    ("M3 Ultra · numpy fp32",           38_175,    5.0),
    ("M3 Ultra · mlx fp32 (cpu)",        5_407,    5.0),
    ("M3 Ultra · mlx fp32 (gpu)",        1_785,    5.0),
    ("M3 Ultra · c fp32+NEON",       3_632_988,    5.0),
    ("M3 Ultra · c Q4.12",           2_935_620,    5.0),

    ("M1 Max · pure-python",             4_600,    5.0),
    ("M1 Max · numpy fp32",             28_866,    5.0),
    ("M1 Max · mlx fp32 (cpu)",          9_122,    5.0),
    ("M1 Max · mlx fp32 (gpu)",          2_196,    5.0),
    ("M1 Max · c fp32+NEON",         2_910_293,    5.0),
    ("M1 Max · c Q4.12",             2_345_483,    5.0),

    ("Grace · pure-python",              6_455,    3.0),
    ("Grace · numpy fp32",              41_032,    3.0),
    ("Grace · c fp32+NEON",          4_364_405,    3.0),
    ("Grace · c Q4.12",              3_007_686,    3.0),

    ("Blackwell · cuda fp32",           19_127,   19.96),
    ("Blackwell · cuda persistent",    413_603,   19.96),

    ("FPGA · TALOS-V2",                 53_000,    2.0),
]

OUT = "charts"
os.makedirs(OUT, exist_ok=True)

PLATFORM_COLORS = (
    ("FPGA",        "#d62728"),
    ("Blackwell",   "#2c5e1e"),
    ("Grace",       "#76b900"),
    ("M3 Ultra",    "#555555"),
    ("M1 Max",      "#bbbbbb"),
    ("M4 Max",      "#888888"),
)


def color_for(label):
    for prefix, c in PLATFORM_COLORS:
        if label.startswith(prefix):
            return c
    return "#888"


def horizontal_bars(values, title, xlabel, fname, caption=None):
    labels = [r[0] for r in RESULTS]
    colors = [color_for(l) for l in labels]
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    vals = [values[i] for i in order]
    cols = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 10.5), dpi=140)
    bars = ax.barh(labels, vals, color=cols, edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=12, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(left=False)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt))
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    xmax = max(vals)
    ax.set_xlim(left=0, right=xmax * 1.18)
    for bar, v in zip(bars, vals):
        ax.text(v + xmax * 0.012, bar.get_y() + bar.get_height() / 2,
                _fmt(v, 0), va="center", ha="left", fontsize=9, color="#222")

    if caption:
        fig.text(0.01, 0.005, caption, fontsize=8, color="#555")

    fig.tight_layout(rect=(0, 0.02, 1, 1) if caption else (0, 0, 1, 1))
    path = os.path.join(OUT, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _fmt(v, _pos=None):
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}k" if v < 100_000 else f"{v/1_000:.0f}k"
    return f"{v:.0f}"


tps = [r[1] for r in RESULTS]
ppw = [r[1] / r[2] for r in RESULTS]

horizontal_bars(
    tps,
    "throughput — single-thread, batch=1, char-by-char",
    "tokens / second",
    "throughput.png",
)
horizontal_bars(
    ppw,
    "perf-per-watt — same workload, rough power estimates",
    "tokens / second / watt",
    "perf_per_watt.png",
    caption="power: Apple cores ~5 W (Apple per-core figures); Grace ~3 W (est., not measured); "
            "Blackwell 19.96 W (measured, nvidia-smi -lms 100, n=257); FPGA ~2 W (TALOS-V2).",
)
