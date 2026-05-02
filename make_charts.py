"""Generate the two bar charts in the README from a representative benchmark run.

Usage: python3 make_charts.py

Numbers below are from a single run of ./run.sh on an M4 Max MacBook Pro
(macOS 25.3, clang 17, Python 3.12, numpy 2.3, mlx 0.30). Re-run and edit
RESULTS if you want to refresh.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# (label, tok/sec, watts).
# Power is rough: ~5 W per active M4 Max core (Apple's published per-core
# numbers under SIMD load); ~2 W for the Cyclone V FPGA fabric on the DE1-SoC.
RESULTS = [
    ("pure-python",         7_430,    5.0),
    ("numpy fp32",         40_244,    5.0),
    ("mlx fp32 (cpu)",      9_350,    5.0),
    ("mlx fp32 (gpu)",      3_337,    5.0),
    ("c fp32+NEON",     3_756_165,    5.0),
    ("c Q4.12",         3_143_586,    5.0),
    ("TALOS-V2 (FPGA)",    53_000,    2.0),
]

OUT = "charts"
os.makedirs(OUT, exist_ok=True)


def horizontal_bars(values, title, xlabel, fname):
    labels = [r[0] for r in RESULTS]
    colors = ["#888" if r[0] != "TALOS-V2 (FPGA)" else "#d62728" for r in RESULTS]
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    vals = [values[i] for i in order]
    cols = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=140)
    bars = ax.barh(labels, vals, color=cols, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=12, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(left=False)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt))
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    xmax = max(vals)
    ax.set_xlim(right=xmax * 4.0)
    for bar, v in zip(bars, vals):
        ax.text(v * 1.08, bar.get_y() + bar.get_height() / 2,
                _fmt(v, 0), va="center", ha="left", fontsize=9, color="#222")

    fig.tight_layout()
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
    "tokens / second  (log scale)",
    "throughput.png",
)
horizontal_bars(
    ppw,
    "perf-per-watt — same workload, rough power estimates",
    "tokens / second / watt  (log scale)",
    "perf_per_watt.png",
)
