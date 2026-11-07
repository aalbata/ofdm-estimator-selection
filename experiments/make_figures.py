"""Figures of the study, produced from the result files only.

Every figure is a vector PDF sized to a single IEEE column (3.45 in wide) with serif fonts
of 7 pt, thin lines and a colour per estimator that is the same in every figure. A PNG copy
of each figure is written for inspection.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "lines.linewidth": 0.9, "lines.markersize": 3, "axes.linewidth": 0.5, "grid.linewidth": 0.4, "grid.linestyle": ":",
    "legend.frameon": False, "pdf.fonttype": 42, "figure.dpi": 150,
})
COLUMN = 3.45
DOUBLE = 7.1

STYLE = {
    "nearest pilot": ("#7f7f7f", "o", ":"),
    "linear (constant edges)": ("#1f77b4", "s", "-"),
    "cubic spline (constant edges)": ("#17becf", "^", "-"),
    "DFT interpolation": ("#2ca02c", "v", "--"),
    "DFT least squares (ridge)": ("#2ca02c", "v", "--"),
    "Wiener (uniform to 2344 ns)": ("#ff7f0e", "D", "-."),
    "Wiener (uniform to 4688 ns)": ("#ff7f0e", "D", "-."),
    "Wiener (uniform to 112000 ns)": ("#ff7f0e", "D", "-."),
    "Wiener (matched)": ("#d62728", "x", "-"),
}
LABEL = {
    "nearest pilot": "nearest pilot", "linear (constant edges)": "linear", "cubic spline (constant edges)": "cubic spline",
    "DFT interpolation": "DFT, prefix window", "DFT least squares (ridge)": "tap fit, ridge", "Wiener (matched)": "Wiener, matched",
}


def label_of(name: str) -> str:
    if name.startswith("Wiener (uniform"):
        return "Wiener, uniform to prefix"
    return LABEL.get(name, name)


def style_of(name: str):
    return STYLE.get(name, ("#000000", ".", "-"))


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png", dpi=220, metadata={"Software": None})
    plt.close(fig)


def fig_regime(df: pd.DataFrame) -> None:
    """Error versus delay spread at 15 dB for four grids (two by two panels)."""
    panels = ["NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS", "DVB-T2 8K, PP3 (spacing 6)"]
    titles = ["NR DM-RS type 1 (spacing 2)", "NR DM-RS type 2 (pairs)", "LTE CRS (spacing 6)", "DVB-T2 PP3 (spacing 6)"]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 1.9), sharey=False)
    for ax, grid, title in zip(axes.ravel(), panels, titles):
        sub = df[(df.grid == grid) & (df.profile_kind == "sweep") & (df.snr_db == 15.0)]
        for est in sub.estimator.unique():
            s = sub[sub.estimator == est].sort_values("rms_delay_spread_ns")
            c, m, ls = style_of(est)
            ax.semilogx(s.rms_delay_spread_ns, s.error_db, color=c, marker=m, linestyle=ls, label=label_of(est), markevery=2)
        ax.set_title(title, pad=2)
        ax.grid(True, which="both")
        ax.set_xlabel("RMS delay spread (ns)")
        ax.set_ylabel("Error (dB)")
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.8)
    save(fig, "regime_delay_spread")


def fig_snr(df: pd.DataFrame) -> None:
    """Error versus SNR on the LTE grid for the machine-shop and automotive profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.75), sharey=True)
    for ax, prof in zip(axes, ["NIST machine shop (74 ns)", "NIST automotive plant (132 ns)"]):
        sub = df[(df.grid == "LTE 20 MHz, CRS") & (df.profile == prof)]
        for est in sub.estimator.unique():
            s = sub[sub.estimator == est].sort_values("snr_db")
            c, m, ls = style_of(est)
            ax.plot(s.snr_db, s.error_db, color=c, marker=m, linestyle=ls, label=label_of(est))
        ax.set_title(prof.replace("NIST ", ""), pad=2)
        ax.set_xlabel("SNR (dB)")
        ax.grid(True)
    axes[0].set_ylabel("Error (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=0.6)
    save(fig, "error_vs_snr_lte")


def fig_mismatch(df: pd.DataFrame) -> None:
    """Loss of a mismatched Wiener design relative to the matched filter, LTE grid, 15 dB."""
    sub = df[(df.grid == "LTE 20 MHz, CRS") & (df.snr_db == 15.0) & df.profile.str.startswith("exponential")]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 2.2), sharey=True)
    exp_designs = ["exponential x0.25", "exponential x0.5", "exponential x2", "exponential x4", "exponential x8"]
    uni_designs = ["uniform x2", "uniform x4", "uniform x8", "uniform x16", "uniform to CP"]
    markers = ["o", "s", "^", "v", "D"]
    labels = {"exponential x0.25": "exp., 0.25 x", "exponential x0.5": "exp., 0.5 x", "exponential x2": "exp., 2 x", "exponential x4": "exp., 4 x", "exponential x8": "exp., 8 x",
              "uniform x2": "unif., 2 x", "uniform x4": "unif., 4 x", "uniform x8": "unif., 8 x", "uniform x16": "unif., 16 x", "uniform to CP": "unif., prefix"}
    for ax, designs, title in zip(axes, (exp_designs, uni_designs), ("Exponential design", "Uniform design")):
        for d, mk in zip(designs, markers):
            s = sub[sub.design == d].sort_values("rms_delay_spread_ns")
            ax.semilogx(s.rms_delay_spread_ns, s.loss_db, marker=mk, label=labels[d])
        ax.set_title(title, pad=2)
        ax.set_xlabel("Actual RMS delay spread (ns)")
        ax.grid(True, which="both")
    axes[0].set_ylabel("Loss versus matched (dB)")
    handles = [h for ax in axes for h in ax.get_legend_handles_labels()[0]]
    names = [l for ax in axes for l in ax.get_legend_handles_labels()[1]]
    fig.legend(handles, names, loc="lower center", ncol=5, handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=0.6)
    save(fig, "wiener_mismatch")


def fig_ber(df: pd.DataFrame) -> None:
    """QPSK bit error rate versus SNR on the LTE grid, automotive-plant profile."""
    sub = df[(df.grid == "LTE 20 MHz, CRS") & (df.profile == "NIST automotive plant (132 ns)")]
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    for est in sub.estimator.unique():
        s = sub[sub.estimator == est].sort_values("snr_db")
        c, m, ls = style_of(est)
        ax.semilogy(s.snr_db, s.qpsk_ber, color=c, marker=m, linestyle=ls, label=label_of(est))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("QPSK bit error rate")
    ax.grid(True, which="both")
    ax.legend(ncol=2)
    fig.tight_layout()
    save(fig, "ber_lte_automotive")


def fig_per_tone(df: pd.DataFrame) -> None:
    """Per-subcarrier error along the LTE band, automotive-plant profile, 15 dB."""
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    for est in df.estimator.unique():
        s = df[df.estimator == est].sort_values("subcarrier")
        c, m, ls = style_of(est)
        ax.plot(s.subcarrier, 10 * np.log10(s.error), color=c, linestyle=ls, label=label_of(est), linewidth=0.8)
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("Error (dB)")
    ax.set_xlim(0, 1199)
    ax.grid(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "per_tone_lte_automotive")


def fig_profiles() -> None:
    """Measured average power-delay profiles of the three NIST runs."""
    folder = RESULTS / "nist_profiles"
    files = [("steam_plant", "steam plant"), ("machine_shop", "machine shop"), ("automotive_plant", "automotive plant")]
    if not all((folder / f"nist_profile_{k}.csv").exists() for k, _ in files):
        return
    fig, ax = plt.subplots(figsize=(COLUMN, 1.8))
    for (key, label), c, m in zip(files, ["#1f77b4", "#2ca02c", "#d62728"], ["o", "s", "^"]):
        t = pd.read_csv(folder / f"nist_profile_{key}.csv")
        ax.plot(t.delay_s * 1e9, 10 * np.log10(t.power / t.power.max()), color=c, marker=m, markersize=2, linewidth=0.8, label=label)
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Power relative to peak (dB)")
    ax.set_ylim(-32, 2)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save(fig, "measured_profiles")


def main() -> None:
    regime = pd.read_csv(RESULTS / "regime.csv")
    fig_regime(regime)
    fig_snr(regime)
    if (RESULTS / "per_tone.csv").exists():
        fig_per_tone(pd.read_csv(RESULTS / "per_tone.csv"))
    fig_profiles()
    if (RESULTS / "mismatch.csv").exists():
        fig_mismatch(pd.read_csv(RESULTS / "mismatch.csv"))
    if (RESULTS / "ber.csv").exists():
        fig_ber(pd.read_csv(RESULTS / "ber.csv"))
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
