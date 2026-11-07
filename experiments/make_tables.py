"""Tables and inline numbers of the paper, generated from the result files only.

Writes tables/numbers.tex (one macro per number quoted in the text) and the table files in
tables/. When the paper sources are present next to this folder, every macro they use must
be defined here; the script checks that and stops otherwise.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)
PAPER = ROOT / "paper"

SHORT = {
    "nearest pilot": "Nearest pilot", "linear (constant edges)": "Linear", "cubic spline (constant edges)": "Spline",
    "DFT interpolation": "DFT", "DFT least squares (ridge)": "Tap fit (ridge)", "Wiener (matched)": "Wiener, matched",
}


def short(name: str) -> str:
    if name.startswith("Wiener (uniform"):
        return "Wiener, uniform"
    return SHORT.get(name, name)


def fmt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}"


def macros_from_validation(v: pd.DataFrame, out: dict) -> None:
    out["ValCells"] = str(len(v))
    out["ValInsidePct"] = fmt(100 * v.inside_95.mean(), 1)
    out["ValMaxRelPct"] = fmt(v.relative_error_percent.abs().max(), 1)
    out["ValMedianRelPct"] = fmt(v.relative_error_percent.abs().median(), 2)
    out["ValMaps"] = str(v["map"].nunique())
    out["ValEstimators"] = str(v["estimator"].nunique())
    out["ValSymbols"] = "10,000"
    rows = []
    for m, g in v.groupby("map", sort=False):
        rows.append(f"{m} & {g.n_pilots.iloc[0]}/{g.n_tones.iloc[0]} & {len(g)} & {100*g.inside_95.mean():.0f} & {g.relative_error_percent.abs().max():.2f} \\\\")
    body = "\n".join(rows)
    (TABLES / "validation.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Exact error against Monte Carlo (10,000 symbols per cell). Columns: pilot map, pilots per analysed tones, cells (estimators, profiles and SNRs), share of cells where the exact value lies inside the 95 percent interval of the simulation, and largest relative deviation in percent.}\n\\label{tab:validation}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccc}\n\\toprule\nMap & Pilots & Cells & Inside (\\%) & Max dev. (\\%) \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def macros_from_regime(r: pd.DataFrame, out: dict) -> None:
    sweep = r[r.profile_kind == "sweep"]
    sites = r[r.profile_kind == "site"]
    # Recommendation table: cheapest estimator within 1 dB of matched Wiener, per grid x site x SNR.
    rows = []
    rec_rows = []
    for (grid, prof), g in sites.groupby(["grid", "profile"], sort=False):
        for snr, h in g.groupby("snr_db"):
            ref = h[h.estimator == "Wiener (matched)"].error_db.iloc[0]
            ok = h[h.error_db <= ref + 1.0].sort_values(["real_multiplications", "error_db"])
            chosen = ok.iloc[0]
            rec_rows.append({"grid": grid, "profile": prof, "snr_db": snr, "chosen": chosen.estimator, "chosen_db": chosen.error_db, "matched_db": ref, "linear_db": h[h.estimator == "linear (constant edges)"].error_db.iloc[0], "uniform_db": h[h.estimator.str.startswith("Wiener (uniform")].error_db.iloc[0]})
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(RESULTS / "recommendations.csv", index=False)
    out["RecCells"] = str(len(rec))
    counts = rec.chosen.map(short).value_counts()
    out["RecMatchedPct"] = fmt(100 * counts.get("Wiener, matched", 0) / len(rec), 0)
    # Site table: error at the factory profiles for the matched Wiener, the uniform design
    # spanning eight times the delay spread (from the mismatch study), the uniform design
    # spanning the prefix, and linear interpolation.
    mis = pd.read_csv(RESULTS / "mismatch.csv") if (RESULTS / "mismatch.csv").exists() else None
    grids = ["NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS"]
    gshort = {"NR 20 MHz, DM-RS type 1": "NR type 1", "NR 20 MHz, DM-RS type 2": "NR type 2", "LTE 20 MHz, CRS": "LTE CRS"}
    profiles = ["NIST steam plant (36 ns)", "NIST machine shop (74 ns)", "NIST automotive plant (132 ns)"]
    pshort = {"NIST steam plant (36 ns)": "Steam plant, 36 ns", "NIST machine shop (74 ns)": "Machine shop, 74 ns", "NIST automotive plant (132 ns)": "Automotive plant, 132 ns"}
    lines = []
    u8_losses, cp_losses, lin_gaps = [], [], []
    for prof in profiles:
        for snr in (5.0, 15.0, 25.0):
            cells = []
            for grid in grids:
                h = sites[(sites.grid == grid) & (sites.profile == prof) & (sites.snr_db == snr)]
                matched = float(h[h.estimator == "Wiener (matched)"].error_db.iloc[0])
                uniform_cp = float(h[h.estimator.str.startswith("Wiener (uniform")].error_db.iloc[0])
                linear = float(h[h.estimator == "linear (constant edges)"].error_db.iloc[0])
                u8 = np.nan
                if mis is not None:
                    row = mis[(mis.grid == grid) & (mis.profile == prof) & (mis.snr_db == snr) & (mis.design == "uniform x8")]
                    if not row.empty:
                        u8 = matched + float(row.loss_db.iloc[0])
                        u8_losses.append(float(row.loss_db.iloc[0]))
                cp_losses.append(uniform_cp - matched)
                lin_gaps.append(linear - matched)
                cells.append(f"{matched:.1f} & {u8:.1f} & {uniform_cp:.1f} & {linear:.1f}")
            lines.append(f"{pshort[prof]} & {snr:.0f} & " + " & ".join(cells) + " \\\\")
    header = " & ".join("\\multicolumn{4}{c}{" + gshort[g] + "}" for g in grids)
    (TABLES / "recommend.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Error in dB at the factory profiles (exponential, NIST mean RMS delay spreads at 2.2 GHz) for four estimators: the matched Wiener filter (M), the Wiener filter designed on a uniform profile spanning eight times the delay spread (U8), the Wiener filter designed on a uniform profile spanning the cyclic prefix (UP), and linear interpolation (L).}\n\\label{tab:sites}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{ll rrrr rrrr rrrr}\n\\toprule\n & & " + header + " \\\\\n\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14}\nProfile & SNR (dB) & M & U8 & UP & L & M & U8 & UP & L & M & U8 & UP & L \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    out["SiteUEightLossMin"] = fmt(min(u8_losses)) if u8_losses else "--"
    out["SiteUEightLossMax"] = fmt(max(u8_losses)) if u8_losses else "--"
    out["SiteCpLossMin"] = fmt(min(cp_losses))
    out["SiteCpLossMax"] = fmt(max(cp_losses))
    out["SiteLinearGapMin"] = fmt(min(lin_gaps))
    out["SiteLinearGapMax"] = fmt(max(lin_gaps))
    interp = sites[(sites.snr_db <= 25.0) & sites.estimator.isin(["linear (constant edges)", "cubic spline (constant edges)"]) & sites.grid.isin(grids)]
    out["InterpBiasShareMaxPct"] = fmt(100 * (interp.bias / (interp.bias + interp.noise)).max(), 1)
    # Inline numbers for the regime text.
    def err(grid, prof, snr, est):
        h = r[(r.grid == grid) & (r.profile == prof) & (r.snr_db == snr) & (r.estimator == est)]
        return float(h.error_db.iloc[0])
    lte, nr1, nr2 = "LTE 20 MHz, CRS", "NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2"
    auto, shop, steam = profiles[2], profiles[1], profiles[0]
    out["LteAutoLinearFifteen"] = fmt(err(lte, auto, 15, "linear (constant edges)"))
    out["LteAutoMatchedFifteen"] = fmt(err(lte, auto, 15, "Wiener (matched)"))
    out["LteAutoUniformFifteen"] = fmt(err(lte, auto, 15, "Wiener (uniform to 4688 ns)"))
    out["LteAutoDftFifteen"] = fmt(err(lte, auto, 15, "DFT interpolation"))
    out["LteAutoSplineFifteen"] = fmt(err(lte, auto, 15, "cubic spline (constant edges)"))
    out["LteAutoNearestFifteen"] = fmt(err(lte, auto, 15, "nearest pilot"))
    out["LteSteamLinearFifteen"] = fmt(err(lte, steam, 15, "linear (constant edges)"))
    out["LteSteamMatchedFifteen"] = fmt(err(lte, steam, 15, "Wiener (matched)"))
    out["NrOneShopLinearFifteen"] = fmt(err(nr1, shop, 15, "linear (constant edges)"))
    out["NrOneShopMatchedFifteen"] = fmt(err(nr1, shop, 15, "Wiener (matched)"))
    out["NrOneShopUniformFifteen"] = fmt(err(nr1, shop, 15, "Wiener (uniform to 2344 ns)"))
    out["NrTwoShopLinearFifteen"] = fmt(err(nr2, shop, 15, "linear (constant edges)"))
    out["NrTwoShopMatchedFifteen"] = fmt(err(nr2, shop, 15, "Wiener (matched)"))
    # Where linear interpolation is within 1 dB of matched, in the sweep, per grid: largest delay spread.
    lines = []
    for grid in ["NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS", "DVB-T2 8K, PP1 (spacing 3)", "DVB-T2 8K, PP3 (spacing 6)", "DVB-T2 8K, PP5 (spacing 12)", "DVB-T2 8K, PP7 (spacing 24)"]:
        g = sweep[sweep.grid == grid]
        for snr in (5.0, 15.0, 25.0):
            h = g[g.snr_db == snr]
            ref = h[h.estimator == "Wiener (matched)"].set_index("rms_delay_spread_ns").error_db
            lin = h[h.estimator == "linear (constant edges)"].set_index("rms_delay_spread_ns").error_db
            uni = h[h.estimator.str.startswith("Wiener (uniform")].set_index("rms_delay_spread_ns").error_db
            dft = h[h.estimator == "DFT interpolation"].set_index("rms_delay_spread_ns").error_db if (h.estimator == "DFT interpolation").any() else None
            gap_lin = (lin - ref).max()
            gap_uni = (uni - ref).max()
            gap_dft = (dft - ref).max() if dft is not None else np.nan
            lines.append((grid, snr, float((lin - ref).min()), float(gap_lin), float((uni - ref).min()), float(gap_uni), float(gap_dft)))
    gap = pd.DataFrame(lines, columns=["grid", "snr_db", "linear_gap_min_db", "linear_gap_max_db", "uniform_gap_min_db", "uniform_gap_max_db", "dft_gap_max_db"])
    gap.to_csv(RESULTS / "gaps.csv", index=False)
    body = []
    names = {"NR 20 MHz, DM-RS type 1": "NR type 1 (2)", "NR 20 MHz, DM-RS type 2": "NR type 2 (pairs)", "LTE 20 MHz, CRS": "LTE CRS (6)", "DVB-T2 8K, PP1 (spacing 3)": "DVB-T2 (3)", "DVB-T2 8K, PP3 (spacing 6)": "DVB-T2 (6)", "DVB-T2 8K, PP5 (spacing 12)": "DVB-T2 (12)", "DVB-T2 8K, PP7 (spacing 24)": "DVB-T2 (24)"}
    for grid in names:
        g = gap[gap.grid == grid].set_index("snr_db")
        cells = []
        for snr in (5.0, 15.0, 25.0):
            row = g.loc[snr]
            d = "--" if np.isnan(row.dft_gap_max_db) else f"{row.dft_gap_max_db:.1f}"
            cells.append(f"{row.linear_gap_min_db:.1f}/{row.linear_gap_max_db:.1f} & {row.uniform_gap_min_db:.1f}/{row.uniform_gap_max_db:.1f} & {d}")
        body.append(f"{names[grid]} & " + " & ".join(cells) + " \\\\")
    (TABLES / "gaps.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Gap to the matched Wiener error over the delay-spread sweep, in dB, as smallest/largest gap of linear interpolation and of the uniform Wiener design, and largest gap of the transform-domain estimator, at three SNRs. The pilot spacing of each map is in parentheses.}\n\\label{tab:gaps}\n\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{l ccc ccc ccc}\n\\toprule\n & \\multicolumn{3}{c}{SNR 5 dB} & \\multicolumn{3}{c}{SNR 15 dB} & \\multicolumn{3}{c}{SNR 25 dB} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\nMap & Linear & Uniform & DFT & Linear & Uniform & DFT & Linear & Uniform & DFT \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    out["SweepPoints"] = str(sweep.groupby("grid").rms_delay_spread_ns.nunique().iloc[0])


def macros_from_mismatch(m: pd.DataFrame, out: dict) -> None:
    sub = m[m.profile.str.startswith("exponential")]
    lte = sub[sub.grid == "LTE 20 MHz, CRS"]
    def worst(design, snr=None, grid=None):
        g = sub if grid is None else sub[sub.grid == grid]
        if snr is not None:
            g = g[g.snr_db == snr]
        return float(g[g.design == design].loss_db.max())
    def median(design, snr=None, grid=None):
        g = sub if grid is None else sub[sub.grid == grid]
        if snr is not None:
            g = g[g.snr_db == snr]
        return float(g[g.design == design].loss_db.median())
    out["MisHalfWorst"] = fmt(worst("exponential x0.5"))
    out["MisQuarterWorst"] = fmt(worst("exponential x0.25"))
    out["MisDoubleWorst"] = fmt(worst("exponential x2"))
    out["MisQuadWorst"] = fmt(worst("exponential x4"))
    out["MisOctWorst"] = fmt(worst("exponential x8"))
    out["MisUniFourWorst"] = fmt(worst("uniform x4"))
    out["MisUniEightWorst"] = fmt(worst("uniform x8"))
    out["MisUniSixteenWorst"] = fmt(worst("uniform x16"))
    out["MisUniCpWorst"] = fmt(worst("uniform to CP"))
    out["MisUniCpMedian"] = fmt(median("uniform to CP"))
    out["MisUniCpWorstLteFifteen"] = fmt(worst("uniform to CP", 15.0, "LTE 20 MHz, CRS"))
    out["MisUniCpWorstNrFifteen"] = fmt(worst("uniform to CP", 15.0, "NR 20 MHz, DM-RS type 1"))
    out["MisDoubleMedian"] = fmt(median("exponential x2"))
    out["MisHalfMedian"] = fmt(median("exponential x0.5"))
    designs = ["exponential x0.25", "exponential x0.5", "exponential x2", "exponential x4", "exponential x8", "uniform x2", "uniform x4", "uniform x8", "uniform x16", "uniform to CP"]
    labels = {"exponential x0.25": "Exp., 0.25 x", "exponential x0.5": "Exp., 0.5 x", "exponential x2": "Exp., 2 x", "exponential x4": "Exp., 4 x", "exponential x8": "Exp., 8 x", "uniform x2": "Uniform, 2 x", "uniform x4": "Uniform, 4 x", "uniform x8": "Uniform, 8 x", "uniform x16": "Uniform, 16 x", "uniform to CP": "Uniform, prefix"}
    grids = ["NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS", "DVB-T2 8K, PP3 (spacing 6)"]
    gshort = {"NR 20 MHz, DM-RS type 1": "NR type 1", "NR 20 MHz, DM-RS type 2": "NR type 2", "LTE 20 MHz, CRS": "LTE CRS", "DVB-T2 8K, PP3 (spacing 6)": "DVB-T2 (6)"}
    body = []
    for d in designs:
        cells = []
        for grid in grids:
            g = sub[(sub.grid == grid) & (sub.design == d)]
            cells.append(f"{g.loss_db.median():.1f} & {g.loss_db.max():.1f}")
        body.append(f"{labels[d]} & " + " & ".join(cells) + " \\\\")
    (TABLES / "mismatch.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Loss of a mismatched Wiener design relative to the matched filter, in dB, as median and maximum over the delay-spread sweep and the SNRs 5, 15 and 25 dB. The design profile is exponential with the actual delay spread scaled by the factor shown, or uniform over the actual delay spread scaled by the factor, or uniform over the cyclic prefix.}\n\\label{tab:mismatch}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{l cc cc cc cc}\n\\toprule\n & \\multicolumn{2}{c}{" + "} & \\multicolumn{2}{c}{".join(gshort[g] for g in grids) + "} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\nDesign & Med. & Max & Med. & Max & Med. & Max & Med. & Max \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def macros_from_ber(b: pd.DataFrame, out: dict) -> None:
    lte = b[(b.grid == "LTE 20 MHz, CRS") & (b.profile == "NIST automotive plant (132 ns)")]
    def ber(est, snr):
        return float(lte[(lte.estimator == est) & (lte.snr_db == snr)].qpsk_ber.iloc[0])
    out["BerLinearTwenty"] = f"{ber('linear (constant edges)', 20):.4f}"
    out["BerMatchedTwenty"] = f"{ber('Wiener (matched)', 20):.4f}"
    out["BerUniformTwenty"] = f"{ber('Wiener (uniform to 4688 ns)', 20):.4f}"
    out["BerNearestTwenty"] = f"{ber('nearest pilot', 20):.4f}"
    out["BerDftTwenty"] = f"{ber('DFT interpolation', 20):.4f}"
    out["BerLinearThirty"] = f"{ber('linear (constant edges)', 30):.4f}"
    out["BerNearestThirty"] = f"{ber('nearest pilot', 30):.4f}"
    out["BerDftThirty"] = f"{ber('DFT interpolation', 30):.4f}"
    out["BerUniformThirty"] = f"{ber('Wiener (uniform to 4688 ns)', 30):.4f}"
    out["BerMatchedThirty"] = f"{ber('Wiener (matched)', 30):.4f}"
    def snr_at(est, target=1e-2):
        g = lte[lte.estimator == est].sort_values("snr_db")
        x, y = g.snr_db.to_numpy(), np.log10(g.qpsk_ber.to_numpy())
        return float(np.interp(np.log10(target), y[::-1], x[::-1]))
    s_lin, s_mat, s_uni = snr_at("linear (constant edges)"), snr_at("Wiener (matched)"), snr_at("Wiener (uniform to 4688 ns)")
    out["BerSnrLinearAtOnePct"] = fmt(s_lin)
    out["BerSnrMatchedAtOnePct"] = fmt(s_mat)
    out["BerSnrUniformAtOnePct"] = fmt(s_uni)
    out["BerSnrGainMatchedVsLinear"] = fmt(s_lin - s_mat)
    out["BerCells"] = str(len(b))
    out["BerInsidePct"] = fmt(100 * (abs(b.mc_error_db - b.exact_error_db) <= b.mc_half_width_db).mean(), 0)
    out["BerMaxDevDb"] = f"{(b.mc_error_db - b.exact_error_db).abs().max():.2f}"


def macros_from_ber16(b: pd.DataFrame, out: dict) -> None:
    lte = b[(b.grid == "LTE 20 MHz, CRS") & (b.profile == "NIST automotive plant (132 ns)")]

    def ber(est, snr):
        return float(lte[(lte.estimator == est) & (lte.snr_db == snr)].ber.iloc[0])

    def snr_at(est, target=1e-2):
        g = lte[lte.estimator == est].sort_values("snr_db")
        x, y = g.snr_db.to_numpy(), np.log10(g.ber.to_numpy())
        return float(np.interp(np.log10(target), y[::-1], x[::-1]))

    out["BerSixteenLinearTwenty"] = f"{ber('linear (constant edges)', 20):.4f}"
    out["BerSixteenMatchedTwenty"] = f"{ber('Wiener (matched)', 20):.4f}"
    out["BerSixteenUniformTwenty"] = f"{ber('Wiener (uniform to 4688 ns)', 20):.4f}"
    out["BerSixteenLinearThirty"] = f"{ber('linear (constant edges)', 30):.4f}"
    out["BerSixteenMatchedThirty"] = f"{ber('Wiener (matched)', 30):.4f}"
    out["BerSixteenSnrLinearAtOnePct"] = fmt(snr_at("linear (constant edges)"))
    out["BerSixteenSnrMatchedAtOnePct"] = fmt(snr_at("Wiener (matched)"))
    out["BerSixteenSnrGainMatchedVsLinear"] = fmt(snr_at("linear (constant edges)") - snr_at("Wiener (matched)"))


def macros_from_per_tone(t: pd.DataFrame, out: dict) -> None:
    def db(est, tone):
        g = t[t.estimator == est].sort_values("subcarrier")
        return 10 * np.log10(float(g.error.iloc[tone]))

    def tones_to_interior(est):
        g = t[t.estimator == est].sort_values("subcarrier")
        e = 10 * np.log10(g.error.to_numpy())
        interior = np.median(e[300:900])
        return next(i for i in range(600) if abs(e[i] - interior) < 1.0)

    out["EdgeDftFirst"] = fmt(db("DFT interpolation", 0))
    out["EdgeDftTones"] = str(tones_to_interior("DFT interpolation"))
    out["EdgeMatchedFirst"] = fmt(db("Wiener (matched)", 0))
    out["EdgeMatchedCentre"] = fmt(db("Wiener (matched)", 600))
    out["EdgeMatchedTones"] = str(tones_to_interior("Wiener (matched)"))
    out["EdgeUniformFirst"] = fmt(db("Wiener (uniform to 4688 ns)", 0))
    out["EdgeUniformCentre"] = fmt(db("Wiener (uniform to 4688 ns)", 600))
    out["EdgeLinearFirst"] = fmt(db("linear (constant edges)", 0))
    lin = t[t.estimator == "linear (constant edges)"].sort_values("subcarrier").error.to_numpy()[300:900]
    out["EdgeLinearOscMin"] = fmt(10 * np.log10(lin.max()))
    out["EdgeLinearOscMax"] = fmt(10 * np.log10(lin.min()))


def macros_from_nist(out: dict) -> None:
    """Numbers from the NIST processing statistics and the measured-profile rows of the regime study."""
    folder = RESULTS / "nist_profiles"
    names = {"steam_plant": "Steam", "machine_shop": "Shop", "automotive_plant": "Auto"}
    reported = {"machine_shop": 64.9, "steam_plant": 35.4, "automotive_plant": 135.3}
    rows = []
    for key, short_name in names.items():
        f = folder / f"nist_profile_{key}_stats.csv"
        if not f.exists():
            continue
        st = pd.read_csv(f).iloc[0]
        out[f"Nist{short_name}Records"] = f"{int(st.records_used):,}"
        out[f"Nist{short_name}RmsFifteen"] = fmt(st.rms_delay_spread_15db_grubbs_mean_ns)
        out[f"Nist{short_name}RmsFifteenRaw"] = fmt(st.rms_delay_spread_15db_mean_ns)
        out[f"Nist{short_name}PeakToFloor"] = fmt(st.peak_to_floor_db_mean, 0)
        out[f"Nist{short_name}RmsThirty"] = fmt(st.rms_delay_spread_30db_mean_ns)
        out[f"Nist{short_name}RmsProfile"] = fmt(st.average_profile_rms_delay_spread_ns)
        out[f"Nist{short_name}Taps"] = str(int(st.average_profile_taps))
        out[f"Nist{short_name}SampleNs"] = fmt(1e3 / float(st.sample_rate_mhz))
        out[f"Nist{short_name}ChipNs"] = fmt(1e3 / float(st.sample_rate_mhz) * float(st.samples_per_chip), 0)
        out[f"Nist{short_name}RateMs"] = fmt(float(st.sample_rate_mhz), 0)
        out[f"Nist{short_name}Reported"] = fmt(reported[key])
        rows.append((key, st))
    if rows:
        out["NistSampleNs"] = fmt(1e3 / float(rows[0][1].sample_rate_mhz))
        out["NistChipNs"] = fmt(1e3 / float(rows[0][1].sample_rate_mhz) * float(rows[0][1].samples_per_chip), 0)
        out["NistDynamicDb"] = fmt(min(float(st.peak_to_floor_db_mean) for _, st in rows), 0)


def table_measured_vs_exponential(r: pd.DataFrame, out: dict) -> None:
    """Error on the measured profiles against the exponential site profiles, NR type 1 and LTE, 15 dB."""
    sites = r[(r.profile_kind == "site") & (r.snr_db == 15.0)]
    pairs = [("NIST steam plant", "steam"), ("NIST machine shop", "shop"), ("NIST automotive plant", "automotive")]
    ests = ["linear (constant edges)", "DFT interpolation", "Wiener (matched)"]
    lines = []
    diffs = []
    for prefix, label in pairs:
        for grid, gname in (("NR 20 MHz, DM-RS type 1", "NR type 1"), ("LTE 20 MHz, CRS", "LTE CRS")):
            g = sites[sites.grid == grid]
            measured = g[g.profile.str.startswith(prefix + ", measured")]
            expo = g[g.profile.str.startswith(prefix + " (")]
            if measured.empty or expo.empty:
                continue
            uni_m = measured[measured.estimator.str.startswith("Wiener (uniform")].error_db.iloc[0]
            uni_e = expo[expo.estimator.str.startswith("Wiener (uniform")].error_db.iloc[0]
            cells = []
            for est in ests:
                em = measured[measured.estimator == est].error_db.iloc[0]
                ee = expo[expo.estimator == est].error_db.iloc[0]
                cells.append(f"{em:.1f} & {ee:.1f}")
                diffs.append(em - ee)
            cells.append(f"{uni_m:.1f} & {uni_e:.1f}")
            diffs.append(uni_m - uni_e)
            ds_m = measured.rms_delay_spread_ns.iloc[0]
            ds_e = expo.rms_delay_spread_ns.iloc[0]
            lines.append(f"{label.capitalize()} ({ds_m:.0f} / {ds_e:.0f} ns) & {gname} & " + " & ".join(cells) + " \\\\")
    if not lines:
        return
    out["MeasVsExpMaxAbsDiff"] = fmt(max(abs(d) for d in diffs))
    out["MeasVsExpMeanDiff"] = fmt(sum(diffs) / len(diffs))
    groups = {"Interp": ["nearest pilot", "linear (constant edges)", "cubic spline (constant edges)"], "Wiener": ["Wiener (matched)"], "Dft": ["DFT interpolation"]}
    for gname, members in groups.items():
        vals = []
        for prefix, _ in pairs:
            for grid in ("NR 20 MHz, DM-RS type 1", "NR 20 MHz, DM-RS type 2", "LTE 20 MHz, CRS"):
                g = sites[sites.grid == grid]
                measured = g[g.profile.str.startswith(prefix + ", measured")]
                expo = g[g.profile.str.startswith(prefix + " (")]
                for est in members:
                    a = measured[measured.estimator == est]
                    b = expo[expo.estimator == est]
                    if not a.empty and not b.empty:
                        vals.append(abs(float(a.error_db.iloc[0]) - float(b.error_db.iloc[0])))
                if gname == "Wiener":
                    a = measured[measured.estimator.str.startswith("Wiener (uniform")]
                    b = expo[expo.estimator.str.startswith("Wiener (uniform")]
                    if not a.empty and not b.empty:
                        vals.append(abs(float(a.error_db.iloc[0]) - float(b.error_db.iloc[0])))
        if vals:
            out[f"MeasVsExp{gname}Max"] = fmt(max(vals), 2 if gname == "Interp" else 1)
    (TABLES / "measured.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Error in dB at 15 dB SNR on the measured factory profiles (M) and on the exponential profiles with the site delay spreads reported by NIST (E). The RMS delay spreads of the two profiles are given after the site name.}\n\\label{tab:measured}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{ll rr rr rr rr}\n\\toprule\n & & \\multicolumn{2}{c}{Linear} & \\multicolumn{2}{c}{DFT} & \\multicolumn{2}{c}{Wiener, matched} & \\multicolumn{2}{c}{Wiener, uniform prefix} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\nSite (M / E) & Map & M & E & M & E & M & E & M & E \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def macros_from_cost(c: pd.DataFrame, out: dict) -> None:
    lte = c[c.grid == "LTE 20 MHz, CRS"].set_index("estimator")
    out["CostLinearMults"] = f"{int(lte.loc['linear (constant edges)'].real_multiplications):,}"
    out["CostWienerMults"] = f"{int(lte.loc['Wiener (matched)'].real_multiplications):,}"
    out["CostDftMults"] = f"{int(lte.loc['DFT interpolation'].real_multiplications):,}"
    out["CostRatioWienerLinear"] = f"{lte.loc['Wiener (matched)'].real_multiplications / lte.loc['linear (constant edges)'].real_multiplications:.0f}"
    out["CostLinearUs"] = fmt(lte.loc["linear (constant edges)"].structured_apply_us_per_symbol, 0)
    out["CostWienerUs"] = fmt(lte.loc["Wiener (matched)"].dense_apply_us_per_symbol, 0)
    out["CostDftUs"] = fmt(lte.loc["DFT interpolation"].structured_apply_us_per_symbol, 0)
    out["CostWienerDesignMs"] = fmt(lte.loc["Wiener (matched)"].design_ms, 0)
    body = []
    for grid in ["NR 20 MHz, DM-RS type 1", "LTE 20 MHz, CRS", "DVB-T2 8K, PP3 (spacing 6)"]:
        g = c[c.grid == grid]
        gname = {"NR 20 MHz, DM-RS type 1": "NR type 1", "LTE 20 MHz, CRS": "LTE CRS", "DVB-T2 8K, PP3 (spacing 6)": "DVB-T2 (6)"}[grid]
        for _, row in g.iterrows():
            s = "--" if np.isnan(row.structured_apply_us_per_symbol) else f"{row.structured_apply_us_per_symbol:.0f}"
            body.append(f"{gname} & {short(row.estimator)} & {int(row.real_multiplications):,} & {row.design_ms:.1f} & {row.dense_apply_us_per_symbol:.0f} & {s} \\\\")
    (TABLES / "cost.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Cost per OFDM symbol: real multiplications of a straightforward implementation, one-time design time of the matrix, and measured NumPy application time per symbol as a stored matrix (dense) and as a structured implementation where one exists. Times on the author's machine, indicative only.}\n\\label{tab:cost}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{llrrrr}\n\\toprule\nGrid & Estimator & Mults & Design & Dense & Struct. \\\\\n & & & (ms) & (\\si{\\micro\\second}) & (\\si{\\micro\\second}) \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def macros_from_maps(small: pd.DataFrame, maps: pd.DataFrame, out: dict) -> None:
    small = small.copy()
    small["above"] = small.error_db - small.optimum_db
    anc = small[small["map"] == "anchored"].above
    cen = small[small["map"] == "centred"].above
    out["SmallGrids"] = str(small.groupby(["n_tones", "n_pilots"]).ngroups)
    out["SmallCells"] = str(len(cen))
    out["SmallCentredMax"] = fmt(cen.max(), 2)
    out["SmallCentredMedian"] = fmt(cen.median(), 2)
    out["SmallAnchoredMax"] = fmt(anc.max(), 1)
    out["SmallAnchoredMedian"] = fmt(anc.median(), 1)
    out["SmallMapsMax"] = f"{int(small.maps_enumerated.max()):,}"
    clean = maps[maps.scenario == "clean band"]
    inter = maps[maps.scenario != "clean band"]
    def val(df, m):
        return df[df["map"] == m].objective_db
    out["MapCleanCentredVsAnchored"] = fmt((val(clean, "anchored").values - val(clean, "centred").values).max(), 1)
    out["MapCleanGaVsCentred"] = fmt((val(clean, "centred").values - val(clean, "genetic algorithm").values).max(), 2)
    best_rule = np.minimum(val(clean, "anchored").values, val(clean, "centred").values)
    out["MapCleanGaVsBestRule"] = fmt((best_rule - val(clean, "genetic algorithm").values).max(), 2)
    best_moved = inter.groupby(["rms_delay_spread_ns", "scenario"]).apply(lambda g: g[g["map"].str.contains("moved")].objective_db.min(), include_groups=False)
    ga = inter[inter["map"] == "genetic algorithm"].set_index(["rms_delay_spread_ns", "scenario"]).objective_db
    gain = (best_moved - ga).dropna()
    out["MapInterGaGainMin"] = fmt(gain.min(), 1)
    out["MapInterGaGainMax"] = fmt(gain.max(), 1)
    out["MapInterCentredLoss"] = fmt((inter[inter["map"] == "centred"].set_index(["rms_delay_spread_ns", "scenario"]).objective_db - ga).max(), 1)
    out["MapGaSeconds"] = fmt(inter[inter["map"] == "genetic algorithm"].ga_seconds.max(), 1)
    out["MapGaEvaluations"] = f"{int(inter[inter['map'] == 'genetic algorithm'].ga_maps_evaluated.max()):,}"
    body = []
    sub = maps[maps.rms_delay_spread_ns == maps.rms_delay_spread_ns.max()]
    for scenario in sub.scenario.unique():
        g = sub[sub.scenario == scenario].set_index("map").objective_db
        cells = [f"{g.get(m, np.nan):.1f}" if not np.isnan(g.get(m, np.nan)) else "--" for m in ["anchored", "centred", "anchored, moved off interferer", "centred, moved off interferer", "genetic algorithm"]]
        label = scenario.replace(" (+20 dB)", "").replace("interferer on tones ", "interferer, tones ").replace("clean band", "clean band")
        body.append(label + " & " + " & ".join(cells) + " \\\\")
    (TABLES / "maps.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Free pilot maps on the 64-tone link with 8 pilots and linear interpolation. Mean exact error over 10 and 20 dB SNR, in dB, at the largest delay spread of the study, for the anchored (A) and centred (C) uniform maps, the same maps with pilots moved off the interfered tones (A moved, C moved), and the genetic-algorithm map (GA).}\n\\label{tab:maps}\n\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{lrrrrr}\n\\toprule\nScenario & A & C & A moved & C moved & GA \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out["MapDelaySpread"] = fmt(maps.rms_delay_spread_ns.max(), 0)


def setup_table() -> None:
    (TABLES / "setup.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Grids and pilot maps of the study (DVB-T2 in 8K mode). Columns: subcarrier spacing, active subcarriers analysed, pilots, pilot spacing in subcarriers and in kHz, and cyclic prefix (CP) or guard interval (GI).}\n\\label{tab:setup}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lrrrrr}\n\\toprule\nGrid and map & $\\Delta f$ (kHz) & $N$ & $K$ & Spacing & CP (\\si{\\micro\\second}) \\\\\n\\midrule\n"
        "NR, DM-RS type 1 & 30 & 612 & 306 & 2 (60 kHz) & 2.34 \\\\\n"
        "NR, DM-RS type 2 & 30 & 612 & 204 & pairs (180 kHz) & 2.34 \\\\\n"
        "LTE, CRS port 0 & 15 & 1200 & 200 & 6 (90 kHz) & 4.69 \\\\\n"
        "DVB-T2 PP1, GI 1/4 & 1.116 & 1024 & 342 & 3 (3.3 kHz) & 224 \\\\\n"
        "DVB-T2 PP3, GI 1/8 & 1.116 & 1024 & 171 & 6 (6.7 kHz) & 112 \\\\\n"
        "DVB-T2 PP5, GI 1/16 & 1.116 & 1024 & 86 & 12 (13.4 kHz) & 56 \\\\\n"
        "DVB-T2 PP7, GI 1/128 & 1.116 & 1024 & 43 & 24 (26.8 kHz) & 7 \\\\\n"
        "Proprietary link & 312.5 & 64 & 8 & free & 0.8 \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main() -> None:
    out = {}
    test_count = subprocess.run(["python", "-m", "pytest", "--collect-only", "-q", str(ROOT / "ofdmest" / "tests")], capture_output=True, text=True, cwd=ROOT / "ofdmest")
    n_tests = sum(1 for line in test_count.stdout.splitlines() if "::" in line)
    out["TestCount"] = str(n_tests)
    out["RepositoryUrl"] = "https://github.com/aalbata/ofdm-estimator-selection"
    macros_from_validation(pd.read_csv(RESULTS / "validation.csv"), out)
    regime = pd.read_csv(RESULTS / "regime.csv")
    macros_from_regime(regime, out)
    macros_from_nist(out)
    table_measured_vs_exponential(regime, out)
    if (RESULTS / "mismatch.csv").exists():
        macros_from_mismatch(pd.read_csv(RESULTS / "mismatch.csv"), out)
    if (RESULTS / "ber.csv").exists():
        macros_from_ber(pd.read_csv(RESULTS / "ber.csv"), out)
    if (RESULTS / "ber_16qam.csv").exists():
        macros_from_ber16(pd.read_csv(RESULTS / "ber_16qam.csv"), out)
    if (RESULTS / "per_tone.csv").exists():
        macros_from_per_tone(pd.read_csv(RESULTS / "per_tone.csv"), out)
    if (RESULTS / "cost.csv").exists():
        macros_from_cost(pd.read_csv(RESULTS / "cost.csv"), out)
    if (RESULTS / "pilot_maps.csv").exists():
        macros_from_maps(pd.read_csv(RESULTS / "pilot_maps_small.csv"), pd.read_csv(RESULTS / "pilot_maps.csv"), out)
    setup_table()
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(out.items())]
    (TABLES / "numbers.tex").write_text("% Generated by experiments/make_tables.py from the result files. Do not edit.\n" + "\n".join(lines) + "\n")
    used = set()
    for tex in PAPER.glob("*.tex"):
        if tex.name == "numbers.tex":
            continue
        used |= set(re.findall(r"\\([A-Z][A-Za-z]+)(?=\{\}|[^A-Za-z])", tex.read_text()))
    latex_names = {"IEEEmembership", "IEEEkeywords", "SI", "ValCells"}
    missing = sorted(u for u in used if u not in out and u not in latex_names and u[0].isupper() and len(u) > 4 and u not in {"Delta", "Sigma", "Fig", "Table", "Section", "Equation", "RMS", "SNR", "OFDM"})
    missing = [m for m in missing if m not in {"IEEEtran", "LaTeX", "Matplotlib", "NumPy", "SciPy", "Monte", "Carlo", "Wiener"}]
    unknown = [m for m in missing if PAPER.exists() and re.search(r"\\" + m + r"\{\}", "".join(t.read_text() for t in PAPER.glob("*.tex")))]
    if unknown:
        raise SystemExit(f"macros used in the paper but not generated: {unknown}")
    print(f"wrote {TABLES / 'numbers.tex'} with {len(out)} macros and {len(list(TABLES.glob('*.tex'))) - 1} tables")


if __name__ == "__main__":
    main()
