"""Average power-delay profiles from the NIST industrial wireless channel-sounder files.

The post-processed NIST files (MATLAB 7.3, HDF5) hold the dataset ``IQdata`` with one complex
impulse response per record (rows) over one code period (columns, 8,188 samples at the
sample rate given in ``Strct_Metadata/SampleRate_MHz_num``), the per-record range in
``IQdata_Range_m`` (row 3, metres), and the acquisition metadata in ``Strct_Metadata``.

Sample selection follows the analysis code published by NIST with the data (repository
usnistgov/IndustrialWirelessAnalysis, files select_cir_samples.m and compute_delay_spread.m):
each record is truncated to its first 80 percent, its power is normalised to the peak, the
noise floor is the mean normalised power over the last fifth of the truncated record, and
the samples kept are those at least 10 dB above the floor and within 30 dB of the peak;
records closer than 3 m are skipped. The RMS delay spread is reported in two forms: as in the
NIST code, over the kept samples within 15 dB of the peak, which makes it comparable with
the values of NIST Technical Note 1951 (also after the iterative Grubbs outlier test that the
NIST code applies before reporting), and over all kept samples.

For the average profile of a run, the kept samples of each record are aligned so that the
first kept sample is at zero delay, normalised to unit power, and averaged over the records;
the average is truncated where it falls 30 dB below its own peak.

Usage:
    python -I process_nist.py inspect <file.mat>
    python -I process_nist.py process <site label> <file.mat> <output folder> [--every K]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

TRUNCATE_FRACTION = 0.8
FLOOR_FRACTION = 0.8
FLOOR_MARGIN = 10.0
PEAK_WINDOW = 1.0 / 1000.0
SPREAD_WINDOW = 1.0 / 31.6
MIN_RANGE_M = 3.0
PROFILE_CUTOFF = 1.0 / 1000.0
MAX_DELAY_S = 2.0e-6


def _text(ds) -> str:
    arr = np.asarray(ds[()]).reshape(-1)
    return "".join(chr(int(c)) for c in arr if 0 < int(c) < 65536)


def metadata(f) -> dict:
    g = f["Strct_Metadata"]
    out = {}
    for key in g:
        ds = g[key]
        cls = ds.attrs.get("MATLAB_class", b"")
        cls = cls.decode() if isinstance(cls, bytes) else str(cls)
        if cls == "char":
            out[key] = _text(ds)
        elif cls == "double" and ds.size == 1:
            out[key] = float(np.asarray(ds[()]).reshape(-1)[0])
    return out


def inspect(path: Path) -> None:
    with h5py.File(path, "r") as f:
        d = f["IQdata"]
        print("IQdata:", d.shape, d.dtype, "chunks", d.chunks)
        print("IQdata_Range_m:", f["IQdata_Range_m"].shape)
        for k, v in metadata(f).items():
            print(f"  {k}: {v}")


def grubbs_filtered(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Iterative two-sided Grubbs test as in NIST's deleteoutliers.m (alpha 0.05)."""
    from scipy.stats import t as student_t

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    while x.size > 2:
        n = x.size
        mean, std = x.mean(), x.std(ddof=1)
        if std == 0:
            break
        dev = np.abs(x - mean) / std
        tcrit = student_t.ppf(1 - alpha / (2 * n), n - 2)
        gcrit = (n - 1) / np.sqrt(n) * np.sqrt(tcrit**2 / (n - 2 + tcrit**2))
        worst = int(np.argmax(dev))
        if dev[worst] > gcrit:
            x = np.delete(x, worst)
        else:
            break
    return x


def spread(p: np.ndarray, sample_s: float) -> tuple[float, float]:
    """Mean delay and RMS delay spread of a power vector on the sample grid."""
    tau = np.arange(p.size) * sample_s
    total = p.sum()
    mean = float((p * tau).sum() / total)
    rms = float(np.sqrt(max((p * tau**2).sum() / total - mean**2, 0.0)))
    return mean, rms


def record_profile(h: np.ndarray, sample_s: float):
    """NIST sample selection on one record; returns the aligned normalised profile and statistics."""
    n = int(round(h.size * TRUNCATE_FRACTION))
    p = np.abs(h[:n]) ** 2
    peak = p.max()
    if not np.isfinite(p).all() or peak <= 0:
        return None
    p = p / peak
    floor = float(np.mean(p[int(round(n * FLOOR_FRACTION)):]))
    keep = (p > floor * FLOOR_MARGIN) & (p > PEAK_WINDOW)
    if keep.sum() < 2:
        return None
    kept = np.where(keep, p, 0.0)
    _, rms_all = spread(kept, sample_s)
    narrow = np.where(kept >= SPREAD_WINDOW, kept, 0.0)
    _, rms_nist = spread(narrow, sample_s)
    start = int(np.flatnonzero(keep)[0])
    aligned = kept[start:]
    return aligned / aligned.sum(), rms_nist, rms_all, int(keep.sum()), 10 * np.log10(1.0 / floor)


def process(label: str, path: Path, out_dir: Path, every: int = 1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        meta = metadata(f)
        sample_s = 1e-6 / meta["SampleRate_MHz_num"]
        d = f["IQdata"]
        ranges = np.asarray(f["IQdata_Range_m"][()])[2]
        n_records = d.shape[0]
        acc = np.zeros(int(MAX_DELAY_S / sample_s) + 1)
        count = skipped_range = 0
        rms_nist, rms_all, kept_counts, dynamic = [], [], [], []
        for row in range(0, n_records, every):
            if ranges[row] < MIN_RANGE_M:
                skipped_range += 1
                continue
            raw = d[row]
            res = record_profile(raw["real"] + 1j * raw["imag"], sample_s)
            if res is None:
                continue
            prof, r_nist, r_all, kept, dyn = res
            n = min(prof.size, acc.size)
            acc[:n] += prof[:n]
            count += 1
            rms_nist.append(r_nist)
            rms_all.append(r_all)
            kept_counts.append(kept)
            dynamic.append(dyn)
    if count == 0:
        raise SystemExit("no usable records")
    avg = acc / count
    last = int(np.flatnonzero(avg >= avg.max() * PROFILE_CUTOFF)[-1])
    avg = avg[: last + 1]
    avg = avg / avg.sum()
    mean_delay, rms_avg = spread(avg, sample_s)
    tau = np.arange(avg.size) * sample_s
    with (out_dir / f"nist_profile_{label}_records.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rms_delay_spread_15db_ns", "rms_delay_spread_30db_ns", "kept_samples", "peak_to_floor_db"])
        for a, b, c, dd in zip(rms_nist, rms_all, kept_counts, dynamic):
            w.writerow([f"{1e9*a:.4f}", f"{1e9*b:.4f}", c, f"{dd:.3f}"])
    filtered = grubbs_filtered(np.array(rms_nist))
    with (out_dir / f"nist_profile_{label}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["delay_s", "power"])
        for t, pw in zip(tau, avg):
            w.writerow([f"{t:.10e}", f"{pw:.10e}"])
    stats = {
        "site": label,
        "file": path.name,
        "location": meta.get("Location_str", ""),
        "frequency_ghz": meta.get("Frequency_GHz_num", float("nan")),
        "sample_rate_mhz": meta.get("SampleRate_MHz_num", float("nan")),
        "code_length_chips": meta.get("BasePNCcodeLength_num", float("nan")),
        "samples_per_chip": meta.get("PNOversample_num", float("nan")),
        "records_in_file": n_records,
        "records_used": count,
        "records_skipped_range": skipped_range,
        "record_step": every,
        "rms_delay_spread_15db_mean_ns": 1e9 * float(np.mean(rms_nist)),
        "rms_delay_spread_15db_median_ns": 1e9 * float(np.median(rms_nist)),
        "rms_delay_spread_15db_grubbs_mean_ns": 1e9 * float(np.mean(filtered)),
        "records_after_grubbs": int(filtered.size),
        "rms_delay_spread_30db_mean_ns": 1e9 * float(np.mean(rms_all)),
        "kept_samples_mean": float(np.mean(kept_counts)),
        "peak_to_floor_db_mean": float(np.mean(dynamic)),
        "average_profile_taps": int(avg.size),
        "average_profile_rms_delay_spread_ns": 1e9 * rms_avg,
        "average_profile_mean_delay_ns": 1e9 * mean_delay,
    }
    with (out_dir / f"nist_profile_{label}_stats.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats))
        w.writeheader()
        w.writerow(stats)
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "inspect":
        inspect(Path(sys.argv[2]))
    elif len(sys.argv) >= 5 and sys.argv[1] == "process":
        step = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
        process(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), step)
    else:
        print(__doc__)
