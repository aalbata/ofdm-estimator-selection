# ofdmest

Exact error analysis of linear pilot-aided OFDM channel estimators.

A pilot-aided OFDM receiver turns a few pilot measurements into a channel estimate for
every subcarrier. The estimator is the receiver designer's choice: nearest pilot, linear or
cubic spline interpolation, a transform-domain (DFT) estimator, or a Wiener filter built on
an assumed delay profile. Every one of these is a linear map from the pilot observations to
the estimate, so its expected error follows exactly from the pilot map, the second-order
channel statistics (the power-delay profile) and the per-tone noise level. This package
computes that error in closed form, in milliseconds, for standard pilot maps (5G NR DM-RS,
LTE CRS, DVB-T2 scattered pilots) and custom maps, and validates it against Monte Carlo
simulation. It also compares the computational cost of the estimators, recommends the
cheapest estimator that meets an error target, and optimises free pilot maps for
proprietary links, including maps that avoid tones hit by narrowband interference.

The analysis covers one OFDM symbol (block fading within the symbol, cyclic prefix longer
than the delay span) and uses second-order statistics only. Bit error rates are measured by
simulation, not computed in closed form.

## Installation

```
pip install -e .
```

Requirements: Python 3.12, NumPy, SciPy, pandas and Matplotlib at the versions pinned in
`pyproject.toml`. Tested on Windows 11 with Python 3.12.10.

## First result in one minute

Exact error of six estimators on a 20 MHz LTE grid with a 200 ns exponential delay profile:

```
ofdmest error --grid lte:100 --profile exp:200ns --snr 0,15,30
```

Cheapest estimator within 1 dB of the matched Wiener filter in a factory hall (3GPP indoor
factory model, line of sight, 48,000 m3 hall with 14,560 m2 of surface):

```
ofdmest recommend --grid nr:51:30 --profile inf:48000:14560:los --snr 5,20
```

Error over a delay-spread and SNR grid, written to a CSV file:

```
ofdmest sweep --grid lte:100 --ds 20ns:2us:12 --snr 0:30:7 --out sweep.csv
```

Check the exact values against Monte Carlo simulation and report the QPSK bit error rate:

```
ofdmest validate --grid custom:64:312500:64:0.8us --map centred:8 --profile exp:100ns --snr 15
```

Optimise an 8-pilot map on a 64-tone proprietary link with an interferer on tones 20 to 27:

```
ofdmest search --grid custom:64:312500:64:0.8us --pilots 8 --profile exp:200ns --snr 10,20 --interference 20:28:20
```

## Using the library

```python
from ofdmest import OfdmGrid, DelayProfile, lte_crs, LinearInterpolation, WienerFilter, exact_error

grid = OfdmGrid(1200, 15e3, fft_size=2048, cp_duration_s=4.69e-6)
pilots = lte_crs(100)
profile = DelayProfile.exponential(200e-9, grid.sample_period_s, max_delay_s=3e-6)
for est in (LinearInterpolation(), WienerFilter()):
    res = exact_error(est, grid, pilots, profile, snr_db=15.0)
    print(est.name, round(res.total_db, 2), "dB")
```

`exact_error` returns the per-tone error as well, split into the channel-mismatch part
(what the estimator cannot follow) and the noise part (what it lets through). A measured
delay profile is given as `DelayProfile(delays_s, powers)`; tones hit by interference are
given as an array of extra noise levels in dB.

## Grid presets of the command line

| Preset | Meaning |
|---|---|
| `nr:<prb>[:<scs_khz>[:<type>]]` | 5G NR with the given resource blocks, subcarrier spacing (default 30 kHz) and DM-RS type 1 or 2 (3GPP TS 38.211) |
| `lte:<prb>` | LTE with cell-specific reference signals of port 0 (3GPP TS 36.211) |
| `dvbt2:<fft_k>:<pattern>[:<gi>]` | DVB-T2 scattered pilots PP1 to PP8 of one symbol (ETSI EN 302 755); large symbols are analysed over a window set with `--window` |
| `custom:<n>:<spacing_hz>:<fft>[:<cp>]` | any grid; give the map with `--map` |

Estimator names: `nearest`, `linear` (constant edges) or `linear:linear`, `spline`,
`dft[:window]`, `ridge[:lambda]`, `robust[:span]` (Wiener with a uniform profile), `wiener`
(matched) or `wiener:<delay spread>[:<design SNR>]` (mismatched design).

## Reproducing the study

The `experiments` folder holds the scripts that produced every table and figure of the
accompanying paper; `reproduce.py` regenerates them from the shipped result files (fast
path) or from scratch (`--full`).

## Licence

MIT. Please cite the accompanying paper and this repository when you use the package.
