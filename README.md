# Exact error analysis of pilot-aided OFDM channel estimators

Repository: <https://github.com/aalbata/ofdm-estimator-selection>

This repository holds the `ofdmest` library and the scripts, result files and figures of the
study that selects the channel estimator of a pilot-aided OFDM receiver from an exact error
model, with factory channels as the test case.

A pilot-aided OFDM receiver turns a few pilot measurements into a channel estimate for every
subcarrier. The standards fix the pilot positions; the estimator (nearest pilot, linear or
spline interpolation, a transform-domain estimator, or a Wiener filter built on an assumed
delay profile) is the designer's choice. Every one of these is a linear map from the pilot
observations to the estimate, so its expected error follows exactly from the pilot map, the
second-order channel statistics and the per-tone noise level. The library computes that error
in closed form and validates it against Monte Carlo simulation; the study applies it to the
5G NR, LTE and DVB-T2 pilot maps at the delay spreads measured by NIST in three factory halls
and given by the 3GPP indoor factory model.

## Layout

| Path | Content |
|---|---|
| `ofdmest/` | The library (installable package, tests, package README) |
| `experiments/` | Scripts that produced every result file, figure and table of the paper |
| `results/` | Result files written by the experiment scripts (CSV) |
| `figures/` | Figures of the paper (PDF and PNG), produced by `experiments/make_figures.py` |
| `reproduce.py` | Regenerates figures and tables from the shipped results, or everything from scratch with `--full` |

## Installation and reproduction

Tested on Windows 11 with Python 3.12.10. From the repository root:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e ofdmest
python -m pytest ofdmest/tests -q
python reproduce.py
```

`python reproduce.py` rebuilds the figures and the paper's tables and inline numbers from the
shipped result files in a few seconds. `python reproduce.py --full` reruns every experiment
first; it takes about an hour on a laptop and writes the result files anew. All experiments
are deterministic given their recorded seeds and the four linear-algebra threads that
`reproduce.py` sets, except `results/cost.csv` and the `ga_seconds` column of
`results/pilot_maps.csv`, whose measured run times depend on the machine.

## Results

Error of four estimators at the factory profiles, from `results/regime.csv` and
`results/mismatch.csv` (normalised mean squared error of the channel estimate in dB, lower is
better; 20 MHz LTE grid, cell-specific reference signals, 15 dB SNR):

| Profile | Matched Wiener | Wiener, uniform over 8 x delay spread | Wiener, uniform over the prefix | Linear interpolation |
|---|---|---|---|---|
| Steam plant, 36 ns | -29.1 | -28.7 | -18.3 | -16.7 |
| Machine shop, 74 ns | -26.8 | -26.3 | -18.4 | -16.7 |
| Automotive plant, 132 ns | -24.9 | -24.2 | -18.4 | -16.7 |

Exact error against Monte Carlo (`results/validation.csv`): 408 cells, 96.3 percent of the
exact values inside the 95 percent interval of the simulation, largest relative deviation
2.3 percent.

## Measured factory profiles

`results/nist_profiles/` holds the average power-delay profiles derived from the public NIST
industrial wireless channel-sounder measurements (doi:10.18434/T44S3N), one run per site at
2.245 GHz with vertical polarisation, produced by `experiments/process_nist.py` from the
original MATLAB 7.3 files (1.5 to 5.8 GB each, not shipped). Reproducing those files needs
the `h5py` package (version 3.16.0 was used) and the three NIST files named in
`results/nist_profiles/README.md`; everything else in the study runs from the shipped
profiles.

Statistics of the three runs (from the `_stats.csv` files):

| Site | Records used | RMS delay spread, 15 dB window, after outlier test (ns) | Reported by NIST for the run (ns) | RMS delay spread, all kept samples (ns) | RMS delay spread of the averaged profile (ns) | Taps |
|---|---|---|---|---|---|---|
| Steam plant | 12,815 | 32.6 | 35.4 | 44.8 | 54.8 | 76 |
| Machine shop | 11,099 | 65.3 | 64.9 | 82.4 | 93.4 | 53 |
| Automotive plant | 34,881 | 116.7 | 135.3 | 184.3 | 150.0 | 73 |

## Result file columns

`results/regime.csv`: `grid` and `map` (pilot map), `n_tones`, `n_pilots`, `profile_kind`
(`sweep` for the exponential delay-spread sweep, `site` for the factory profiles), `profile`,
`rms_delay_spread_ns`, `snr_db`, `estimator`, `error` (normalised mean squared error over the
active tones) and `error_db`, `bias` (part of the error the estimator cannot follow) and
`noise` (part let through from the pilot noise), `edge_error` (mean over the two band-edge
tones), `interior_error` (mean over the central half of the band), `real_multiplications`
per OFDM symbol.

`results/mismatch.csv`: `grid`, `profile`, `rms_delay_spread_ns`, `snr_db`, `design`
(design profile of the Wiener filter), `design_scale_ns` (span or delay spread of the design
profile), `error_db`, `matched_db` (error of the matched filter), `loss_db` (difference),
`bias`, `noise`.

`results/validation.csv`: `map`, `n_tones`, `n_pilots`, `rms_delay_spread_ns`, `snr_db`,
`estimator`, `exact_mse`, `mc_mse` (Monte Carlo mean over 10,000 symbols),
`mc_half_width_95`, `relative_error_percent`, `inside_95`, `mc_ber` (QPSK bit error rate).

`results/ber.csv` and `results/ber_16qam.csv`: `grid`, `profile`, `rms_delay_spread_ns`,
`snr_db`, `estimator`, `exact_error_db`, `mc_error_db`, `mc_half_width_db`, `modulation`,
`ber`, `bit_errors`, `bits`, `n_symbols`, `seed`.

`results/cost.csv`: `grid`, `n_tones`, `n_pilots`, `estimator`, `real_multiplications`,
`design_ms` (time to build the estimator matrix), `dense_apply_us_per_symbol`,
`structured_apply_us_per_symbol` (where a structured implementation exists), `machine`.

`results/pilot_maps_small.csv`: `n_tones`, `n_pilots`, `rms_delay_spread_samples`, `snr_db`,
`map` (anchored or centred rule), `indices`, `error_db`, `optimum_db` and `optimum_indices`
(exhaustive optimum), `maps_enumerated`.

`results/pilot_maps.csv`: `rms_delay_spread_ns`, `scenario`, `map`, `indices`,
`objective_db` (mean exact error over 10 and 20 dB SNR), `ga_seconds`, `ga_maps_evaluated`.

`results/per_tone.csv`: `estimator`, `subcarrier`, `error`, `bias`, `noise` for the LTE grid,
automotive-plant profile, 15 dB SNR.

## Licence

MIT. Please cite the accompanying paper and this repository when you use the code.
