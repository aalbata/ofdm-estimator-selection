# Measured factory profiles

Average power-delay profiles derived from three runs of the public NIST industrial wireless
channel-sounder measurements, "Radio Frequency Measurements for Selected Manufacturing and
Industrial Environments" (NIST, 2016, doi:10.18434/T44S3N), documented in NIST Technical
Note 1951. The runs are at 2.245 GHz with vertically polarised omnidirectional antennas;
each record is a complex impulse response over one 2,047-chip code period with four samples
per chip, sampled at 80 MS/s (12.5 ns per sample) in the machine-shop and automotive runs and
at 200 MS/s (5 ns per sample) in the steam-plant run.

| Site | File (folder active_meas of the NIST bucket) | Size (bytes) |
|---|---|---|
| Machine shop, NIST Gaithersburg | GBurgD3_2GHz_TX2low_vpol_run1_pp.mat | 1,463,535,322 |
| Steam plant, NIST Boulder | Tx2 2G Boiler Vpol run2_pp.mat | 1,960,158,015 |
| Automotive assembly plant | AAPlantD1_2GHz_TX1_vpol_run3_pp.mat | 5,800,724,343 |

Files are at https://s3.amazonaws.com/nist-ics-wireless/measurements/active_meas/ (spaces in
the names percent-encoded). They are not shipped. `experiments/process_nist.py` produced the
profiles here with the sample selection of NIST's published analysis code (record truncated
to its first 80 percent, noise floor as the mean power of the last fifth, samples kept 10 dB
above the floor and within 30 dB of the peak, records closer than 3 m skipped), aligned on
the first kept sample, normalised and averaged over the records, then truncated 30 dB below
the peak of the average. The `_stats.csv` files give the per-record RMS delay spreads, in
NIST's 15 dB form (comparable with Technical Note 1951) and over all kept samples.

SHA-256 of the downloaded files:

| File | SHA-256 |
|---|---|
| GBurgD3_2GHz_TX2low_vpol_run1_pp.mat | a81e5f8c843e0cd98f646674f430b25e11c002e55446bed6a0772e1910203353 |
| Tx2 2G Boiler Vpol run2_pp.mat | 38bafbc0fe935b897b01b6865c839da53da90d30a18aa8152014653181f2bfcf |
| AAPlantD1_2GHz_TX1_vpol_run3_pp.mat | 2dff3aeb4ff8665a346fa2b14f3003a3b60d3d9ba265f55722a70af365718dc2 |
