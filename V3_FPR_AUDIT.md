# V3 False-Positive Audit

## Frozen calibration behavior

The 500 MBPP canonical negatives were deterministically split before fitting into 250 calibration and 250 held-out rows. Whitening and the empirical p-value threshold used only the calibration half.

- Primary-key positives on calibration rows: 11/250 = 4.4%
- Task bootstrap 95% interval: [2.0%, 7.2%]
- Wrong-key positives on calibration rows: 1/250 = 0.4%
- Wrong-key positives on V3 pilot outputs: 1/30 = 3.33%
- Target FPR: 5%

The 4.4% figure is not an unbiased held-out FPR estimate: these rows define the empirical null and whitening transform. It is reported only as a calibration consistency check. The held-out file was not opened because the pilot exact-replay hard gate failed, so held-out FPR is **not evaluated** and no general FPR claim is supported.

The empirical p-value is `(1 + count(calibration_score >= observed_score)) / 251`, with conservative greater-than-or-equal tie handling and a minimum of three independent units.
