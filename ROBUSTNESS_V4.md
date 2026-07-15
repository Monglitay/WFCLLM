# V4 Robustness Status

Robustness attacks were **not run**. The preregistration allowed formatting,
identifier-renaming, dead-code insertion, independent-statement deletion, and
independent-statement reordering only after clean pilot, held-out, and full gates.
The pilot failed TPR-improvement and paired correctness gates, so running attacks
would have violated the escalation policy.

The only clean statistics available are the frozen pilot results: V4 TPR 0/30,
pass 15/30, mean score 0.070407573, mean independent units 5.8667, and mean
structural erasure rate 0.0636905. There are no attacked statistics, attacked TPRs,
degradation estimates, attack-specific exact replay values, or empirical
independent-unit changes.

Unit tests cover canonical formatting and identifier normalization contracts, and
Stage E covers batch/order/cache invariance. Those engineering tests are not a
substitute for the forbidden confirmatory robustness experiment. No successful
subset is reported.
