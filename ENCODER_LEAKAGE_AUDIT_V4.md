# V4 Encoder and Training-Leakage Audit

## Formal mechanism

The formal V4 signed representation does not load or call a neural encoder. It is
constructed from Python's standard-library AST, normalized identifier slots,
statement role, and bounded previous-to-current data-flow edges. Consequently:

- no encoder checkpoint influences V4 representation or signature evidence;
- no batch-dependent embedding is stored or supplied to R3;
- no model training corpus is consulted by the formal detector;
- encoder checkpoint leakage cannot explain the V4 pilot statistic.

The preregistration retains the V3 CodeT5 checkpoint identity for diagnosis and
candidate comparison, but Candidate C was selected precisely because the neural
path could not meet both exactness and cost.

## Dataset leakage scope

The V4 negative split audit checked exact IDs, normalized code, canonical AST,
V3 MBPP-test overlap, and HumanEval prompt/final-code overlap. No overlap was
found after removing five conflicting or duplicate rows. A definitive historical
training-corpus membership proof is unavailable for public CodeT5 or the code
generation model; this limitation is disclosed rather than treated as zero risk.

Because V4 uses frozen benchmark candidate pools and a single model/seed family,
the result supports only within-benchmark mechanism validation. It does not
support cross-model, cross-language, or natural-code generalization.
