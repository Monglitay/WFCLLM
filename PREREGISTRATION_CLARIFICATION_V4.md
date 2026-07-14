# V4 Preregistration Clarification: Key-Independent Calibration Reference

Date: 2026-07-14

Status: frozen before calibration fitting, debug, pilot, held-out access, or any
formal V4 outcome.

## Ambiguity resolved

The original preregistration correctly prohibited public private-key
fingerprints and key SHA-256 values, but did not say whether the full public
calibration score vector could be computed with the primary private key. Such a
vector, paired with the public calibration corpus, could help confirm a guessed
key and would violate the stronger no-key-confirmation requirement.

## Frozen resolution

The public calibration null distribution is fitted with a deterministic public
reference derived only from the domain
`v4-public/calibration-null-reference`. It is not a private key and is unrelated
to the V4 primary or wrong-control key. Public calibration artifacts therefore
contain no primary-key-dependent score vector.

The formal primary private key remains new and independent. It is used only for
V4 candidate selection and R3 detection. Calibration negative rate under the
primary key and the domain-separated wrong-key control remain hard pilot gates.
The empirical p-value formula, target FPR, split, minimum independent units,
bit count, candidate mechanism, pilot IDs, retry count, and all escalation
gates are unchanged.

## Timing and evidence isolation

This clarification was made while formal tests were red/green and before any
calibration fit or formal debug/pilot execution. The new held-out JSONL has not
entered a detector. No outcome informed this change; it closes a secret-hygiene
gap in the written contract.
