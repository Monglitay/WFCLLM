from __future__ import annotations

FORBIDDEN_SECRET_FIELDS = {"secret_key", "raw_secret_key"}

POSTHOC_PASS_REQUIRED_FLAGS = {
    "posthoc_only",
    "not_used_for_generation",
    "not_used_for_retry",
    "not_used_for_selection",
    "not_used_for_calibration",
    "not_used_for_detection",
}
