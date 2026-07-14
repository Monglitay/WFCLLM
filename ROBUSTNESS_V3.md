# Watermark Mechanism V3 Robustness

## Status

Measured robustness is **not evaluated**. The preregistration allowed attacks only on pilot positives after all clean-pilot gates passed. Base R3 exactness failed, so running attacks would spend evidence on an invalid mechanism and risk post-hoc interpretation.

## Structural expectations (not measurements)

| Attack | Expected evidence effect | Bound/limitation |
|---|---|---|
| Format normalization | Usually canonicalized away by `ast.unparse` | only for parse-equivalent Python accepted by the same parser |
| Identifier rename | Changes canonical current unit and often its unit ID/context | expected to re-key or erase affected units; no invariance claim |
| Dead-code insertion | Adds units and changes previous-unit context for downstream statements | can dilute evidence and perturb a suffix of context-linked units |
| Statement deletion | Removes one unit and changes the next surviving unit's previous context | at least the deleted unit is lost; adjacent evidence may also change |
| Statement reorder | Changes previous/current context pairs and possibly structural roles | reordered units can be re-keyed broadly; no order-invariance claim |

The independent-unit upper bound after an attack is at most the number of unique, eligible unit IDs whose canonical current form, public role, and required previous context survive unchanged. Since the detector requires at least three independent units, any attack leaving fewer than three recoverable units forces an ineligible decision. These are theoretical consequences of the frozen contract, not empirical TPR results.
