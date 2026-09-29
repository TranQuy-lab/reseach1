# RF bounded comparator — verified five-seed results

Protocol: four datasets, multiclass, seeds 11/22/33/44/55, 25 trees, max_depth 16, full-data splits.

| Dataset | Mean macro-F1 | SD | Min | Max |
|---|---:|---:|---:|---:|
| NF-UNSW-NB15-v2 | 0.648972 | 0.001096 | 0.647647 | 0.650467 |
| NF-BoT-IoT-v2 | 0.941160 | 0.000495 | 0.940440 | 0.941650 |
| NF-ToN-IoT-v2 | 0.872002 | 0.001215 | 0.870819 | 0.873706 |
| NF-CSE-CIC-IDS2018-v2 | 0.668459 | 0.002469 | 0.664181 | 0.670339 |

## Verification

- Passed: true
- Runs checked: 20
- Largest metric recomputation error: 1.1102230246251565e-16
- Largest probability replay error: 6.661338147750939e-16

The bounded RF arm is descriptive; it does not support claims about all tabular baselines. It uses five seeds but remains bounded at 25 trees and max_depth 16.

## Provenance note

The original full-run checkpoint directories were not present on this fresh server. Each preprocessor was regenerated with the repository Preprocessor code using only that dataset/task train split; validation and test were not used to fit preprocessing. This is a bounded same-split comparator, not a replay of the original GNN preprocessor artifact.
