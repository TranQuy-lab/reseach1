# RF bounded comparator — verified results

Protocol: four datasets, multiclass, seeds 11/22, 25 trees, max_depth 16, full-data splits.

| Dataset | Mean macro-F1 | SD | Min | Max |
|---|---:|---:|---:|---:|
| NF-UNSW-NB15-v2 | 0.649504 | 0.001362 | 0.648541 | 0.650467 |
| NF-BoT-IoT-v2 | 0.941010 | 0.000807 | 0.940440 | 0.941581 |
| NF-ToN-IoT-v2 | 0.871067 | 0.000350 | 0.870819 | 0.871314 |
| NF-CSE-CIC-IDS2018-v2 | 0.669713 | 0.000885 | 0.669087 | 0.670339 |

## Verification

- Passed: **true**
- Runs checked: **8**
- Largest metric recomputation error: 1.1102230246251565e-16
- Largest probability replay error: 6.661338147750939e-16

The bounded RF arm is descriptive; it does not support claims about all tabular baselines or full five-seed RF stability.

## Provenance note

The original full-run checkpoint directories were not present on this fresh server. For this bounded comparator, each preprocessor was regenerated with the repository Preprocessor code using only that dataset/task train split; validation and test were not used to fit preprocessing. This is a bounded same-split comparator, not a replay of the original GNN preprocessor artifact.
