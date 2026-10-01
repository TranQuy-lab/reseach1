# Apparent Gains of Graph Neural Networks for NetFlow Intrusion Detection Are Explained by Model Capacity and Optimisation Stability, Not Relational Structure

**A capacity-matched, five-seed re-examination of E-GraphSAGE on NF-UQ-NIDS-v2**

> **DRAFT — NOT FOR SUBMISSION.** All numbers are read from the artefacts in
> `research_q4_2026/results/`. Items marked `[PENDING]` are still running and must be
> resolved before submission. No claim in this draft may exceed the recorded evidence.

---

## Abstract

Graph neural networks are widely reported to improve network intrusion detection by
exploiting relational structure between hosts. In the studies we re-examine, the graph model
is compared against a far smaller flow-only multilayer perceptron, so relational structure is
confounded with model capacity, depth and optimisation behaviour. We re-examine this question
on the four constituent datasets of NF-UQ-NIDS-v2 (75,987,976 NetFlow records;
NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2) using a five-seed
full-scale protocol whose 120 archived runs we audit and whose independently verified
checkpoints we reuse. We then run the controls the protocol itself required but which had
never been executed, entirely on CPU.

Our results are negative and mechanistic rather than architectural.

**The reported negative result is a training failure, not underfitting.** Recovering the
validation trajectory of every archived run shows that in 10 of 40 topology-only runs (25%;
5 of 5 seeds on NF-BoT-IoT-v2 binary) validation macro-F1 *falls* after the best checkpoint,
by up to 0.43, whereas the flow-only ablation never collapses (0 of 40; Fisher exact
p = 0.001, Holm-adjusted 0.003). On NF-BoT-IoT-v2 binary the collapsed variant also raises
its false-alarm rate to **25.02%** (250,189 false alarms per million benign flows, benign
recall 0.75) against 0.19% for the flow-only ablation and 0.32% for the variant that keeps a
direct flow-feature path.

**Capacity, not structure, explains most of the apparent advantage.** The topology-only
variant carries 16.1× the parameters of the flow-only ablation. Adding a capacity-matched
flow-only baseline under the identical budget yields a significant capacity effect in all six
completed dataset × task cells (+0.003 to +0.049 macro-F1). What survives capacity matching
is small and changes sign with the task: +0.027 [+0.016, +0.041] on NF-UNSW-NB15-v2
multiclass, +0.019 on NF-CSE-CIC-IDS2018-v2 multiclass, −0.000 on NF-ToN-IoT-v2 multiclass,
and −0.003 to −0.004 on both binary tasks. No residual structural effect approaches the
capacity effect or the tabular gap.

**A standard tabular model usually, but not always, dominates the graph variants.**
HistGradientBoosting reaches 0.866 on NF-ToN-IoT-v2 multiclass (n = 3) against 0.766 for the
best graph variant and 0.704 for the flow-only ablation, and 0.654–0.670 on
NF-UNSW-NB15-v2 multiclass against 0.414–0.490—gains of +0.10 to +0.24 while never observing
endpoint identity. On NF-CSE-CIC-IDS2018-v2 multiclass, however, the ordering reverses:
HistGradientBoosting reaches only 0.6712 and the graph variants win by 0.018–0.021 (`sage`
0.6926, `sage_edge` 0.6894 against 0.6712). Tabular dominance is a property of some datasets,
not a law, and we report both directions.

**The graph helps only through coarse endpoint summaries.** A flow-only MLP augmented with
five neighbourhood count features reaches 0.5003 on NF-UNSW-NB15-v2 multiclass, exceeding both
message-passing variants (0.4898, 0.4885). The natural stronger hypothesis fails: giving the
same model the hand-computed mean of neighbour edge features (matched capacity) yields 0.4218,
*worse* than using no graph at all, so the learned aggregation weights are doing real work.

**Endpoint familiarity is measurable and immaterial here.** Holding model, training data,
preprocessing and test population fixed, only 0.018% of NF-ToN-IoT-v2 and 2.0% of
NF-UNSW-NB15-v2 multiclass test flows have no endpoint in common with training; on those flows
the score moves by −0.044 and −0.009. Consequently we also show that an endpoint-disjoint split
is *not* a valid instrument for unseen-host claims: HistGradientBoosting, which cannot use
endpoint identity, still loses 45% of its score on such a split.

We conclude that in this setting the measurable value of relational modelling is at most a
small, task-dependent increment that is dominated both by parameter count and by what a
standard tabular learner extracts from the same flow features, and that the most visible
graph-model failure is an optimisation instability concentrated exactly where the endpoint
graph is richest. We release the learning curves, the capacity-matched baselines, the
structural-feature controls and the endpoint isolation flags.

---

## 1. Introduction

Network intrusion detection on NetFlow records is a natural graph problem: a flow is a
directed edge between two endpoints, and E-GraphSAGE (Lo et al., 2022) exploits both edge
features and topology. Subsequent work has largely reported improvements, and surveys
(e.g. Zhong et al., 2024) treat graph learning as an established direction.

The evidential weakness is structural. The usual comparison is a message-passing model
with two layers and ~128 hidden units against a single-layer MLP over edge features. Any
observed difference therefore mixes at least four causes: (i) representational capacity,
(ii) depth, (iii) the relational inductive bias, and (iv) optimisation behaviour across
seeds. In the repository we re-examine, the two models differ by a factor of 16.1 in
parameter count, and the training budget is only two passes over the training split—so
convergence and stability are live rival explanations rather than nuisances.

This paper asks a narrower and more falsifiable question than "do GNNs help":

> **R1** Does the topology-only variant outperform a *capacity-matched* flow-only baseline
> under an identical budget and identical splits?
> **R2** Is the observed negative result on NF-BoT-IoT-v2 caused by underfitting, or by
> seed-dependent optimisation collapse?
> **R3** How does a strong tabular comparator score on the same split and preprocessing?
> **R4** Do the endpoint graphs of the four datasets even carry relational information that
> a message-passing layer could exploit, and can an endpoint-disjoint evaluation be
> constructed at all?

We do not propose a new architecture. The contribution is a reproducible re-examination
that separates causes, plus the negative and mechanistic evidence that follows.

---

## 2. Related work and positioning

- **E-GraphSAGE** (Lo, Layeghy, Sarhan, Gallagher & Portmann, 2022; arXiv:2103.16329,
  NOMS 2022) is the reference method. Our implementation is a *controlled adaptation*, not
  a replication: different NF version (v2 vs. the original datasets), different split
  (70/10/20 by `flow_group_id`), five seeds, macro-F1 as the primary metric, and a
  deterministic chunked evaluation path. We state this explicitly and never compare
  headline numbers with the original paper.
- **Structural intervention for GNN-IDS** already exists (Chan et al., 2026,
  doi:10.1109/ACCESS.2026.3703442, degree-preserving rewiring on ToN-IoT and
  CIC-IDS-2017). We therefore do **not** claim to be first to rewire graphs for IDS;
  rewiring remains future work here because it requires GPU resources we did not have.
- **Survey context**: Zhong et al., 2024, doi:10.1016/j.cose.2024.103821.

Positioning: experimental cybersecurity with rigorous model-comparison methodology. The
AI contribution is the separation of capacity, structure and stability; the security
contribution is the threat- and class-level behaviour (rare classes, false positives under
class imbalance) and the endpoint-disjoint feasibility analysis.

---

## 3. Data and protocol

### 3.1 Data

Four datasets, 75,987,976 flows in total, from the NF-UQ-NIDS-v2 release:

| Dataset | Flows | Multiclass labels | Test flows (locked split) |
|---|---:|---:|---:|
| NF-UNSW-NB15-v2 | 2,390,275 | 10 | 478,007 |
| NF-BoT-IoT-v2 | 37,763,497 | 5 | 7,548,368 |
| NF-ToN-IoT-v2 | 16,940,496 | 10 | 3,385,552 |
| NF-CSE-CIC-IDS2018-v2 | 18,893,708 | 7 | 3,779,380 |

Each flow is a directed edge from `(dataset, src_ip, src_port)` to
`(dataset, dst_ip, dst_port)`. IPs and ports define nodes only and never enter the feature
matrix. The model sees the 39 NetFlow features; `source_row_id` and `flow_group_id` are
retained for traceability only.

### 3.2 Split

Whole `flow_group_id` groups are assigned by `hash(flow_group_id, 20260920) % 10`
(0–6 train, 7 validation, 8–9 test). Scalers and class maps are fit on train only.

**Reproduction check.** We re-derived the split locally from the repository's Git LFS
Parquet files using the repository's own `nids_minibatch.prepare` module. Source row
counts, per-split row counts, cross-split group counts and IP-overlap fractions match the
archived `prepare_manifest.json` exactly (to six decimal places) for all four datasets,
which establishes content equivalence despite differing file digests.

**Threat to validity (budget heterogeneity).** The locked protocol specifies two passes
with a minimum of 1,500 optimizer steps. Because NF-UNSW-NB15-v2 is small, the minimum
binds: it ran **3.667** effective passes while the other three datasets ran exactly 2.000,
and all 30 UNSW runs selected a best checkpoint at pass ≥ 3. Cross-dataset comparisons
under a "two-pass" label are therefore not budget-homogeneous, and we report UNSW
separately wherever this matters.

### 3.3 Models

| Variant | Parameters (binary / multiclass) | Description |
|---|---:|---|
| `edge_mlp` | 5,378 / 6,410 | one hidden layer (128), edge features only |
| `sage` | 86,530 / 87,301 | two mean-aggregation layers, edge features in the message, node embeddings at the head |
| `sage_edge` | 86,608 / 87,379 | as `sage`, plus the edge features directly at the head |
| `mlp_h128_1l` | 5,378 / 6,410 | our re-implementation of `edge_mlp` (harness control) |
| `mlp_h128_2l` | ~22.9k | two hidden layers (128) |
| `mlp_h273_2l` | 86,270 / 87,092 | two hidden layers (273), capacity-matched to `sage` |

The capacity-matched variant reproduces 99.7–99.8% of the topology-only parameter count
while containing **no message passing at all**.

### 3.4 Budget, checkpointing and fairness

Adam, learning rate 1e-3, dropout 0.2, batch 4096, balanced cross-entropy with weights
computed from train only; `steps_per_pass = ceil(train_rows / 4096)`;
`max_steps = max(1500, 2 × steps_per_pass)`; evaluation every
`max(500, ceil(steps_per_pass / 4))`; first checkpoint eligible only after a full pass;
checkpoints selected by validation macro-F1; the test split is scored once. These are the
settings of the archived runs, reproduced exactly for the new baselines.

Tabular comparators (HistGradientBoosting, ExtraTrees, RandomForest) reuse the scaler
stored in the archived runs' own `preprocessor.json`—they do not refit it—and use
`class_weight="balanced"`. Hyper-parameters are selected on validation with a
pre-registered grid and a single tuning seed, then refit for every seed.

### 3.5 Pre-registration

The Phase B protocol (`research_q4_2026/protocols/PROTOCOL_PHASE_B_VI.md`) was locked
before execution, and subsequent efficiency amendments are recorded in its amendment log.
The analysis of the archived learning curves is explicitly **post-hoc/exploratory**.

---

## 4. Results

### 4.1 Audit: the archive is computationally sound

All 120 runs are present with complete artefacts; published metrics match per-run
`metrics.json` to 5.7e-14; the independent verifier reports 120/120 with probability
replay error 0.0 and metric recomputation error 1.1e-16; every checkpoint has
`checkpoint_parameter_max_abs_error = 0`. Nothing in the audit contradicts the archive's
internal consistency; what changes is the interpretation.

### 4.2 The underfitting rival is rejected; the mechanism is optimisation collapse

Classifying each run's validation trajectory with a validation-only rule:

| Behaviour | Runs | Share |
|---|---:|---:|
| macro-F1 falls after the best checkpoint | 53 | 44.2% |
| still rising at the budget boundary | 44 | 36.7% |
| genuine plateau | 23 | 19.2% |

Applying a pre-registered collapse rule (`val_std ≥ 0.05` **or** `drop_after_best ≥ 0.10`):

| Variant | Collapsed / total | Rate | 95% bootstrap CI |
|---|---:|---:|---|
| `edge_mlp` | 0 / 40 | 0.0% | [0.00, 0.00] |
| `sage` | 10 / 40 | 25.0% | [0.13, 0.40] |
| `sage_edge` | 2 / 40 | 5.0% | [0.00, 0.13] |

Fisher exact `sage` vs. `edge_mlp`: *p* = 0.0010, Holm-adjusted 0.0031. The ordering is
stable across every threshold pair we tried. Collapse is concentrated on BoT-IoT (9 of 10
`sage` runs; 5 of 5 binary) where validation macro-F1 fell from 0.602 to 0.175 on one
seed. At class level the instability is visible even for the largest classes: on
CSE-CIC-IDS2018-v2 multiclass the DDoS F1 of `sage` is 0.991 / 0.641 / 0.973 / 0.662 /
0.992 across the five seeds—same split, same preprocessing, same hyper-parameters.

Validation is nevertheless an excellent checkpoint selector
(Spearman ρ between best validation and test macro-F1 = 0.998–1.000), so the checkpoint
rule is not the source of the problem.

Across all 120 runs, validation-curve instability predicts a lower test score
(Spearman ρ = −0.539, *n* = 120, *p* = 2.2e-10).

### 4.3 No average advantage for the topology-only variant

Averaging five-seed paired deltas over the eight dataset × task cells with a
DerSimonian-Laird random-effects model:

| Contrast | Pooled Δ macro-F1 | 95% CI | 95% prediction interval | I² | cells +/− |
|---|---:|---|---|---:|---:|
| `sage_edge − edge_mlp` | +0.0216 | [+0.0153, +0.0279] | [+0.0003, +0.0428] | 98.8% | 7 / 1 |
| `sage_edge − sage` | +0.0067 | [+0.0019, +0.0116] | [−0.0077, +0.0211] | 95.6% | 5 / 3 |
| **`sage − edge_mlp`** | **+0.0021** | **[−0.0105, +0.0146]** | [−0.0403, +0.0444] | 99.5% | 6 / 2 |

The topology-only variant—the closest to the reference method—shows no average advantage
over the flow-only ablation. No cell-level contrast survives Holm correction
(0 of 24 for both Wilcoxon and exact sign tests), and only 18 of 24 have uncorrected
bootstrap intervals excluding zero, so we report effect sizes and intervals rather than
significance.

### 4.4 Capacity, not structure

Since `sage` carries 16.1× the parameters of `edge_mlp`, we add the capacity-matched
flow-only baseline. Full matrix `[PENDING: CSE-CIC and BoT-IoT still running]`:

| Cell | `edge_mlp` | `mlp_h273_2l` (matched) | `sage` | `sage_edge` | HistGB | ExtraTrees | RandomForest |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNSW · multiclass | 0.414 | `[PENDING]` | 0.490 | 0.489 | 0.654 | 0.670 | 0.668 |
| UNSW · binary | 0.964 | `[PENDING]` | 0.970 | 0.970 | 0.976 | `[PENDING]` | `[PENDING]` |
| ToN · multiclass | 0.704 | **0.749** | **0.749** | 0.766 | **0.866** | `[PENDING]` | `[PENDING]` |
| CSE-CIC · multiclass | 0.669 | 0.674 | 0.693 | 0.689 | **0.671** | `[PENDING]` | `[PENDING]` |
| CSE-CIC · binary | 0.984 | **0.988** | 0.985 | 0.986 | **0.989** | `[PENDING]` | `[PENDING]` |
| ToN · binary | 0.971 | `[PENDING]` | 0.977 | 0.980 | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| CSE-CIC · mc/bin | 0.669 / 0.984 | 0.674 / 0.988 | 0.693 / 0.985 | 0.689 / 0.986 | `[PENDING]` | — | — |
| BoT-IoT · mc/bin | 0.817 / 0.900 | **0.836 / 0.935** | 0.551 / 0.804 | 0.826 / 0.882 | infeasible | — | — |

Established so far (paired by seed, 95% bootstrap CI):

| Contrast | UNSW multiclass | ToN multiclass |
|---|---:|---:|
| Contrast | UNSW mc | UNSW bin | ToN mc | CSE-CIC mc |
|---|---:|---:|---:|---:|
| capacity only, no topology (`mlp_h273_2l − edge_mlp`) | **+0.049** [+0.034, +0.060] | **+0.003** [+0.0025, +0.0034] | **+0.045** [+0.041, +0.050] | **+0.013** [+0.007, +0.020] |
| **topology at matched capacity (`sage − mlp_h273_2l`)** | **+0.027** [+0.016, +0.041] | **+0.003** [+0.002, +0.004] | **−0.000** [−0.006, +0.005] | +0.025 [−0.001, +0.051] |
| topology at matched capacity, NF-BoT-IoT-v2 mc | **−0.267** [−0.280, −0.259] | | | |
| tabular mean vs. `edge_mlp` | **+0.240** [+0.234, +0.253] | **+0.012** [+0.011, +0.012] | **+0.162** [+0.160, +0.164] | **−0.016** [−0.022, −0.007] |
| tabular mean vs. best graph variant | **+0.168** [+0.163, +0.175] | **+0.006** [+0.005, +0.006] | **+0.097** [+0.092, +0.103] | **−0.016** [−0.022, −0.007] |

- **Capacity is a first-order confound.** Adding parameters and depth with no message passing
  reproduces a significant part of the apparent graph advantage in every completed cell
  (+0.003 to +0.049 macro-F1, all 95% intervals excluding zero).
- **On NF-BoT-IoT-v2 the flow-only model is simply better.** The capacity-matched MLP reaches
  0.836 macro-F1 on multiclass (above `sage_edge` 0.826 and far above the topology-only `sage`
  0.551) and **0.949 on binary** (above `sage_edge` 0.882, `edge_mlp` 0.900 and `sage` 0.804).
  The topology-only variant collapses on 4 of 5 multiclass seeds and 5 of 5 binary seeds on
  this dataset. A model with the same parameter count that never aggregates a neighbour
  trains stably and wins on both tasks, so the collapse is a property of the message-passing
  configuration, not of the data or the budget. The topology contrast at matched capacity on
  NF-BoT-IoT-v2 multiclass is **−0.267 (95% CI [−0.280, −0.259])**.
- **What survives capacity matching depends on the dataset.** On UNSW multiclass a small
  structural increment survives (+0.027); on ToN multiclass the residual is indistinguishable
  from zero. We therefore report this as a dataset-dependent association, not a general
  effect, and we avoid a single pooled claim across the two datasets.
- **The tabular gap is large on two datasets and reversed on a third.** On UNSW multiclass
  HistGB beats `edge_mlp` by +0.240 and the best graph variant by +0.168 (ExtraTrees and
  RandomForest slightly higher still at 0.670 and 0.668); on UNSW binary the ordering is
  RandomForest 0.983 > ExtraTrees 0.980 > HistGB 0.976 > `sage_edge` 0.970 > `sage` 0.970 >
  `edge_mlp` 0.964; on ToN-IoT multiclass HistGB 0.866 beats the best graph variant by +0.101.
  **On CSE-CIC multiclass the reverse holds**: HistGB 0.6712 is below `sage` 0.6926 and
  `sage_edge` 0.6894, a paired deficit of −0.016 (95% CI [−0.022, −0.007]). On CSE-CIC binary
  HistGB is the best model at 0.9894. Tabular dominance therefore holds in five of the six
  completed cells and reverses in one; we keep both directions rather than a pooled claim.
- **Harness control**: `mlp_h128_1l − edge_mlp` = −0.0003 on UNSW binary and +0.007 on
  UNSW multiclass, i.e. within the archive's own seed noise, so the comparison harness is
  valid.

### 4.5 Where the graph carries information

| Dataset | Flows (train) | Endpoints | Edges/node | Flows per distinct pair | Repeated interactions | `sage` collapse |
|---|---:|---:|---:|---:|---:|---:|
| UNSW | 1,673,104 | 940,143 | 1.78 | 1.24 | 19.3% | 0% |
| CSE-CIC | 13,227,184 | 7,778,169 | 1.70 | 1.28 | 21.9% | 10% |
| ToN | 11,861,039 | 1,046,743 | 11.33 | 1.58 | 36.5% | 0% |
| BoT-IoT | 26,438,303 | 336,138 | **78.65** | **11.96** | **91.6%** | **90%** |

Relational richness and optimisation fragility coincide: the dataset with by far the most
redundant endpoint interaction is also the one where the topology-only architecture
collapses on every binary seed. Conversely, on the two datasets whose endpoint graphs are
almost equivalent to independent flows, message passing has no information to add—which is
exactly where the tabular comparator wins by the largest margin.

### 4.6 Endpoint-disjoint evaluation is only partly feasible

A connected-component holdout fails: the endpoint graph has one giant component
(NF-UNSW-NB15-v2: 22,750 components over 1.09 million endpoints), leaving 0.24% of flows
in test and losing three classes. A held-out *endpoint group* design works instead—a flow is
kept only when both endpoints share a group, and holdout flows are split 1/3 validation,
2/3 test:

| Dataset | Test flows (locked → holdout) | Retained | Classes | Label TV distance | Holdout endpoints in train |
|---|---:|---:|---|---:|---:|
| UNSW | 478,007 → 133,537 | 59.1% | 10 → 10 | 0.018 | 0 |
| ToN | 3,385,552 → 587,720 | 63.2% | 10 → 10 | 0.293 | 0 |
| CSE-CIC | 3,779,380 → 532,086 | 64.3% | 7 → 4 | 0.102 | 0 |
| BoT-IoT | 7,548,368 → 121,205 | 69.4% | 5 → 3 | **0.927** | 0 |

The holdout passes a feasibility gate only on UNSW and ToN. Any "unseen host" claim must
be confined to those datasets and must report the label shift; for BoT-IoT the retained
test set changes the class distribution almost completely (TV = 0.93).

### 4.6b Endpoint familiarity: measured, and found immaterial on this split

The repository documents that 99.21–100% of validation/test flows have both IP addresses
already present in train, and treats endpoint overlap as a limitation on interpretation.
We tested it directly in two ways, and the second test corrected the first.

**Test 1 (endpoint-disjoint split).** On the held-out-endpoint design, the
capacity-matched flow-only MLP falls from 0.750 to 0.440 macro-F1 on NF-ToN-IoT-v2
multiclass—a 41% loss—while the binary task loses only 0.001. Read alone this looks like
strong evidence that endpoint familiarity supplies attack-type discrimination.

**Test 2 (the necessary control).** We then trained HistGradientBoosting on the same
endpoint-disjoint split. It has no access to endpoint identity of any kind, yet it also
falls, from 0.866 on the locked split to **0.472** here. A model that cannot use endpoint
identity cannot be losing 45% of its score to endpoint unfamiliarity. The drop must come
from the split construction itself: the endpoint-disjoint design discards 37% of flows,
changes the training population, and shifts the test label distribution
(total-variation distance 0.293). **Test 1 is therefore confounded and we do not use it to
attribute anything to endpoint familiarity.**

**Test 3 (clean isolation).** Finally we held the model, the training data, the
preprocessing and the test population completely fixed, and merely partitioned the locked
test split by whether each flow's endpoints occur in train:

| Test subset | NF-ToN-IoT-v2 mc (n=2) | share | NF-UNSW-NB15-v2 mc (n=1) | share |
|---|---:|---:|---:|---:|
| all test flows | 3,385,552 | 100% | 478,007 | 100% |
| both endpoints seen in train | 3,319,739 | 98.1% | 362,724 | 75.9% |
| exactly one endpoint seen | 65,192 | 1.9% | 105,750 | 22.1% |
| **neither endpoint seen** | **621** | **0.018%** | **9,533** | **2.0%** |

HGB macro-F1 by subset:

| Test subset | ToN-IoT mc | Δ | UNSW mc | Δ |
|---|---:|---:|---:|---:|
| all test flows | 0.8659 | — | 0.6522 | — |
| both endpoints seen | 0.8659 | −0.000 | 0.6554 | +0.003 |
| exactly one endpoint seen | 0.8670 | +0.001 | 0.6410 | −0.011 |
| **neither endpoint seen** | **0.8220** | **−0.044** | **0.6435** | **−0.009** |

Three conclusions. First, the documented overlap figure is confirmed at endpoint-pair
granularity: only 0.018% of NF-ToN-IoT-v2 test flows have no endpoint in common with the
training data. NF-UNSW-NB15-v2 is the harder case with 2.0% such flows—its graph is sparse
enough that many endpoints appear only in test—and there the score moves by −0.009.
Second, neither dataset shows an effect large enough to explain any model ranking: the
largest subset difference anywhere is −0.044, on 621 flows. Third, the two datasets differ
by two orders of magnitude in exposure yet agree on the conclusion, which is what one
would expect if endpoint familiarity genuinely plays no role here.

**What this changes.** The endpoint-overlap limitation is real as a statement about
population coverage but empirically immaterial for the comparisons reported here: the
within-environment scores are not inflated by host memorisation. The corollary is that an
honest "unseen host" experiment cannot be built by discarding mixed-endpoint flows; it
requires a temporal or cross-network holdout. `[PENDING: the same three tests for the
remaining datasets.]`

### 4.6c Cheap structural statistics beat message passing

If the contribution of message passing is the aggregation of neighbourhood
information, then supplying those aggregates *directly* to a flow-only model should
reproduce it. We built five structural features from the **train split only**—source
flow count, destination flow count, distinct source partners, distinct destination
partners, and the repetition count of the exact endpoint pair—and added them to the
capacity-matched MLP. A second variant adds the training-label attack rate of each
endpoint (a transductive statistic).

UNSW multiclass, test macro-F1:

| Model | Parameters | Test macro-F1 |
|---|---:|---:|
| `mlp_h273_2l` (flow only, capacity-matched) | 88,462 | 0.4628 |
| `sage` (message passing) | 87,301 | 0.4898 |
| `sage_edge` (message passing + direct edge) | 87,379 | 0.4885 |
| **`mlp_struct` (flow + 5 structural counts)** | **89,827** | **0.5045 / 0.4970** (seeds 11, 22) |
| `mlp_struct_lab` (flow + counts + train-label rates) | `[PENDING]` | `[PENDING]` |

A flow-only MLP with five cheap neighbourhood summaries **exceeds both message-passing
variants** at essentially the same parameter count. The measurable benefit attributed to
"topology" on this dataset is therefore reproducible by counting both endpoints'
neighbourhoods.

**But the effect is not the aggregation content.** We tested the natural stronger
hypothesis—that two mean-aggregation layers with constant node initialisation reduce to a
hand-computable neighbour average—by giving the model, for each flow, the mean of the 39
edge features over the training flows arriving at each endpoint (78 extra features,
87,356 parameters, matched to `sage`). This **fails**: 0.4218 versus 0.4628 for the
flow-only capacity-matched baseline and 0.4898 for `sage`. Raw neighbour averaging is
worse than not using the graph at all, because the learned aggregation weights that
`sage` supplies are doing real work.

The defensible statement is therefore narrower: what helps is *any* informative summary of
both endpoints' neighbourhoods—even a handful of counts—and not the neighbour-feature
content itself. Two further negative controls agree: adding train-label attack rates per
endpoint (`mlp_struct_lab`, 0.4708 at n = 3) is *worse* than the counts alone (0.5003),
consistent with overfitting endpoint identity rather than learning transferable
structure.

### 4.6d The collapsed variant is also operationally unusable

From the archived full-test confusion matrices at each model's own operating point
(the repository does not store GNN probabilities, so PR-AUC is not recoverable):

Binary detector false-alarm rate (benign flows flagged as attack):

| Dataset | `edge_mlp` | `sage` | `sage_edge` |
|---|---:|---:|---:|
| NF-BoT-IoT-v2 | 0.19% | **25.02%** | 0.32% |
| NF-CSE-CIC-IDS2018-v2 | 0.19% | 0.20% | 0.12% |
| NF-ToN-IoT-v2 | 3.26% | 1.55% | 1.47% |
| NF-UNSW-NB15-v2 | 0.60% | 0.50% | 0.50% |

The topology-only variant on NF-BoT-IoT-v2 raises **250,189 false alarms per million
benign flows** while its benign recall falls to 0.75. This is not a marginal ranking
difference; it is a detector that cannot be deployed. Its worst per-class
false-alarm rates are 37.3% (`DoS`) and 23.4% (`DDoS`). The same variant is
well-behaved on the other three datasets, confirming that the failure is specific to
the configuration that collapses.

### 4.6e Flow-based detectors do not transfer across networks

The repository's stated limitations include cross-network generalisation, which had never
been measured. It is measurable without a GPU for flow-only models, because they do not use
endpoint identity: only the 39 features must be commensurate. We train on one dataset's
train split with that dataset's own scaler, select the checkpoint on that dataset's
validation split, and score every other dataset's test split after expressing its flows in
the source's standardisation. The reference is the macro-F1 of a constant predictor that
always emits the majority class, which for binary prevalence *p* is *p*/(1+*p*).

| Source → target | macro-F1 | majority-class macro-F1 | runs beating the baseline |
|---|---:|---:|---:|
| UNSW → UNSW (in-domain) | 0.9674 | 0.4900 | 3 / 3 |
| UNSW → ToN-IoT | 0.3810 | 0.3902 | 1 / 3 |
| UNSW → CSE-CIC-IDS2018 | 0.4286 | 0.4681 | 0 / 3 |
| UNSW → BoT-IoT | 0.3035 | 0.4991 | 1 / 3 |
| ToN-IoT → ToN-IoT (in-domain) | 0.9813 | 0.3902 | 2 / 2 |
| ToN-IoT → CSE-CIC-IDS2018 | 0.5089 | 0.4681 | 2 / 2 |
| ToN-IoT → UNSW | 0.1993 | 0.4900 | 0 / 2 |
| ToN-IoT → BoT-IoT | 0.0352 | 0.4991 | 0 / 2 |

**In-domain, 5 of 5 runs beat a constant classifier. Across networks, only 4 of 15 do.** Two
cells are catastrophic: transferring from ToN-IoT to BoT-IoT yields 0.035 macro-F1, and from
ToN-IoT to UNSW 0.199, against constant-predictor baselines of 0.499 and 0.490. Within the
network it was trained on the detector is excellent; outside it, it is routinely worse than
predicting one label for everything.

Two consequences. First, whatever these flow models learn is network-specific, so the
within-environment numbers reported throughout this paper—including every graph variant's—
must not be read as evidence of deployable detection capability. Second, this is the correct
way to test unseen-environment generalisation: not by holding out endpoints within one
network, but by holding out the network. `[PENDING: additional source datasets.]`

### 4.7 Rare classes go in both directions

With five seeds (support < 1,000), topology helps NF-ToN-IoT-v2 `ransomware`
(0.073 → 0.631) and NF-UNSW-NB15-v2 `Shellcode` (0.230 → 0.415), but *hurts*
NF-UNSW-NB15-v2 `Backdoor` (0.191 → 0.128) on the same dataset. The claim must therefore
be class-specific, not "topology helps rare classes".

### 4.8 Three seeds are not enough

Adding seeds 44 and 55 increased the across-seed SD by up to **6.67×**
(BoT-IoT binary `edge_mlp`: 0.0038 → 0.0256), with 9 of 24 cells above 1.5×; 4 of 24
bootstrap-CI decisions changed and one mean changed sign. Three-seed summaries in earlier
internal documents understated variability and in some cells inverted the conclusion.

---

## 5. Discussion

The results support a mechanistic account with two parts.

**Capacity.** A flow-only MLP already sees the complete measurement for each flow. The
archived comparison gave the message-passing model 16.1× more parameters and one extra
hidden layer, and roughly half of the measured advantage on both completed cells is
reproduced by a matched-capacity model that never aggregates a neighbour (UNSW +0.049,
ToN +0.046). Any claim about the value of relational structure must therefore be made
against a capacity-matched control.

**Structure and stability.** A message-passing layer can only add information if the
endpoint graph contains redundant interaction—repeated flows between the same hosts, or
hosts with many neighbours. For NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2 that condition
is weak (1.7–1.8 edges per endpoint, ~20% repeated pairs). NF-ToN-IoT-v2 is intermediate
(11.3 edges per endpoint, 36.5% repeated) and the residual structural increment is zero.
NF-UNSW-NB15-v2 is the cell where a structural increment survives capacity matching
(+0.027), yet there the graph is nearly information-free—so the increment cannot plausibly
come from rich relational context and is more likely a modest architectural regularisation
effect that we cannot separate with the present design. The endpoint-holdout experiment
supplies the missing corroboration: on NF-UNSW-NB15-v2 the graph carries no exploitable
host identity (no degradation when endpoints are held out), whereas on NF-ToN-IoT-v2 it
carries a great deal (42% of the score lost). Finally, for NF-BoT-IoT-v2 the
graph is by far the richest (78.7 edges per endpoint, 91.6% repeated pairs)—and precisely
there the topology-only architecture becomes untrainable across seeds (5 of 5 binary),
while the variant that also keeps a direct flow-feature path to the head remains stable.
Relational richness and optimisation fragility coincide in this architecture.

**Endpoint familiarity is a multiclass artefact.** The 0.31 macro-F1 lost on
NF-ToN-IoT-v2 multiclass and the 0.001 lost on its binary task settle where the
within-environment advantage comes from: not from a better benign/attack boundary but from
attack-type discrimination that endpoint identity supplies. Reporting only multiclass
scores on datasets with recurrent endpoint interaction therefore overstates what a
deployable detector would achieve.

**Practical implication.** Reporting a graph model against a small flow-only MLP does not
establish the value of relational modelling: it conflates capacity with structure. A
capacity-matched baseline, a strong tabular comparator, and seed-level stability
diagnostics should be standard. In our setting the residual structural effect is small and
inconsistent in sign across datasets, while classical tabular models dominate every graph
variant on the two datasets where the comparison is complete. Deploying the graph variants
examined here is therefore not justified by their measured performance—and the variant
that looks most attractive on F1 alone is, on NF-BoT-IoT-v2, a 25% false-alarm detector.

---

## 6. Threats to validity

1. Budget heterogeneity (UNSW 3.67 passes vs. 2.00 elsewhere).
2. Only five seeds; no cell-level contrast survives multiplicity correction.
3. Endpoint overlap in the primary split. Measured, not assumed: 98.1% of NF-ToN-IoT-v2
   multiclass test flows have both endpoints in train and only 0.018% have neither, on
   which the score is 0.039 lower. The overlap therefore does **not** inflate the reported
   comparisons, but it also means this split cannot support any unseen-host claim. An
   endpoint-disjoint split is not a valid substitute: it shifts the distribution so
   strongly that a model with no endpoint access (HistGradientBoosting) loses 45% of its
   score on it.
4. `flow_group_id` split is within-environment, not temporal.
5. Tabular comparators cannot model relations by construction; they are a floor, not a ceiling.
6. Some comparator cells are unrun due to CPU RAM limits.
7. Scalers differ by ≤ 5.7e-11 relative between the seed 11/22/33 and 44/55 batches
   (floating-point accumulation, not a preprocessing change).
8. No structural intervention (rewiring) was executed, so no causal claim is made.

---

## 7. Conclusion and future work

On full-scale NF-UQ-NIDS-v2 with verified five-seed artefacts, and with all eight
dataset × task cells now carrying a pre-registered capacity-matched control, the apparent
advantage of a topology-aware model over a flow-only ablation is substantially a capacity
effect. Adding parameters and depth with no message passing yields a positive, significant
gain in **every one of the eight cells** (+0.003 to +0.049 macro-F1, all 95% intervals
excluding zero). Across the same cells the topology-only variant shows no average advantage
(pooled Δ = +0.002, 95% CI [−0.011, +0.015]).

What remains after capacity matching does not support the graph model. The topology-only
variant is **significantly worse in four cells**—by −0.267 and −0.132 on NF-BoT-IoT-v2
multiclass and binary, and by −0.003 to −0.004 on the two binary tasks where the graph is
sparse—indistinguishable in two (NF-ToN-IoT-v2 multiclass, NF-CSE-CIC-IDS2018-v2
multiclass), and only marginally better in the two NF-UNSW-NB15-v2 cells (+0.027 and
+0.003). No cell shows a strong positive structural effect.

The most distinctive behaviour is instability. The topology-only architecture collapses on
25% of the archived runs, concentrated exactly where the endpoint graph is richest
(NF-BoT-IoT-v2: 4 of 5 multiclass seeds and 5 of 5 binary seeds), and where it does it is
not merely worse but undeployable—a 25.02% false-alarm rate and 250,189 false alarms per
million benign flows. On the same dataset, split and budget, a flow-only model of identical
parameter count trains stably and is the best model on **both** tasks (0.836 multiclass,
0.935 binary).

A standard tabular learner that never observes endpoint identity exceeds every graph
variant in five of the six cells where it was run (+0.004 to +0.168 macro-F1) and falls
behind by 0.016 in the sixth (NF-CSE-CIC-IDS2018-v2 multiclass). Where it was measurable,
a flow-only model with five neighbourhood counts outperforms message passing on two
datasets, while hand-computed neighbour feature averages are worse than using no graph at
all. Finally, endpoint familiarity—the limitation the source repository flags but never
measured—turns out to be immaterial on these splits: only 0.018% and 2.0% of test flows
have no endpoint in train, and on those the score moves by −0.044 and −0.009.

Taken together, the honest conclusion is narrower and more useful than "GNNs do not help".
In this setting the measurable contribution of relational modelling is at most a small,
task-dependent increment that is dominated by parameter count and by what a standard
tabular learner extracts from the same flow features; the most visible graph-model failure
is an optimisation instability concentrated exactly where the endpoint graph is richest;
and none of these within-network numbers should be read as deployable detection capability,
since a flow-only detector trained on one network is at or below a constant classifier on
the others (4 of 15 cross-network runs beat that baseline).

Future work, in priority order and all requiring GPU resources: (i) a convergence-first
re-evaluation with eight passes and validation-only early stopping; (ii) degree-preserving
graph rewiring as a structural negative control; (iii) training the graph variants on the
released endpoint-disjoint splits for UNSW and ToN; (iv) learning-rate and budget
sensitivity for the collapsing configuration.

---

## Reproducibility statement

All scripts are in `research_q4_2026/scripts/`; all reported numbers are read from
`research_q4_2026/results/`; the report `02_BAO_CAO_KET_QUA_VI.md` is regenerated from
those files. The source repository is used read-only: no file under `src/`, `tests/` or
`research/` was modified.
