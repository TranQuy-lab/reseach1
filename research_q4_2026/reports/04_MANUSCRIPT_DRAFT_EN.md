# Apparent Gains of Graph Neural Networks for NetFlow Intrusion Detection Are Explained by Model Capacity and Optimisation Stability, Not Relational Structure

**A capacity-matched, five-seed re-examination of E-GraphSAGE on NF-UQ-NIDS-v2**

> **DRAFT — NOT FOR SUBMISSION.** All numbers are read from the artefacts in
> `research_q4_2026/results/`. Items marked `[PENDING]` are still running and must be
> resolved before submission. No claim in this draft may exceed the recorded evidence.

---

## Abstract

Graph neural networks are widely reported to improve network intrusion detection by
exploiting relational structure between hosts. Most such reports compare a graph model
against a much smaller flow-only multilayer perceptron, so relational structure is
confounded with model capacity, depth and optimisation behaviour. We re-examine this
question on the four constituent datasets of NF-UQ-NIDS-v2 (75,987,976 NetFlow records;
NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2) using a
reproducible five-seed full-scale protocol whose 120 archived runs we re-audit and whose
independently verified checkpoints we reuse.

We make four contributions. First, we recover the validation learning curves of all 120
runs and show that the central negative result—the collapse of the topology-only variant
on NF-BoT-IoT-v2—is **not underfitting**: in 10 of 40 topology-only runs (25%; 5 of 5 seeds
on BoT-IoT binary) validation macro-F1 *falls* after the best checkpoint by up to 0.43,
whereas the flow-only ablation never collapses (0 of 40; Fisher exact *p* = 0.001,
Holm-adjusted 0.003). Second, we show that the topology-only variant carries **16.1× the
parameters** of the flow-only ablation, and we close that confound with a capacity-matched
flow-only baseline trained under the identical budget. Capacity alone accounts for roughly
half of the apparent graph advantage on both datasets we have completed (+0.049 on
NF-UNSW-NB15-v2, +0.046 on NF-ToN-IoT-v2). What survives capacity matching is
**dataset-dependent**: on NF-UNSW-NB15-v2 multiclass the topology-only variant retains a
small positive increment over the matched baseline (+0.027, 95% CI [+0.016, +0.041]),
whereas on NF-ToN-IoT-v2 multiclass the residual is zero (−0.002, 95% CI [−0.011, +0.006]).
Third, on NF-UNSW-NB15-v2 a strong tabular comparator reaches 0.654–0.670 macro-F1
against 0.414–0.490 for all graph variants, i.e. any residual structural gain is an order
of magnitude smaller than the gap to classical tabular learning. Fourth,
we show that the endpoint graph itself explains where relational modelling could help:
NF-BoT-IoT-v2 has 78.7 edges per endpoint and 91.6% repeated endpoint interactions,
whereas NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2 have only 1.7–1.8 edges per endpoint
and ~20% repeated interactions.

We report these as controlled negative and mechanistic evidence, not as a new
architecture, and we release the endpoint-disjoint splits, the learning curves and the
capacity-matched baselines.

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
| ToN · multiclass | 0.704 | **0.750** | **0.749** | 0.766 | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| ToN · binary | 0.971 | `[PENDING]` | 0.977 | 0.980 | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| CSE-CIC · mc/bin | 0.669 / 0.984 | `[PENDING]` | 0.693 / 0.985 | 0.689 / 0.986 | `[PENDING]` | — | — |
| BoT-IoT · mc/bin | 0.817 / 0.900 | `[PENDING]` | 0.551 / 0.804 | 0.826 / 0.882 | `[PENDING]` | — | — |

Established so far (paired by seed, 95% bootstrap CI):

| Contrast | UNSW multiclass | ToN multiclass |
|---|---:|---:|
| Contrast | UNSW mc | UNSW bin | ToN mc | CSE-CIC mc |
|---|---:|---:|---:|---:|
| capacity only, no topology (`mlp_h273_2l − edge_mlp`) | **+0.049** [+0.034, +0.060] | **+0.003** [+0.0025, +0.0034] | **+0.045** [+0.041, +0.050] | **+0.013** [+0.007, +0.020] |
| **topology at matched capacity (`sage − mlp_h273_2l`)** | **+0.027** [+0.016, +0.041] | **+0.003** [+0.002, +0.004] | **−0.000** [−0.006, +0.005] | +0.025 [−0.001, +0.051] |
| tabular vs. `edge_mlp` | **+0.240** [+0.234, +0.253] | **+0.012** [+0.011, +0.012] | `[PENDING]` | `[PENDING]` |
| tabular vs. best graph variant | **+0.168** [+0.163, +0.175] | **+0.006** [+0.005, +0.006] | `[PENDING]` | `[PENDING]` |

- **Capacity is a first-order confound.** Roughly half of the apparent graph advantage on
  both completed cells is reproduced by adding parameters and depth with no message passing.
- **What survives capacity matching depends on the dataset.** On UNSW multiclass a small
  structural increment survives (+0.027); on ToN multiclass the residual is indistinguishable
  from zero. We therefore report this as a dataset-dependent association, not a general
  effect, and we avoid a single pooled claim across the two datasets.
- **The residual structural gain is small compared with the tabular gap.** On UNSW
  multiclass HistGB beats `edge_mlp` by +0.240 and the best graph variant by +0.168, while
  ExtraTrees and RandomForest are slightly higher still (0.670, 0.668). On UNSW binary the
  ordering is RandomForest 0.983 > ExtraTrees 0.980 > HistGB 0.976 > `sage_edge` 0.970 >
  `sage` 0.970 > `edge_mlp` 0.964.
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

### 4.6b Endpoint familiarity matters only where the graph is rich

We trained the *same* flow-only architectures on both splits, with identical budgets and
seeds, fitting the scaler on each split's own train. On NF-UNSW-NB15-v2 multiclass the
endpoint-disjoint test set costs **+0.008 to +0.017** macro-F1 (95% CI includes zero), and
on NF-UNSW-NB15-v2 binary it *improves* by 0.016. On NF-ToN-IoT-v2 multiclass, by
contrast, it collapses:

| Dataset · task · model | Locked split | Endpoint holdout | Change |
|---|---:|---:|---:|
| UNSW · binary · `mlp_h128_1l` | 0.9641 | 0.9802 | **+0.016** |
| UNSW · multiclass · `mlp_h128_1l` | 0.4218 | 0.4053 | −0.017 (CI [−0.038, +0.013]) |
| UNSW · multiclass · `mlp_h273_2l` | 0.4697 | 0.4616 | −0.008 (CI [−0.029, +0.023]) |
| ToN · multiclass · `mlp_h273_2l` | 0.7508 | 0.4385 | **−0.312** (41.6%) |
| **ToN · multiclass · `mlp_h128_1l`** | **0.6963** | **0.4088** | **−0.287** (41.3%, CI [0.273, 0.298]) |

This is the clearest evidence for the mechanism. On the dataset whose endpoint graph is
nearly information-free, unseen endpoints cost nothing: there is no host identity to rely
on. On NF-ToN-IoT-v2, where endpoints recur 11.3 times on average and 36.5% of flows
repeat an endpoint pair, a flow-only model loses **0.29–0.31 macro-F1**—about 41% of its
score—when those endpoints are held out, and the drop is far outside the seed noise
(paired 95% CI [0.273, 0.298]). Published within-environment scores on such datasets
therefore substantially measure endpoint familiarity rather than transferable flow
structure. `[PENDING: ToN binary and `mlp_h273_2l` cells; CSE-CIC and BoT-IoT fail the
feasibility gate.]`

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

**Practical implication.** Reporting a graph model against a small flow-only MLP does not
establish the value of relational modelling: it conflates capacity with structure. A
capacity-matched baseline, a strong tabular comparator, and seed-level stability
diagnostics should be standard. In our setting the residual structural effect is small and
inconsistent in sign across datasets, while classical tabular models dominate every graph
variant on the two datasets where the comparison is complete. Deploying the graph variants
examined here is therefore not justified by their measured performance.

---

## 6. Threats to validity

1. Budget heterogeneity (UNSW 3.67 passes vs. 2.00 elsewhere).
2. Only five seeds; no cell-level contrast survives multiplicity correction.
3. Endpoint overlap in the primary split (92.7–100% of holdout IPs already appear in
   train). We now quantify its cost rather than merely flagging it: negligible on
   NF-UNSW-NB15-v2, severe on NF-ToN-IoT-v2 (−0.29 macro-F1), and not measurable on
   NF-CSE-CIC-IDS2018-v2 or NF-BoT-IoT-v2 because the endpoint-disjoint split loses
   classes there.
4. `flow_group_id` split is within-environment, not temporal.
5. Tabular comparators cannot model relations by construction; they are a floor, not a ceiling.
6. Some comparator cells are unrun due to CPU RAM limits.
7. Scalers differ by ≤ 5.7e-11 relative between the seed 11/22/33 and 44/55 batches
   (floating-point accumulation, not a preprocessing change).
8. No structural intervention (rewiring) was executed, so no causal claim is made.

---

## 7. Conclusion and future work

On full-scale NF-UQ-NIDS-v2 with verified five-seed artefacts, the apparent advantage of a
topology-aware model over a flow-only ablation is substantially a capacity effect: about
half of it is reproduced by a capacity-matched model with no message passing, and across
the eight dataset × task cells the topology-only variant shows no average advantage
(pooled Δ = +0.002, 95% CI [−0.011, +0.015]). What survives capacity matching is
dataset-dependent: a small positive increment on NF-UNSW-NB15-v2 multiclass (+0.027) and
none on NF-ToN-IoT-v2 multiclass (−0.002). The most distinctive behaviour is instability:
the topology-only architecture collapses on 25% of runs, concentrated exactly where the
endpoint graph is richest. On the two datasets where the comparison is complete, strong
tabular models exceed every graph variant, by +0.17 to +0.24 macro-F1 on UNSW multiclass
`[PENDING: remaining cells]`.

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
