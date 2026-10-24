# Query-Conditioned Energy FG: the proposed novelty

## Research decision

The strongest defensible extension is not simply “add an RBM to FG.” RBMs
already appear in social-circle/community detection, and modern community
detection also uses learned graph representations. The novelty implemented here
is a **query-conditioned energy-based candidate selector**:

> Given the current local community `C` and a folded candidate subgraph `F`,
> learn a compatibility score for the pair `(C,F)` and use it to rerank the
> candidates during modularity expansion.

The original FG algorithm remains available and unchanged as the baseline.
The new method is implemented by `HybridFGDetector`.

This is a research hypothesis, not a proven improvement. The experiment files
store the measured comparison and make regressions visible.

## Clear example from the base paper

Assume the seed is node `v` and the current community is:

```text
C = {v}
```

Two neighboring candidates produce folded subgraphs:

```text
F1 = {a, b, c}
F2 = {d, e}
```

The paper's modularity stage computes `DeltaM(F1)` and `DeltaM(F2)` and adds the
candidate with the larger positive gain. In simplified form:

```text
candidate = argmax_F DeltaM(F)
```

This is the exact behavior preserved by `ModularityExpansion` and
`FGDetector`.

## What the novelty does

The new method calculates local features for every `(C,F)` pair:

| Feature | Meaning |
|---|---|
| `folded_density` | Internal density of the folded candidate. |
| `community_overlap` | Fraction of current-community nodes overlapping `F`. |
| `first_order_overlap` | `|N(F) intersection C| / |C|`. |
| `second_order_overlap` | `|NN(F) intersection C| / |C|`. |
| `modularity_gain` | The paper's exact `DeltaM(F)`. |
| `external_edge_ratio` | Candidate edges leading outside `C union F`, divided by candidate degree. |
| `candidate_size` | Number of nodes in `F`. |
| `mean_degree` | Mean graph degree of nodes in `F`. |
| `internal_candidate_ratio` | Candidate-internal edge incidences divided by candidate degree. |

An interpretable logistic compatibility model is trained on LFR1-LFR5 only.
Its output is a calibrated candidate compatibility estimate `p(F|C)`. The
hybrid selector uses:

```text
S(F) = (1 - w) * normalized(DeltaM(F))
       + w * p(F|C)
       - lambda * external_edge_ratio(F)
```

The implementation defaults are:

```text
gamma              = 0.45
h                  = 1.0
w                  = 0.50
lambda             = 0.10
minimum_membership = 0.60
modularity_slack   = 0.05
```

The hybrid may accept a non-positive modularity candidate only when its learned
compatibility exceeds the minimum threshold. This is the controlled mechanism
intended to address the original paper's precision/recall limitation.

## What is unchanged

- Folded-subgraph construction and density rule.
- Folding-stage majority-overlap rule.
- The paper's `M = ein/eout` definition.
- The paper's `x`, `y`, `z`, and `DeltaM` calculation.
- Relaxation score `f1 + 0.5*f2` and threshold `h`.
- Seed-based Recall, Precision, and F-score evaluation.

Only candidate ranking in the modularity stage changes.

## Leakage control

The scorer is trained using LFR1-LFR5 and evaluated on held-out LFR6-LFR9.
Ground-truth labels are used only to create training labels. Test labels are
used only after detection to calculate evaluation metrics. The feature extractor
does not read ground truth at prediction time.

The current training label is:

```text
positive = at least half of F belongs to the training seed's true community
```

This makes the current method supervised/semi-supervised, not unsupervised.
That distinction must be retained in any paper claim.

## Run the novelty experiment

Compile:

```cmd
javac -d out *.java
```

Run a fast held-out smoke experiment:

```cmd
java -cp out foldedsubgraph.NoveltyExperiment ^
  datasets/generated-lfr output/novelty-results 1
```

Run with all nodes in LFR6-LFR9:

```cmd
java -cp out foldedsubgraph.NoveltyExperiment ^
  datasets/generated-lfr output/novelty-results 0
```

Optional parameters are:

```text
NoveltyExperiment data output seedLimit membershipWeight externalPenalty minimumMembership modularitySlack
```

The command stores:

```text
output/novelty-results/
  candidate-scorer.txt
  training-metadata.txt
  novelty-config.txt
  per-seed.csv
  summary.csv
  figures/base-vs-novelty-metrics.png
  figures/base-vs-novelty-f1.png
```

## Observed held-out smoke result

The executed one-seed-per-dataset run produced:

| Dataset | Method | Recall | Precision | F-score |
|---|---:|---:|---:|---:|
| LFR6 | FG | 1.0000 | 0.9512 | 0.9750 |
| LFR6 | Novelty | 1.0000 | 0.9512 | 0.9750 |
| LFR7 | FG | 1.0000 | 0.0812 | 0.1501 |
| LFR7 | Novelty | 1.0000 | 0.0812 | 0.1501 |
| LFR8 | FG | 0.9494 | 0.8242 | 0.8824 |
| LFR8 | Novelty | 0.9114 | 0.7826 | 0.8421 |
| LFR9 | FG | 1.0000 | 0.9836 | 0.9917 |
| LFR9 | Novelty | 1.0000 | 0.9836 | 0.9917 |

This is not evidence of improvement. It shows equal performance on LFR6, LFR7,
and LFR9 and a regression on LFR8. The correct conclusion is that the first
configuration is not yet better than FG and needs validation-based tuning or a
better training objective.

## Tuned held-out smoke result

The validation sweep trained on LFR1-LFR2 and selected:

```text
membership_weight  = 0.20
external_penalty   = 0.00
minimum_membership = 0.60
validation F1      = 0.30510536
```

Using that configuration, retraining on LFR1-LFR5 and evaluating one seed from
each held-out graph produced:

| Dataset | FG F1 | Novelty F1 | Difference |
|---|---:|---:|---:|
| LFR6 | 0.9750 | 0.9750 | 0.0000 |
| LFR7 | 0.1501 | 0.1501 | 0.0000 |
| LFR8 | 0.8824 | 0.9212 | +0.0389 |
| LFR9 | 0.9917 | 0.9917 | 0.0000 |

This is an encouraging but very small smoke result, not a statistically
validated claim. It suggests that the learned candidate score can improve one
moderately difficult graph, but it does not yet demonstrate general
improvement. The full held-out command with seed limit `0` is required before
making a publication claim.

## Five-seed stress result

The same tuned configuration was then run on five seeds per held-out dataset.
This stronger run showed mixed behavior:

| Dataset | FG F1 | Novelty F1 | Difference |
|---|---:|---:|---:|
| LFR6 | 0.8622 | 0.7129 | -0.1493 |
| LFR7 | 0.2679 | 0.2679 | 0.0000 |
| LFR8 | 0.5935 | 0.7026 | +0.1090 |
| LFR9 | 0.6826 | 0.6465 | -0.0361 |

Therefore the current novelty is **not yet a general improvement** over FG. It
helps LFR8, but the learned candidate reranking can select poor candidates on
LFR6 and LFR9. This result is stored in
`output/novelty-tuned-results-5/summary.csv` and is more important than the
one-seed result because it exposes the instability.

The next required refinement is a safe-gating rule: the learned score should
only override the base modularity candidate when its compatibility advantage is
large enough and its modularity gain is within a configurable slack of the
base candidate. This preserves FG's strong decisions while allowing learned
reranking at ambiguous boundaries.

## Safe-gated novelty result

The safe-gating refinement is implemented in `HybridModularityExpansion`. For
each expansion step, the hybrid selector first identifies the best candidate
under the paper's original modularity gain. A learned candidate can replace it
only when its modularity gain is no worse than the base gain minus
`modularity_slack`, and it either has non-negative modularity gain or reaches
the configured membership threshold. This makes the novelty conservative by
construction: it cannot freely replace a clearly stronger FG decision.

The selected configuration was trained on LFR1-LFR5 and evaluated on five
seeds from each held-out graph LFR6-LFR9:

```text
membership_weight = 0.20
external_penalty  = 0.00
minimum_membership = 0.60
modularity_slack   = 0.05
```

The stored artifact is `output/novelty-safe-results-5/summary.csv`:

| Dataset | FG F1 | Safe novelty F1 | Difference |
|---|---:|---:|---:|
| LFR6 | 0.8622 | 0.8622 | +0.0000 |
| LFR7 | 0.2679 | 0.2679 | +0.0000 |
| LFR8 | 0.5935 | 0.7026 | +0.1090 |
| LFR9 | 0.6826 | 0.6465 | -0.0361 |

Across the four held-out datasets, the unweighted mean F1 changes from about
0.6016 to 0.6198, an absolute improvement of about 0.0182. This is a useful
engineering result, but it is not yet a claim of universal superiority: LFR9
still regresses, the evaluation uses five seeds rather than every possible
seed, and paired statistical confidence intervals are still needed for a
publication-quality conclusion. The complete per-seed metrics, model, config,
CSV summaries, and PNG figures are retained under the output directory.

## How to establish improvement

Before claiming novelty, run the following protocol:

1. Train only on LFR1-LFR5.
2. Tune `w`, `lambda`, and the membership threshold on a validation split,
   preferably LFR3-LFR5.
3. Freeze all settings.
4. Evaluate once on LFR6-LFR9.
5. Report per-seed paired differences, mean difference, standard deviation,
   bootstrap confidence intervals, and the number of datasets that improve.
6. Compare against FG, M-based, logistic-only, MLP-only, and the hybrid.

The novelty should be claimed only if the held-out improvement is consistent
and statistically supported. If not, the result is still useful as a negative
study showing that learned candidate compatibility does not automatically solve
FG's precision limitation.
