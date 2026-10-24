# FG-Based Local Community Detection in Java

This project implements the method from **“A Local Community Detection Method
Based on Folded Subgraph”** by Mengting Zhang and Weihong Bi, IEEE TKDE 2025.
It is a dependency-free Java project: the algorithm, graph loading, benchmark
generation, metric calculation, CSV result storage, and PNG report generation
are all implemented in Java.

## What the project does

For a selected seed node, the FG-based detector finds its local community using
three stages:

1. Folding: construct a dense folded subgraph around each candidate.
2. Modularity expansion: greedily add the folded subgraph with the best
   positive gain in local modularity `M = ein / eout`.
3. Relaxation: add one further candidate using first- and second-order
   neighborhood similarity.

The implementation evaluates Recall, Precision, F-score, runtime-oriented
experiment records, and variance across seed nodes. It can write per-seed CSV
files and PNG images for the paper tables, metric curves, parameter analysis,
and detected network structure.

## Source files

### Core graph and input modules

| File | Responsibility |
|---|---|
| `Graph.java` | Undirected simple graph with adjacency sets, degrees, edge count, and edge lookup. |
| `DatasetReader.java` | Reads edge-list graphs, the original adjacency-list format, and one-community-per-line ground truth files. Supports zero-based and one-based IDs. |
| `Metrics.java` | Computes Recall, Precision, F-score, and finds the ground-truth community containing a seed. |

### Paper algorithm modules

| File | Responsibility |
|---|---|
| `Folding.java` | Algorithm 1. Greedy folded-subgraph construction, density threshold `gamma`, connection-count tie-breaking, and density calculation. |
| `FoldingCommunity.java` | Algorithm 2. Repeated folding-stage expansion; adds a folded subgraph when at least half of it overlaps the current community. |
| `ModularityExpansion.java` | Algorithm 3. Local modularity `M`, the paper's `x`, `y`, `z`, and `Delta M` calculations, and greedy positive-gain expansion. |
| `RelaxationStage.java` | Algorithm 4. Computes `f1`, `f2`, and `f = f1 + beta*f2`; uses the paper's `beta = 0.5`. |
| `FGDetector.java` | Algorithm 5. Orchestrates the folding, modularity, and relaxation stages until the paper's stopping condition is reached. |
| `MBaseline.java` | Single-node M-based baseline used for comparison with FG-based results. |
| `CandidateFeatures.java` | Extracts local structural features for a `(current community, folded candidate)` pair. |
| `CandidateTrainingData.java` | Creates supervised candidate examples from training graphs only. |
| `LogisticCandidateScorer.java` | Trains and stores the standardized logistic compatibility model used by the novelty. |
| `HybridModularityExpansion.java` | Novel candidate reranking stage using normalized modularity, learned compatibility, and external-edge penalty. |
| `HybridFGDetector.java` | Full FG pipeline with only the modularity-stage candidate selection replaced. |

### Command-line runners

| File | Responsibility |
|---|---|
| `Main.java` | Runs FG-based detection for one seed and prints the detected community and metrics. |
| `RunExperiments.java` | Runs every seed in one graph and writes a per-seed CSV plus an average row. |
| `ExperimentRunner.java` | Reusable Java API behind batch evaluation. |
| `CompareMethods.java` | Compares FG-based and M-based average Recall, Precision, and F-score. |
| `ParameterSweep.java` | Searches `gamma` and `h` and selects the highest average F-score. |
| `LfrExperimentRunner.java` | Runs the generated LFR benchmark files and writes dataset-level summaries. |
| `RunAllResults.java` | Complete result-bundle command: generates datasets, runs experiments, stores metadata, and produces all PNG reports. |

### Dataset and benchmark modules

| File | Responsibility |
|---|---|
| `LfrSpec.java` | Stores the exact nine LFR parameter rows transcribed from the paper's Table I. |
| `SyntheticLfrGenerator.java` | Creates deterministic large benchmark graphs with the paper's LFR node counts and macro parameters. |
| `datasets/karate.edgelist` | Included standard 34-node, 78-edge Karate graph. |
| `datasets/karate.truth` | Included two-faction Karate ground truth. |

### PNG report modules

| File | Responsibility |
|---|---|
| `ReportGenerator.java` | Generates paper-reference tables, Karate per-seed curves, and a Karate detected-community network image using Java2D. |
| `PaperMetricImages.java` | Generates Recall, Precision, and F-score images for the paper's Tables III and IV. |
| `AblationMetricImages.java` | Generates Recall, Precision, and F-score images for Table VI's three-stage ablation. |
| `GeneratePaperOutputs.java` | Generates all nine benchmark files and the standard PNG report set. |
| `NoveltyExperiment.java` | Trains on LFR1-LFR5 and compares base FG against the novelty on held-out LFR6-LFR9. |
| `NoveltyParameterSweep.java` | Tunes membership weight, external penalty, and minimum compatibility on validation LFR3-LFR5 after training on LFR1-LFR2. |
| `NoveltyVisualization.java` | Converts computed novelty summary CSV files into base-versus-novelty PNG charts. |

## Build

Open a terminal in `E:\VII Sem\Monicka` and run:

```cmd
javac -d out *.java
```

If `out` already exists, that is normal. The compiler overwrites the class
files that it needs. Do not use placeholder filenames unless those files exist.

## Run one seed

For the included one-based Karate files:

```cmd
java -cp out foldedsubgraph.Main ^
  datasets/karate.edgelist datasets/karate.truth 1 0.45 1.0 true
```

Arguments:

```text
Main graph-file truth-file seed gamma h oneBased
```

For zero-based input, omit the final `true` or set it to `false`.

## Run all seeds in one dataset

```cmd
java -cp out foldedsubgraph.RunExperiments ^
  datasets/karate.edgelist datasets/karate.truth ^
  output/karate-results.csv 0.45 1.0 true
```

The CSV contains one row per seed and a final `average` row with Recall,
Precision, and F-score.

## Tune parameters

```cmd
java -cp out foldedsubgraph.ParameterSweep ^
  datasets/karate.edgelist datasets/karate.truth true
```

The sweep evaluates `gamma` from 0.25 to 0.95 and `h` from 0.1 to 1.5. For
the included Karate graph, the best tested setting was:

```text
gamma = 0.45
h     = 1.0
Recall    = 0.946367
Precision = 0.921053
F-score   = 0.931818
```

## Generate all paper-sized benchmark files and images

```cmd
java -cp out foldedsubgraph.GeneratePaperOutputs ^
  datasets/generated-lfr output/figures

java -cp out foldedsubgraph.PaperMetricImages output/figures
java -cp out foldedsubgraph.AblationMetricImages output/figures
```

The generated LFR sizes are:

```text
LFR1  200 nodes
LFR2  200 nodes
LFR3  200 nodes
LFR4  200 nodes
LFR5  200 nodes
LFR6  1000 nodes
LFR7  1000 nodes
LFR8  500 nodes
LFR9  500 nodes
```

The PNG directory contains:

- Table III Recall, Precision, and F-score images
- Table IV Recall, Precision, and F-score images
- Table V runtime, Recall, Precision, F-score, and variance image
- Table VI ablation Recall, Precision, and F-score images
- Karate per-seed metric curves
- Karate detected-community network graph

## Store one complete result bundle

```cmd
java -cp out foldedsubgraph.RunAllResults ^
  output/results datasets/generated-lfr 1
```

This writes:

```text
output/results/
  run-metadata.txt
  karate-per-seed.csv
  lfr1-per-seed.csv ... lfr9-per-seed.csv
  figures/*.png
```

The last argument is the LFR seed limit. Use `1` for a quick complete smoke
run. Use `0` to evaluate every node in all nine LFR graphs:

```cmd
java -cp out foldedsubgraph.RunAllResults ^
  output/results datasets/generated-lfr 0
```

The `average` row in each CSV records the exact number of evaluated seeds.

## Novelty experiment

The proposed novelty is documented in [novelty.md](novelty.md). It is a
query-conditioned energy-style candidate scorer: the current community and
folded candidate are represented by local structural features, and a learned
compatibility score reranks candidates during the modularity stage. The
original FG method remains unchanged as the baseline.

The consolidated results, tables, figure locations, model comparison, and
full-run status are in [result-novelty.md](result-novelty.md).

Run a fast held-out experiment:

```cmd
java -cp out foldedsubgraph.NoveltyExperiment ^
  datasets/generated-lfr output/novelty-results 1
```

Run all nodes in held-out LFR6-LFR9:

```cmd
java -cp out foldedsubgraph.NoveltyExperiment ^
  datasets/generated-lfr output/novelty-results 0
```

The novelty output stores the trained model, training metadata, per-seed base
and novelty metrics, an aggregate summary, and comparison PNGs. The first
executed one-seed smoke experiment was equal to FG on LFR6, LFR7, and LFR9 and
worse on LFR8; it must not be described as an improvement until the held-out
tuning protocol in `novelty.md` produces a consistent gain.

The safe-gated five-seed comparison is stored in
`output/novelty-safe-results-5/`. It preserves the FG result on LFR6 and LFR7,
improves LFR8 from F1 0.5935 to 0.7026, and regresses LFR9 from 0.6826 to
0.6465. The aggregate mean improves modestly, but the result is mixed and
should be reported with statistical uncertainty.

Tune the novelty parameters on validation graphs:

```cmd
java -cp out foldedsubgraph.NoveltyParameterSweep ^
  datasets/generated-lfr output/novelty-tuning 1
```

This stores `tuning.csv`, `best-config.txt`, and the validation scorer. Use the
selected values plus a modularity slack of `0.05` as the final four arguments
to `NoveltyExperiment`. The full optional argument order is:

```text
seedLimit membershipWeight externalPenalty minimumMembership modularitySlack
```

## Rigorous held-out evaluation

`RigorousEvaluation.java` is the reproducible evaluator for the research claim.
It keeps the original `FGDetector` baseline, tunes only on LFR3-LFR5 after
training on LFR1-LFR2, freezes the selected configuration, retrains on
LFR1-LFR5, and evaluates LFR6-LFR9. The comparison includes logistic
regression, a one-hidden-layer MLP, and a genuine Bernoulli-Bernoulli RBM
trained with CD-1. The RBM is not a label applied to the logistic model.

Run a quick format/execution check with one eligible seed per held-out graph:

```cmd
java -cp out foldedsubgraph.RigorousEvaluation ^
  datasets/generated-lfr output/rigorous-smoke 1
```

Run the complete eligible-seed experiment without the optional seed limit:

```cmd
java -cp out foldedsubgraph.RigorousEvaluation ^
  datasets/generated-lfr output/rigorous-evaluation
```

The evaluator is restartable: completed successful dataset/seed/model rows in
`per_seed_results.csv` are skipped. It writes paired rows, validation sweep,
selected configuration, manifest, override audit, candidate-ranking metrics,
bootstrap confidence intervals, paired sign-permutation tests with Holm
adjustment, model comparison, report, and five PNG figures. A run using a
positive third argument is a smoke run and must not be reported as the final
all-seed experiment.

The verified smoke artifact in this workspace is
`output/rigorous-smoke-final2/`. It contains one eligible seed from each test
graph and validates execution and file formats only. The no-limit command above
is the only command that can produce final all-eligible-seed evidence.

## Input formats

### Edge list

One undirected edge per line:

```text
0 1
1 2
2 0
```

Comments beginning with `#` or `%` are ignored. Duplicate and self-loop edges
are safely handled by `Graph`.

### Ground truth

One community per line:

```text
0 1 2
3 4 5
```

The evaluator selects the community containing the seed node.

## Reproducibility and dataset accuracy

The supplied paper gives the exact LFR parameter table and result tables, but
does not include the original raw graph realizations or random seeds. The Java
generator therefore preserves the paper's node counts and macro parameters
with a deterministic seed, but it is a planted benchmark replacement, not a
claim that it is the authors' original LFR realization.

The PNGs named `paper-table-*` contain values transcribed from the supplied
paper for reference. CSV files named `*-per-seed.csv` contain values actually
computed by this Java implementation. These two sources are intentionally
kept distinct.

For exact Table III-V reproduction, provide the original Karate, Dolphins,
US-politics-books, Amazon, and LFR graph/truth files used by the authors,
then run the same commands with those paths. Record the graph hash, ID base,
generator version, seed, `gamma`, and `h` in the experiment metadata.

## Project documentation

See [PAPER_ANALYSIS.md](PAPER_ANALYSIS.md) for the equations, algorithm
mapping, reported findings, exact Table I LFR parameters, and the distinction
between paper-reference values and code-generated results.
