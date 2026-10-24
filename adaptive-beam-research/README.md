# AdaptiveBeam-FG Research Package

This directory is the self-contained research record for the AdaptiveBeam
extension of folded-subgraph local community detection. It gathers the Java
implementation, source inventory, generated benchmark data, every stored
experiment output, figures, the paper analysis, and the IEEE-style manuscript
in one place.

The package is written to make the research claim auditable. The original FG
algorithm and the new search policy are described separately, validation is
kept separate from held-out testing, and the complete all-seed results are
stored rather than represented only by rounded tables.

## Research contribution

The original FG method makes an irreversible greedy modularity decision: it
keeps only the candidate with the largest immediate `DeltaM`. The new method
is named:

> Validation-Tuned Adaptive Candidate-Budget Beam-Search FG

The implementation name is `AdaptiveBeam`.

It preserves the original folded-subgraph density rule, modularity equations,
non-negative modularity eligibility rule, and relaxation stage. The changed
part is the search policy:

1. Rank boundary candidates using inexpensive local connectivity information.
2. Evaluate only a validation-selected candidate budget.
3. Keep a small beam of alternative expansion paths.
4. Rank paths primarily by accumulated modularity gain and secondarily by a
   learned local compatibility score.

The frozen configuration is:

```text
beam_width       = 2
max_depth        = 5
learned_weight   = 0.10
candidate_budget = 64
gamma            = 0.45
h                = 1.0
```

## Complete held-out result

The final evaluation uses every eligible seed in LFR6-LFR9: 3,000 seeds per
method and 9,000 per-seed result rows.

| Method | Mean recall | Mean precision | Mean F1 | Mean runtime |
|---|---:|---:|---:|---:|
| Original FG | 0.948242 | 0.538054 | 0.621031 | 1323.625 ms |
| Full BeamLogistic | 0.889162 | 0.647175 | 0.711682 | 249.489 ms |
| AdaptiveBeam | 0.892300 | 0.648429 | **0.713990** | **180.392 ms** |

AdaptiveBeam improves mean F1 over FG by `+0.092959` (about 14.97% relative)
and is about 27.7% faster than unrestricted beam search in this Java
implementation. The improvement is not universal: LFR6 favors FG, while
AdaptiveBeam improves LFR7-LFR9.

## Directory map

### `source/`

Copies of every Java source file used by the project. Important files include:

- `FGDetector.java`: original end-to-end FG pipeline.
- `Folding.java`: folded-subgraph construction.
- `FoldingCommunity.java`: original folding-community stage.
- `ModularityExpansion.java`: original local modularity equations and
  expansion.
- `RelaxationStage.java`: original relaxation stage.
- `BeamModularityExpansion.java`: multi-path beam expansion.
- `BeamFGDetector.java`: FG pipeline with beam expansion.
- `AdaptiveBeamExperiment.java`: validation, held-out execution, summaries,
  and figure generation for the final method.
- `CandidateFeatures.java`: local candidate feature extraction.
- `LogisticCandidateScorer.java`: learned secondary path scorer.
- `RigorousEvaluationTest.java`: regression and invariant test suite.

### `results/`

All generated experiment directories are retained, including early exploratory
models and the final evaluation:

- `adaptive-beam-results-full/`: final all-seed result, summary, validation
  sweep, configuration, per-seed CSV, and figure.
- `adaptive-beam-results-5/`: five-seed adaptive-beam comparison.
- `beam-search-results-5/`: unrestricted beam comparison.
- `novelty-safe-results-5/`: safe-gated logistic experiment.
- `pairwise-ranking-results-tuned-5/`: pairwise-ranking experiment.
- `boosted-tree-results-5/`: boosted-stump experiment.
- `rigorous-smoke-final2/`: execution smoke test.

The final result is authoritative only in
`results/adaptive-beam-results-full/`. Earlier directories are retained as
research history and negative/ablation evidence.

### `docs/`

- `PAPER_ANALYSIS.md`: mathematical mapping of the supplied paper to code.
- `result-novelty.md`: complete novelty analysis and result interpretation.
- `README.md`: project-level implementation and experiment instructions.
- `novelty.md`: earlier novelty protocol and exploratory notes.

### `paper/`

- `adaptive_beam_fg_ieee.tex`: original IEEE-style manuscript.
- `README.md`: manuscript structure, compilation instructions, and authorship
  replacement instructions.

### `datasets/`

The generated LFR inputs used by the experiments. The paper's exact original
graph realizations and random seeds were not available; therefore the paper
and this package explicitly identify these as deterministic realizations of the
published LFR parameter regimes.

### `references/`

The supplied folded-subgraph paper PDF.

## Reproduce the final result

From the project root:

```text
javac -Xlint:all -d out *.java
java -cp out foldedsubgraph.RigorousEvaluationTest
java -cp out foldedsubgraph.AdaptiveBeamExperiment ^
  datasets/generated-lfr output/adaptive-beam-results-full 0
```

The final command is restartable. It skips valid completed rows, removes
partial rows left by interruption, and regenerates the summary and figure.

## Manuscript status

The manuscript is original prose written for this implementation and follows
IEEE journal organization. It cites the source paper rather than reproducing
its language or claiming its unavailable exact tables. Replace the author and
affiliation placeholders before submission. A LaTeX distribution containing
`IEEEtran.cls` is required to generate the PDF; compilation instructions are
in `paper/README.md`.

## Reproducibility boundary

The final numerical claims are about this Java implementation, the stored LFR
realizations, the frozen configuration, and the 3,000 held-out seeds. They are
not presented as an exact reproduction of the authors' private graph files or
random seeds. This distinction is part of the result, not an omission.
