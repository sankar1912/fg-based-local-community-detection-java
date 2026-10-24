# Code and algorithm analysis

## Execution architecture

```text
AdaptiveBeamExperiment
  ├── CandidateTrainingData
  │     └── CandidateFeatures
  ├── LogisticCandidateScorer
  ├── BeamFGDetector
  │     ├── FoldingCommunity
  │     ├── BeamModularityExpansion
  │     │     ├── Folding
  │     │     ├── ModularityExpansion
  │     │     └── CandidateFeatures
  │     └── RelaxationStage
  ├── Metrics
  └── adaptive-beam-results-full/*
```

## Baseline implementation

`FGDetector` follows the original five-algorithm flow:

1. `Folding` creates a dense folded subgraph from a boundary node.
2. `FoldingCommunity` adds folded candidates with at least 50% overlap.
3. `ModularityExpansion` computes `M = ein/eout` and `DeltaM`.
4. `RelaxationStage` computes first- and second-order overlap.
5. `FGDetector` repeats the stages until no progress remains.

The baseline is retained unchanged as the comparison method. The only
performance changes in the shared implementation are mathematically neutral
caches and incremental counts.

## AdaptiveBeam implementation

`BeamModularityExpansion` represents each state as:

```text
(community nodes, accumulated path score)
```

At each depth it generates boundary candidates, ranks them, constructs exact
folded subgraphs for the candidate budget, rejects candidates with negative or
invalid modularity gain, extracts local features, and retains the best unique
communities. Duplicate communities are removed by sorted-node signatures.

The candidate budget is a computational control, not a learned label. It is
selected from validation data and frozen before held-out evaluation.

## Learned component

The logistic scorer is trained only on local candidate features extracted from
the training graphs. It does not receive ground-truth information during
held-out detection. Its probability is a secondary path score and cannot
override the non-negative modularity requirement.

## Runtime engineering

The following changes preserve algorithmic decisions while reducing repeated
work:

- `Folding` maintains incremental connection counts and induced edge counts.
- `ModularityExpansion` reuses community internal and external edge counts for
  all candidates in one state.
- `FoldingCommunity`, `ModularityExpansion`, and `RelaxationStage` cache folded
  subgraphs by start node and density threshold.
- `AdaptiveBeamExperiment` uses thread-local detector contexts, allowing each
  worker to reuse safe local caches across seeds.
- Completed rows are recognized by `(dataset, seed, method)`.
- Interrupted or malformed rows are removed before summary generation.

## Validation and test boundaries

Training graphs: LFR1-LFR5.

Validation graphs: LFR3-LFR5, first five eligible seeds.

Held-out graphs: LFR6-LFR9, all eligible seeds.

The selected budget is 64. `RigorousEvaluationTest` passes after compilation.
The final CSV contains 9,001 lines: one header and 9,000 method-seed rows.

## Known limitations

- The exact raw graph realizations from the source paper are unavailable.
- AdaptiveBeam is worse than FG on LFR6.
- The stored analysis reports means and paired counts; it does not claim
  statistical significance without a prespecified confidence procedure.
- Runtime numbers are implementation and hardware dependent.
