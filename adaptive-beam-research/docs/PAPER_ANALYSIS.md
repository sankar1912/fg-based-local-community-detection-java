# Paper analysis and implementation map

Paper: Mengting Zhang and Weihong Bi, “A Local Community Detection Method
Based on Folded Subgraph,” IEEE TKDE, vol. 37, no. 7, 2025, pp. 3869-3880.

## Objective

Given a seed node and only local topology, recover its ground-truth community.
The paper targets precision limitation in local-modularity methods and network
noise. The proposed FG-based method compresses tightly related candidate nodes
into folded subgraphs and expands a community in three stages.

## Exact method implemented

1. **Algorithm 1 - Folded Subgraph (`Folding`)**: start `F={v}`, set
   `R=N(v)`. Select from `R` the node with the most links to `F`; ties use
   the lowest degree. Tentatively add it and calculate induced link density
   `2E/(|F|(|F|-1))`. Stop if density is below `gamma`; otherwise add it and
   update `R`.
2. **Algorithm 2 - Folding Stage (`FoldingCommunity`)**: for each candidate,
   build `F(v)`. Add the entire folded subgraph when at least half of its
   nodes already belong to `C`; repeat until no node is added.
3. **Algorithm 3 - Modularity Stage (`ModularityExpansion`)**: use
   `M=ein/eout`. For every candidate folded subgraph calculate the paper's
   `x`, `y`, `z`, and `Delta M` equations (2)-(5), add the maximum positive
   candidate, and repeat.
4. **Algorithm 4 - Relaxation Stage (`RelaxationStage`)**: calculate
   `f1=|N(F) intersection C|/|C|` and
   `f2=|NN(F) intersection C|/|C|`; maximize `f=f1+beta*f2` with
   `beta=0.5`. Add the best folded subgraph only when `f >= h`.
5. **Algorithm 5 - FG-based (`FGDetector`)**: repeat Algorithms 2 and 3,
   stop when modularity expansion makes no progress, then run Algorithm 4 and
   continue until the outer loop terminates.

## Evaluation protocol

For every node in a dataset, use that node as the seed, compare the detected
community with the ground-truth community containing it, and average recall,
precision, and F-score. `ExperimentRunner` writes both per-seed rows and the
final average to CSV.

## Datasets named by the paper

The experiments use Karate, Dolphins, US politics books, Amazon, and nine LFR
networks. The paper also discusses a community-loop network and LFR(0.46) and
LFR(0.18) for parameter analysis. LFR is described by `N`, average degree `k`,
maximum degree `maxk`, mixing coefficient `mu`, degree/community exponents
`t1/t2`, community-size bounds `minc/maxc`, and overlap parameters `on/om`.

The PDF does not ship raw graph files, full generator seeds, or realization
files. Consequently, this repository can reproduce the algorithm and metric
protocol exactly, but cannot honestly claim identical Table I-V numbers unless
the authors' exact files and random seeds are supplied. `datasets/` is left
for those files rather than silently substituting different graphs.

The exact Table I parameter rows are encoded in `LfrSpec.java`: LFR1-LFR5
use N=200, k=7, maxk=15, mu=0.1..0.5, minc=10, maxc=80; LFR6 uses
N=1000, k=15, maxk=50, mu=0.1, minc=10, maxc=50; LFR7 uses the same values
with maxc=300; and LFR8-LFR9 use N=500, k=7, maxk=15, mu=0.1, minc=10,
maxc=80. All rows use t1=2, t2=1, on=0, om=0.

## Reported findings captured from the paper

- On Karate, Dolphins, US politics books, and Amazon, FG-based recall improves
  over M-based by 55.23%, 175.37%, 82.92%, and 26.6%, respectively.
- Precision improves on Karate and drops only about 0.01-0.03 on the other
  three real datasets; FG-based has the best F-score among compared methods.
- On LFR with mixing coefficient at most 0.3, folded subgraphs and relaxation
  both improve the local result; with mixing above 0.3, local modularity itself
  becomes unstable and FG-based performance declines rapidly.
- The paper reports robustness across LFR community sizes and network scales,
  with higher average cost than M/R baselines but lower F-score variance.
- Parameter guidance: use larger `gamma` and `h` for dense communities; for
  lower intra-community density, use a smaller `h` and typically `gamma` in
  about 0.55-0.8.

## Local validation result

Using the included standard Karate graph and two-faction truth labels, the
parameter sweep selected `gamma=0.45, h=1.0` by average F-score. Across all
34 seeds, FG-based achieved recall `0.946367`, precision `0.921053`, and
F-score `0.931818`. The included single-node M-based baseline achieved recall
`0.565744`, precision `0.878262`, and F-score `0.652427`. These are a local
validation result, not a claim that they are the paper's Table III values:
the paper's exact preprocessing and source files are unavailable.
