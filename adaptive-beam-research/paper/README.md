# AdaptiveBeam-FG IEEE manuscript

`adaptive_beam_fg_ieee.tex` is an original IEEE journal-style manuscript for
the implemented method. It follows the conventional research-paper order:

1. abstract and keywords;
2. introduction and contributions;
3. background and mathematical description of the original FG method;
4. failure analysis;
5. the proposed AdaptiveBeam method and pseudocode;
6. experimental protocol;
7. complete all-seed results;
8. discussion, limitations, reproducibility, and conclusion;
9. references.

The manuscript is independently written for this repository. It cites the
folded-subgraph paper as the source of the baseline method and does not copy
its prose, tables, or claimed results. Results in the manuscript are produced
by this Java implementation and are explicitly distinguished from the paper's
published values because the paper's original graph realizations and random
seeds are not available here.

## Compile locally

Install a LaTeX distribution containing `IEEEtran.cls`, then run from the
`paper/` directory:

```text
cd paper
pdflatex -interaction=nonstopmode adaptive_beam_fg_ieee.tex
pdflatex -interaction=nonstopmode adaptive_beam_fg_ieee.tex
```

The second pass resolves references. If the compiler cannot find the figure
because of its working directory, confirm that the command is being run from
`paper/`; the manuscript references the generated figure through
`../output/adaptive-beam-results-full/figures/...`.

Replace the author and affiliation block before submission. The manuscript is
research content, not a declaration of authorship.

## Source of reported results

- `../output/adaptive-beam-results-full/summary.csv`
- `../output/adaptive-beam-results-full/per-seed.csv`
- `../output/adaptive-beam-results-full/validation-sweep.csv`
- `../result-novelty.md`
