package foldedsubgraph;

/** Common prediction interface for learned candidate selectors. */
public interface CandidateScorer {
  double probability(double[] features);

  String modelType();
}
