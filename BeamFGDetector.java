package foldedsubgraph;

import java.util.*;

/** FG detector with beam-search modularity expansion and unchanged relaxation. */
public final class BeamFGDetector {
  private final FoldingCommunity folding;
  private final BeamModularityExpansion modularity;
  private final RelaxationStage relaxation;
  private final Graph graph;

  public BeamFGDetector(
      Graph graph, CandidateScorer scorer, int beamWidth, int maxDepth, double learnedWeight) {
    this(graph, scorer, beamWidth, maxDepth, learnedWeight, Integer.MAX_VALUE);
  }

  public BeamFGDetector(
      Graph graph,
      CandidateScorer scorer,
      int beamWidth,
      int maxDepth,
      double learnedWeight,
      int candidateBudget) {
    this.graph = graph;
    folding = new FoldingCommunity(graph);
    modularity =
        new BeamModularityExpansion(
            graph, scorer, beamWidth, maxDepth, learnedWeight, candidateBudget);
    relaxation = new RelaxationStage(graph);
  }

  public Set<Integer> detect(int seed, double gamma, double h) {
    Set<Integer> community = new LinkedHashSet<>();
    community.add(seed);
    int previousAfterModularity = community.size();
    int previousAfterRelaxation = community.size();
    int iterations = 0;
    while (!closed(community) && iterations++ < graph.size()) {
      community = folding.expand(community, gamma);
      community = modularity.expand(community, gamma);
      int current = community.size();
      if (current == previousAfterRelaxation) break;
      previousAfterModularity = current;
      community = relaxation.expand(community, gamma, h);
      previousAfterRelaxation = community.size();
      if (previousAfterModularity == previousAfterRelaxation) break;
    }
    return community;
  }

  private boolean closed(Set<Integer> community) {
    for (int node : community)
      for (int neighbor : graph.neighbors(node)) if (!community.contains(neighbor)) return false;
    return true;
  }
}
