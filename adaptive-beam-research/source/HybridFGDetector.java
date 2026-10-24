package foldedsubgraph;

import java.util.*;

/** Algorithm 5 with the novel query-conditioned candidate ranking stage. */
public final class HybridFGDetector {
  private final Graph graph;
  private final FoldingCommunity folding;
  private final HybridModularityExpansion modularity;
  private final RelaxationStage relaxation;

  public HybridFGDetector(
      Graph graph,
      CandidateScorer scorer,
      double membershipWeight,
      double externalPenalty,
      double minimumMembership,
      double modularitySlack) {
    this.graph = graph;
    folding = new FoldingCommunity(graph);
    modularity =
        new HybridModularityExpansion(
            graph, scorer, membershipWeight, externalPenalty, minimumMembership, modularitySlack);
    relaxation = new RelaxationStage(graph);
  }

  public Set<Integer> detect(int seed, double gamma, double h) {
    Set<Integer> community = new LinkedHashSet<>();
    community.add(seed);
    int previousAfterModularity = community.size();
    int previousAfterRelaxation = community.size();
    int iterations = 0;
    while (!isClosed(community) && iterations++ < graph.size()) {
      community = folding.expand(community, gamma);
      community = modularity.expand(community, gamma);
      previousAfterModularity = community.size();
      if (previousAfterModularity == previousAfterRelaxation) break;
      community = relaxation.expand(community, gamma, h);
      previousAfterRelaxation = community.size();
    }
    return community;
  }

  public List<HybridModularityExpansion.Decision> decisions() {
    return modularity.decisions();
  }

  public double selectedGainTotal() {
    return modularity.selectedGainTotal();
  }

  private boolean isClosed(Set<Integer> community) {
    for (int node : community) {
      for (int neighbor : graph.neighbors(node)) {
        if (!community.contains(neighbor)) return false;
      }
    }
    return true;
  }
}
