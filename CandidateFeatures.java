package foldedsubgraph;

import java.util.*;

/**
 * Local, prediction-time features for a folded candidate F and current
 * community C. No ground-truth information is used here.
 */
public final class CandidateFeatures {
  public static final String[] NAMES = {
    "folded_density", "community_overlap", "first_order_overlap",
    "second_order_overlap", "modularity_gain", "external_edge_ratio",
    "candidate_size", "mean_degree", "internal_candidate_ratio"
  };

  private CandidateFeatures() {}

  public static double[] extract(
      Graph graph,
      Set<Integer> community,
      Set<Integer> alternative,
      Set<Integer> folded,
      double gamma) {
    Folding folding = new Folding(graph);
    ModularityExpansion modularity = new ModularityExpansion(graph);
    double density = folding.density(folded);
    double first = overlap(graph, folded, community, false);
    double second = overlap(graph, folded, community, true);
    double delta = modularity.delta(community, alternative, folded);
    if (!Double.isFinite(delta)) delta = -10.0;

    int external = 0;
    int totalDegree = 0;
    int internal = 0;
    for (int node : folded) {
      totalDegree += graph.degree(node);
      for (int neighbor : graph.neighbors(node)) {
        if (!community.contains(neighbor) && !folded.contains(neighbor)) external++;
        if (folded.contains(neighbor)) internal++;
      }
    }
    double externalRatio = totalDegree == 0 ? 0.0 : external / (double) totalDegree;
    double meanDegree = folded.isEmpty() ? 0.0 : totalDegree / (double) folded.size();
    double internalRatio = totalDegree == 0 ? 0.0 : internal / (double) totalDegree;
    double communityOverlap = community.isEmpty()
        ? 0.0
        : intersectionSize(folded, community) / (double) community.size();

    return new double[] {
      density,
      communityOverlap,
      first,
      second,
      delta,
      externalRatio,
      folded.size(),
      meanDegree,
      internalRatio
    };
  }

  private static double overlap(
      Graph graph, Set<Integer> folded, Set<Integer> community, boolean secondOrder) {
    Set<Integer> reached = new HashSet<>();
    for (int node : folded) {
      for (int neighbor : graph.neighbors(node)) {
        if (secondOrder) {
          for (int second : graph.neighbors(neighbor)) {
            if (community.contains(second)) reached.add(second);
          }
        } else if (community.contains(neighbor)) {
          reached.add(neighbor);
        }
      }
    }
    return community.isEmpty() ? 0.0 : reached.size() / (double) community.size();
  }

  private static int intersectionSize(Set<Integer> left, Set<Integer> right) {
    int count = 0;
    for (int node : left) if (right.contains(node)) count++;
    return count;
  }
}
