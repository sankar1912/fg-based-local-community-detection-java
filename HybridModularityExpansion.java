package foldedsubgraph;

import java.util.*;

/**
 * Novel candidate-selection stage. The original folded subgraph, M equation, and stopping
 * safeguards remain intact; only candidate ranking changes.
 */
public final class HybridModularityExpansion {
  private final Graph graph;
  private final Folding folding;
  private final ModularityExpansion modularity;
  private final LogisticCandidateScorer scorer;
  private final double membershipWeight;
  private final double externalPenalty;
  private final double minimumMembership;
  private final double modularitySlack;
  private final Map<Integer, Set<Integer>> foldedCache = new HashMap<>();

  public HybridModularityExpansion(
      Graph graph,
      LogisticCandidateScorer scorer,
      double membershipWeight,
      double externalPenalty,
      double minimumMembership,
      double modularitySlack) {
    this.graph = graph;
    this.folding = new Folding(graph);
    this.modularity = new ModularityExpansion(graph);
    this.scorer = scorer;
    this.membershipWeight = membershipWeight;
    this.externalPenalty = externalPenalty;
    this.minimumMembership = minimumMembership;
    this.modularitySlack = modularitySlack;
  }

  public Set<Integer> expand(Set<Integer> initial, double gamma) {
    Set<Integer> community = new LinkedHashSet<>(initial);
    int iterations = 0;
    while (iterations++ < graph.size()) {
      Set<Integer> alternatives = alternatives(community);
      if (alternatives.isEmpty()) return community;

      List<Candidate> candidates = new ArrayList<>();
      for (int candidate : alternatives) {
        Set<Integer> folded =
            foldedCache.computeIfAbsent(candidate, value -> folding.build(value, gamma));
        if (folded.stream().noneMatch(node -> !community.contains(node))) continue;
        double delta = modularity.delta(community, alternatives, folded);
        double[] features =
            CandidateFeatures.extract(graph, community, alternatives, folded, gamma);
        double probability = scorer.probability(features);
        candidates.add(new Candidate(folded, delta, probability, features[5]));
      }
      if (candidates.isEmpty()) return community;

      double minimumDelta = candidates.stream().mapToDouble(Candidate::delta).min().orElse(0.0);
      double maximumDelta = candidates.stream().mapToDouble(Candidate::delta).max().orElse(0.0);
      Candidate best = null;
      double bestScore = Double.NEGATIVE_INFINITY;
      double bestBaseDelta = candidates.stream().mapToDouble(Candidate::delta).max().orElse(0.0);
      for (Candidate candidate : candidates) {
        double normalizedDelta = normalize(candidate.delta(), minimumDelta, maximumDelta);
        double score =
            (1.0 - membershipWeight) * normalizedDelta
                + membershipWeight * candidate.probability()
                - externalPenalty * candidate.externalRatio();
        boolean safe = candidate.delta() >= 0.0 || candidate.probability() >= minimumMembership;
        boolean nearBase = candidate.delta() >= bestBaseDelta - modularitySlack;
        if (safe && nearBase && score > bestScore) {
          best = candidate;
          bestScore = score;
        }
      }
      if (best == null) return community;
      community.addAll(best.folded());
    }
    return community;
  }

  private Set<Integer> alternatives(Set<Integer> community) {
    Set<Integer> alternatives = new LinkedHashSet<>();
    for (int node : community) {
      for (int neighbor : graph.neighbors(node)) {
        if (!community.contains(neighbor)) alternatives.add(neighbor);
      }
    }
    return alternatives;
  }

  private static double normalize(double value, double minimum, double maximum) {
    if (maximum - minimum < 1e-12) return 0.5;
    return (value - minimum) / (maximum - minimum);
  }

  private record Candidate(
      Set<Integer> folded, double delta, double probability, double externalRatio) {}
}
