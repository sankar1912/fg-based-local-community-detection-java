package foldedsubgraph;

import java.util.*;

/** Beam-search replacement for the irreversible greedy modularity expansion. */
public final class BeamModularityExpansion {
  private final Graph graph;
  private final Folding folding;
  private final ModularityExpansion modularity;
  private final CandidateScorer scorer;
  private final int beamWidth;
  private final int maxDepth;
  private final double learnedWeight;
  private final Map<Integer, Set<Integer>> foldedCache = new HashMap<>();

  public BeamModularityExpansion(
      Graph graph, CandidateScorer scorer, int beamWidth, int maxDepth, double learnedWeight) {
    this.graph = graph;
    this.folding = new Folding(graph);
    this.modularity = new ModularityExpansion(graph);
    this.scorer = scorer;
    this.beamWidth = beamWidth;
    this.maxDepth = maxDepth;
    this.learnedWeight = learnedWeight;
  }

  public Set<Integer> expand(Set<Integer> initial, double gamma) {
    List<State> beam = List.of(new State(new LinkedHashSet<>(initial), 0));
    State best = beam.get(0);
    for (int depth = 0; depth < maxDepth; depth++) {
      List<State> next = new ArrayList<>();
      for (State state : beam) {
        Set<Integer> alternatives = alternatives(state.community());
        for (int candidate : alternatives) {
          Set<Integer> folded =
              foldedCache.computeIfAbsent(candidate, v -> folding.build(v, gamma));
          if (folded.stream().noneMatch(node -> !state.community().contains(node))) continue;
          double delta = modularity.delta(state.community(), alternatives, folded);
          if (!Double.isFinite(delta) || delta < 0) continue;
          double[] features =
              CandidateFeatures.extract(graph, state.community(), alternatives, folded, gamma);
          double probability = scorer == null ? .5 : scorer.probability(features);
          if (!Double.isFinite(probability)) probability = .5;
          Set<Integer> community = new LinkedHashSet<>(state.community());
          community.addAll(folded);
          next.add(new State(community, state.totalGain() + delta + learnedWeight * probability));
        }
      }
      if (next.isEmpty()) break;
      next.sort(
          Comparator.comparingDouble(State::totalGain)
              .reversed()
              .thenComparingInt(s -> s.community().size()));
      List<State> unique = new ArrayList<>();
      Set<String> signatures = new HashSet<>();
      for (State state : next) {
        String signature =
            state.community().stream()
                .sorted()
                .map(String::valueOf)
                .collect(java.util.stream.Collectors.joining("|"));
        if (signatures.add(signature)) unique.add(state);
        if (unique.size() == beamWidth) break;
      }
      beam = unique;
      if (beam.get(0).totalGain() > best.totalGain()) best = beam.get(0);
    }
    return best.community();
  }

  private Set<Integer> alternatives(Set<Integer> community) {
    Set<Integer> result = new LinkedHashSet<>();
    for (int node : community)
      for (int neighbor : graph.neighbors(node))
        if (!community.contains(neighbor)) result.add(neighbor);
    return result;
  }

  private record State(Set<Integer> community, double totalGain) {}
}
