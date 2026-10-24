package foldedsubgraph;

import java.util.*;

/** Builds labels only from designated training graphs. */
public final class CandidateTrainingData {
  private CandidateTrainingData() {}

  public static List<LogisticCandidateScorer.TrainingExample> collect(
      Graph graph, List<Set<Integer>> truths, double gamma) {
    List<LogisticCandidateScorer.TrainingExample> examples = new ArrayList<>();
    Folding folding = new Folding(graph);
    Map<Integer, Set<Integer>> foldedCache = new HashMap<>();
    for (int seed = 0; seed < graph.size(); seed++) {
      Set<Integer> community = new LinkedHashSet<>();
      community.add(seed);
      Set<Integer> alternative = alternatives(graph, community);
      Set<Integer> truth = Metrics.truthFor(seed, truths);
      for (int candidate : alternative) {
        Set<Integer> folded =
            foldedCache.computeIfAbsent(candidate, value -> folding.build(value, gamma));
        double[] features = CandidateFeatures.extract(graph, community, alternative, folded, gamma);
        int inTruth = 0;
        for (int node : folded) if (truth.contains(node)) inTruth++;
        int label = inTruth >= Math.ceil(folded.size() / 2.0) ? 1 : 0;
        examples.add(new LogisticCandidateScorer.TrainingExample(features, label));
      }
    }
    return examples;
  }

  private static Set<Integer> alternatives(Graph graph, Set<Integer> community) {
    Set<Integer> alternatives = new LinkedHashSet<>();
    for (int node : community) {
      for (int neighbor : graph.neighbors(node)) {
        if (!community.contains(neighbor)) alternatives.add(neighbor);
      }
    }
    return alternatives;
  }
}
