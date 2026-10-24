package foldedsubgraph;

import java.nio.file.*;
import java.util.*;

/** Dependency-free regression and safety checks for the rigorous evaluator. */
public final class RigorousEvaluationTest {
  public static void main(String[] args) throws Exception {
    Graph graph = DatasetReader.edgeList(Path.of("datasets/karate.edgelist"), false);
    List<Set<Integer>> truth = DatasetReader.communities(Path.of("datasets/karate.truth"), false);
    for (int seed = 0; seed < Math.min(10, graph.size()); seed++) {
      Set<Integer> first = new FGDetector(graph).detect(seed, .45, 1.0);
      Set<Integer> second = new FGDetector(graph).detect(seed, .45, 1.0);
      if (!first.equals(second))
        throw new AssertionError("FG baseline is not deterministic for seed " + seed);
    }
    List<LogisticCandidateScorer.TrainingExample> examples =
        CandidateTrainingData.collect(graph, truth, .45);
    CandidateScorer nanModel =
        new CandidateScorer() {
          @Override
          public double probability(double[] ignored) {
            return Double.NaN;
          }

          @Override
          public String modelType() {
            return "test-nan-model";
          }
        };
    for (int seed = 0; seed < Math.min(10, graph.size()); seed++) {
      Set<Integer> baseline = new FGDetector(graph).detect(seed, .45, 1.0);
      Set<Integer> fallback =
          new HybridFGDetector(graph, nanModel, .8, 0, .6, .05).detect(seed, .45, 1.0);
      if (!baseline.equals(fallback))
        throw new AssertionError("NaN scorer did not fall back to FG for seed " + seed);
    }
    CandidateScorer rbm = RbmCandidateScorer.train(examples, 4, 2, .01, .0001, 17L);
    double probability = rbm.probability(examples.get(0).features());
    if (!rbm.modelType().startsWith("rbm-")
        || !Double.isFinite(probability)
        || probability < 0
        || probability > 1) {
      throw new AssertionError("RBM scorer is invalid");
    }
    Set<Integer> beamFirst = new BeamFGDetector(graph, rbm, 2, 3, .1).detect(0, .45, 1.0);
    Set<Integer> beamSecond = new BeamFGDetector(graph, rbm, 2, 3, .1).detect(0, .45, 1.0);
    if (!beamFirst.equals(beamSecond) || beamFirst.isEmpty()) {
      throw new AssertionError("Beam detector is not deterministic or returned an empty community");
    }
    System.out.println("RigorousEvaluationTest: PASS");
  }
}
