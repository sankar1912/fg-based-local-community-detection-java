package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Tunes novelty weights on validation graphs LFR3-LFR5 without touching LFR6-LFR9. */
public final class NoveltyParameterSweep {
  private static final double GAMMA = 0.45;
  private static final double H = 1.0;

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args.length > 0 ? args[0] : "datasets/generated-lfr");
    Path output = Path.of(args.length > 1 ? args[1] : "output/novelty-tuning");
    int limit = args.length > 2 ? Integer.parseInt(args[2]) : 1;
    Files.createDirectories(output);
    LogisticCandidateScorer scorer = trainOnLfr1And2(data);
    double[] weights = {.2, .5, .8};
    double[] penalties = {0.0, .1};
    double[] thresholds = {.6, .8, .9};
    double bestF1 = Double.NEGATIVE_INFINITY;
    String best = "";
    try (BufferedWriter writer = Files.newBufferedWriter(output.resolve("tuning.csv"))) {
      writer.write("membership_weight,external_penalty,minimum_membership,mean_f1\n");
      for (double weight : weights) {
        for (double penalty : penalties) {
          for (double threshold : thresholds) {
            double f1 = evaluate(data, scorer, limit, weight, penalty, threshold);
            writer.write(String.format(Locale.ROOT, "%.2f,%.2f,%.2f,%.8f%n",
                weight, penalty, threshold, f1));
            if (f1 > bestF1) {
              bestF1 = f1;
              best = String.format(Locale.ROOT,
                  "membership_weight=%.2f\nexternal_penalty=%.2f\nminimum_membership=%.2f\nvalidation_f1=%.8f\n",
                  weight, penalty, threshold, f1);
            }
          }
        }
      }
    }
    Files.writeString(output.resolve("best-config.txt"), best);
    scorer.save(output.resolve("validation-scorer.txt"));
    System.out.println(best);
  }

  private static LogisticCandidateScorer trainOnLfr1And2(Path data) throws IOException {
    List<LogisticCandidateScorer.TrainingExample> examples = new ArrayList<>();
    for (int index = 0; index < 2; index++) {
      LfrSpec spec = LfrSpec.paperTable()[index];
      Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
      List<Set<Integer>> truth = DatasetReader.communities(
          data.resolve(spec.name() + ".truth"), false);
      examples.addAll(CandidateTrainingData.collect(graph, truth, GAMMA));
    }
    return LogisticCandidateScorer.train(examples, 500, .08, .001);
  }

  private static double evaluate(
      Path data,
      LogisticCandidateScorer scorer,
      int limit,
      double weight,
      double penalty,
      double threshold) throws IOException {
    double total = 0.0;
    int count = 0;
    for (int index = 2; index < 5; index++) {
      LfrSpec spec = LfrSpec.paperTable()[index];
      Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
      List<Set<Integer>> truth = DatasetReader.communities(
          data.resolve(spec.name() + ".truth"), false);
      HybridFGDetector hybrid = new HybridFGDetector(graph, scorer, weight, penalty, threshold);
      int seeds = limit <= 0 ? graph.size() : Math.min(limit, graph.size());
      for (int seed = 0; seed < seeds; seed++) {
        Metrics.Score score = Metrics.score(
            hybrid.detect(seed, GAMMA, H), Metrics.truthFor(seed, truth));
        total += score.f1();
        count++;
      }
    }
    return total / count;
  }
}
