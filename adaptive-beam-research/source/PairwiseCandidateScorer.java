package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Query-conditioned pairwise learning-to-rank scorer. It trains on positive-minus-negative
 * candidate pairs from the same seed query, so it learns ordering rather than an independent
 * pointwise membership label.
 */
public final class PairwiseCandidateScorer implements CandidateScorer {
  private final double[] mean;
  private final double[] scale;
  private final double[] weights;

  private PairwiseCandidateScorer(double[] mean, double[] scale, double[] weights) {
    this.mean = mean;
    this.scale = scale;
    this.weights = weights;
  }

  public static PairwiseCandidateScorer train(
      List<List<LogisticCandidateScorer.TrainingExample>> groups,
      int epochs,
      double learningRate,
      double l2) {
    List<LogisticCandidateScorer.TrainingExample> all = new ArrayList<>();
    for (List<LogisticCandidateScorer.TrainingExample> group : groups) all.addAll(group);
    if (all.isEmpty()) throw new IllegalArgumentException("No pairwise training examples");
    int d = CandidateFeatures.NAMES.length;
    double[] mean = new double[d];
    for (var example : all) for (int j = 0; j < d; j++) mean[j] += example.features()[j];
    for (int j = 0; j < d; j++) mean[j] /= all.size();
    double[] scale = new double[d];
    for (var example : all)
      for (int j = 0; j < d; j++) {
        double z = example.features()[j] - mean[j];
        scale[j] += z * z;
      }
    for (int j = 0; j < d; j++) {
      scale[j] = Math.sqrt(scale[j] / all.size());
      if (!Double.isFinite(scale[j]) || scale[j] < 1e-9) scale[j] = 1.0;
    }
    List<double[]> differences = new ArrayList<>();
    for (List<LogisticCandidateScorer.TrainingExample> group : groups) {
      for (var positive : group)
        if (positive.label() == 1) {
          for (var negative : group)
            if (negative.label() == 0) {
              double[] difference = new double[d];
              for (int j = 0; j < d; j++)
                difference[j] =
                    standardized(positive.features(), j, mean, scale)
                        - standardized(negative.features(), j, mean, scale);
              differences.add(difference);
            }
        }
    }
    if (differences.isEmpty())
      throw new IllegalArgumentException("No positive-negative query pairs");
    double[] weights = new double[d];
    for (int epoch = 0; epoch < epochs; epoch++)
      for (double[] difference : differences) {
        double probability = sigmoid(dot(weights, difference));
        double error = probability - 1.0;
        for (int j = 0; j < d; j++)
          weights[j] -= learningRate * (error * difference[j] + l2 * weights[j]);
      }
    return new PairwiseCandidateScorer(mean, scale, weights);
  }

  @Override
  public double probability(double[] features) {
    return sigmoid(dot(weights, standardize(features)));
  }

  @Override
  public String modelType() {
    return "pairwise-logistic-ranking";
  }

  public void save(Path file) throws IOException {
    try (BufferedWriter writer = Files.newBufferedWriter(file)) {
      writer.write("model=pairwise-logistic-ranking\n");
      writer.write(
          "objective=positive candidate outranks negative candidate from same query seed\n");
      writer.write("features=" + String.join(",", CandidateFeatures.NAMES) + "\n");
      writer.write(
          "mean=" + join(mean) + "\nscale=" + join(scale) + "\nweights=" + join(weights) + "\n");
    }
  }

  private double[] standardize(double[] features) {
    double[] result = new double[features.length];
    for (int j = 0; j < features.length; j++) result[j] = standardized(features, j, mean, scale);
    return result;
  }

  private static double standardized(double[] features, int j, double[] mean, double[] scale) {
    return (features[j] - mean[j]) / scale[j];
  }

  private static double dot(double[] left, double[] right) {
    double result = 0;
    for (int j = 0; j < left.length; j++) result += left[j] * right[j];
    return result;
  }

  private static double sigmoid(double value) {
    if (value >= 0) {
      double z = Math.exp(-value);
      return 1 / (1 + z);
    }
    double z = Math.exp(value);
    return z / (1 + z);
  }

  private static String join(double[] values) {
    StringJoiner result = new StringJoiner(",");
    for (double value : values) result.add(Double.toString(value));
    return result.toString();
  }
}
