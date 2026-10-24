package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Deterministic shallow gradient-boosted decision-stump scorer for candidate compatibility. */
public final class GradientBoostedCandidateScorer implements CandidateScorer {
  private final double prior;
  private final List<Stump> stumps;

  private GradientBoostedCandidateScorer(double prior, List<Stump> stumps) {
    this.prior = prior;
    this.stumps = List.copyOf(stumps);
  }

  public static GradientBoostedCandidateScorer train(
      List<LogisticCandidateScorer.TrainingExample> examples,
      int rounds,
      double learningRate,
      double l2) {
    if (examples.isEmpty()) throw new IllegalArgumentException("No training examples");
    int d = CandidateFeatures.NAMES.length;
    double positive =
        examples.stream().mapToDouble(LogisticCandidateScorer.TrainingExample::label).sum();
    double mean = Math.max(1e-4, Math.min(.9999, positive / examples.size()));
    double prior = Math.log(mean / (1 - mean));
    List<Stump> stumps = new ArrayList<>();
    double[] prediction = new double[examples.size()];
    Arrays.fill(prediction, prior);
    for (int round = 0; round < rounds; round++) {
      double[] residual = new double[examples.size()];
      for (int i = 0; i < examples.size(); i++)
        residual[i] = examples.get(i).label() - sigmoid(prediction[i]);
      Stump best = null;
      double bestLoss = Double.POSITIVE_INFINITY;
      for (int feature = 0; feature < d; feature++) {
        final int selectedFeature = feature;
        double[] values =
            examples.stream().mapToDouble(e -> e.features()[selectedFeature]).sorted().toArray();
        for (int q = 1; q < 16; q++) {
          double threshold = values[(int) ((values.length - 1L) * q / 16.0)];
          double leftSum = 0, rightSum = 0;
          int leftCount = 0, rightCount = 0;
          for (int i = 0; i < examples.size(); i++) {
            if (examples.get(i).features()[feature] <= threshold) {
              leftSum += residual[i];
              leftCount++;
            } else {
              rightSum += residual[i];
              rightCount++;
            }
          }
          if (leftCount == 0 || rightCount == 0) continue;
          double left = learningRate * leftSum / (leftCount + l2);
          double right = learningRate * rightSum / (rightCount + l2);
          double loss = 0;
          for (int i = 0; i < examples.size(); i++) {
            double value = examples.get(i).features()[feature] <= threshold ? left : right;
            double error = residual[i] - value;
            loss += error * error;
          }
          if (loss < bestLoss) {
            bestLoss = loss;
            best = new Stump(feature, threshold, left, right);
          }
        }
      }
      if (best == null) break;
      stumps.add(best);
      for (int i = 0; i < examples.size(); i++)
        prediction[i] += best.value(examples.get(i).features());
    }
    return new GradientBoostedCandidateScorer(prior, stumps);
  }

  @Override
  public double probability(double[] features) {
    double score = prior;
    for (Stump stump : stumps) score += stump.value(features);
    return sigmoid(score);
  }

  @Override
  public String modelType() {
    return "gradient-boosted-stumps";
  }

  public void save(Path file) throws IOException {
    try (BufferedWriter writer = Files.newBufferedWriter(file)) {
      writer.write("model=gradient-boosted-decision-stumps\n");
      writer.write("features=" + String.join(",", CandidateFeatures.NAMES) + "\n");
      writer.write("prior_logit=" + prior + "\nrounds=" + stumps.size() + "\n");
      for (Stump stump : stumps)
        writer.write(
            stump.feature + "," + stump.threshold + "," + stump.left + "," + stump.right + "\n");
    }
  }

  private static double sigmoid(double value) {
    if (value >= 0) {
      double z = Math.exp(-value);
      return 1 / (1 + z);
    }
    double z = Math.exp(value);
    return z / (1 + z);
  }

  private record Stump(int feature, double threshold, double left, double right) {
    double value(double[] features) {
      return features[feature] <= threshold ? left : right;
    }
  }
}
