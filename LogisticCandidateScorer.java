package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/** A calibrated logistic compatibility model over local candidate features. */
public final class LogisticCandidateScorer {
  private final double[] mean;
  private final double[] scale;
  private final double[] weights;
  private final double bias;

  private LogisticCandidateScorer(double[] mean, double[] scale, double[] weights, double bias) {
    this.mean = mean;
    this.scale = scale;
    this.weights = weights;
    this.bias = bias;
  }

  public static LogisticCandidateScorer train(
      List<TrainingExample> examples, int epochs, double learningRate, double l2) {
    if (examples.isEmpty()) throw new IllegalArgumentException("No training examples");
    int d = CandidateFeatures.NAMES.length;
    double[] mean = new double[d];
    for (TrainingExample example : examples) {
      for (int j = 0; j < d; j++) mean[j] += example.features()[j];
    }
    for (int j = 0; j < d; j++) mean[j] /= examples.size();
    double[] scale = new double[d];
    for (TrainingExample example : examples) {
      for (int j = 0; j < d; j++) {
        double z = example.features()[j] - mean[j];
        scale[j] += z * z;
      }
    }
    for (int j = 0; j < d; j++) {
      scale[j] = Math.sqrt(scale[j] / examples.size());
      if (scale[j] < 1e-9) scale[j] = 1.0;
    }

    double[] weights = new double[d];
    double bias = 0.0;
    for (int epoch = 0; epoch < epochs; epoch++) {
      double[] gradient = new double[d];
      double biasGradient = 0.0;
      for (TrainingExample example : examples) {
        double[] x = standardize(example.features(), mean, scale);
        double probability = sigmoid(dot(weights, x) + bias);
        double error = probability - example.label();
        for (int j = 0; j < d; j++) gradient[j] += error * x[j];
        biasGradient += error;
      }
      double n = examples.size();
      for (int j = 0; j < d; j++) {
        gradient[j] = gradient[j] / n + l2 * weights[j];
        weights[j] -= learningRate * gradient[j];
      }
      bias -= learningRate * biasGradient / n;
    }
    return new LogisticCandidateScorer(mean, scale, weights, bias);
  }

  public double probability(double[] features) {
    return sigmoid(dot(weights, standardize(features, mean, scale)) + bias);
  }

  public void save(Path file) throws IOException {
    try (BufferedWriter writer = Files.newBufferedWriter(file)) {
      writer.write("model=logistic-candidate-compatibility\n");
      writer.write("features=" + String.join(",", CandidateFeatures.NAMES) + "\n");
      writer.write("mean=" + join(mean) + "\n");
      writer.write("scale=" + join(scale) + "\n");
      writer.write("weights=" + join(weights) + "\n");
      writer.write("bias=" + bias + "\n");
    }
  }

  private static double[] standardize(double[] x, double[] mean, double[] scale) {
    double[] result = new double[x.length];
    for (int j = 0; j < x.length; j++) result[j] = (x[j] - mean[j]) / scale[j];
    return result;
  }

  private static double dot(double[] left, double[] right) {
    double result = 0.0;
    for (int j = 0; j < left.length; j++) result += left[j] * right[j];
    return result;
  }

  private static double sigmoid(double value) {
    if (value >= 0) {
      double z = Math.exp(-value);
      return 1.0 / (1.0 + z);
    }
    double z = Math.exp(value);
    return z / (1.0 + z);
  }

  private static String join(double[] values) {
    StringJoiner joiner = new StringJoiner(",");
    for (double value : values) joiner.add(Double.toString(value));
    return joiner.toString();
  }

  public record TrainingExample(double[] features, int label) {}
}
