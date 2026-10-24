package foldedsubgraph;

import java.util.*;

/** Training-only standardization shared by all learned candidate models. */
public final class FeatureScaling {
  private final double[] mean;
  private final double[] scale;

  private FeatureScaling(double[] mean, double[] scale) {
    this.mean = mean;
    this.scale = scale;
  }

  public static FeatureScaling fit(List<LogisticCandidateScorer.TrainingExample> examples) {
    int d = CandidateFeatures.NAMES.length;
    double[] mean = new double[d];
    for (var example : examples) for (int j = 0; j < d; j++) mean[j] += example.features()[j];
    for (int j = 0; j < d; j++) mean[j] /= examples.size();
    double[] scale = new double[d];
    for (var example : examples)
      for (int j = 0; j < d; j++) {
        double z = example.features()[j] - mean[j];
        scale[j] += z * z;
      }
    for (int j = 0; j < d; j++) {
      scale[j] = Math.sqrt(scale[j] / examples.size());
      if (scale[j] < 1e-9 || !Double.isFinite(scale[j])) scale[j] = 1.0;
    }
    return new FeatureScaling(mean, scale);
  }

  public double[] apply(double[] input) {
    double[] result = new double[input.length];
    for (int j = 0; j < input.length; j++) result[j] = (input[j] - mean[j]) / scale[j];
    return result;
  }

  public String describe() {
    return "mean=" + join(mean) + System.lineSeparator() + "scale=" + join(scale);
  }

  private static String join(double[] values) {
    StringJoiner result = new StringJoiner(",");
    for (double value : values) result.add(Double.toString(value));
    return result.toString();
  }
}
