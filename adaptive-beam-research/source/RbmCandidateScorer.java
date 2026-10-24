package foldedsubgraph;

import java.util.*;

/**
 * Genuine Bernoulli-Bernoulli RBM. Standardized features are thresholded into visible bits and a
 * final visible bit is the compatibility label. Training uses one-step contrastive divergence.
 */
public final class RbmCandidateScorer implements CandidateScorer {
  private final FeatureScaling scaling;
  private final double[][] weights;
  private final double[] visibleBias;
  private final double[] hiddenBias;
  private final int featureCount;

  private RbmCandidateScorer(
      FeatureScaling scaling, double[][] weights, double[] visibleBias, double[] hiddenBias) {
    this.scaling = scaling;
    this.weights = weights;
    this.visibleBias = visibleBias;
    this.hiddenBias = hiddenBias;
    featureCount = CandidateFeatures.NAMES.length;
  }

  public static RbmCandidateScorer train(
      List<LogisticCandidateScorer.TrainingExample> examples,
      int hidden,
      int epochs,
      double learningRate,
      double weightDecay,
      long seed) {
    FeatureScaling scaling = FeatureScaling.fit(examples);
    int visible = CandidateFeatures.NAMES.length + 1;
    Random random = new Random(seed);
    double[][] w = new double[visible][hidden];
    double[] vb = new double[visible];
    double[] hb = new double[hidden];
    for (int i = 0; i < visible; i++)
      for (int j = 0; j < hidden; j++) w[i][j] = random.nextGaussian() * .02;
    for (int epoch = 0; epoch < epochs; epoch++) {
      for (var example : examples) {
        int[] v0 = visible(example, scaling);
        double[] h0 = hiddenProb(v0, w, hb);
        int[] hSample = sample(h0, random);
        int[] v1 = visibleSample(hSample, w, vb, random);
        double[] h1 = hiddenProb(v1, w, hb);
        for (int i = 0; i < visible; i++) {
          vb[i] += learningRate * (v0[i] - v1[i]);
          for (int j = 0; j < hidden; j++) {
            w[i][j] += learningRate * (v0[i] * h0[j] - v1[i] * h1[j] - weightDecay * w[i][j]);
          }
        }
        for (int j = 0; j < hidden; j++) hb[j] += learningRate * (h0[j] - h1[j]);
      }
    }
    return new RbmCandidateScorer(scaling, w, vb, hb);
  }

  @Override
  public double probability(double[] features) {
    int[] v = visibleFeatures(features, scaling);
    double labelFreeEnergy0 = freeEnergy(v, 0);
    double labelFreeEnergy1 = freeEnergy(v, 1);
    double difference = labelFreeEnergy1 - labelFreeEnergy0;
    if (difference >= 0) {
      double z = Math.exp(-difference);
      return 1 / (1 + z);
    }
    double z = Math.exp(difference);
    return z / (1 + z);
  }

  @Override
  public String modelType() {
    return "rbm-bernoulli-cd1";
  }

  private double freeEnergy(int[] features, int label) {
    double value = -visibleBias[featureCount] * label;
    for (int i = 0; i < featureCount; i++) value -= visibleBias[i] * features[i];
    for (int j = 0; j < hiddenBias.length; j++) {
      double z = hiddenBias[j] + weights[featureCount][j] * label;
      for (int i = 0; i < featureCount; i++) z += weights[i][j] * features[i];
      value -= Math.log1p(Math.exp(Math.min(50, z)));
    }
    return value;
  }

  private static int[] visible(LogisticCandidateScorer.TrainingExample e, FeatureScaling scaling) {
    double[] x = scaling.apply(e.features());
    int[] v = new int[x.length + 1];
    for (int i = 0; i < x.length; i++) v[i] = x[i] >= 0 ? 1 : 0;
    v[x.length] = e.label();
    return v;
  }

  private static int[] visibleFeatures(double[] features, FeatureScaling scaling) {
    double[] x = scaling.apply(features);
    int[] v = new int[x.length + 1];
    for (int i = 0; i < x.length; i++) v[i] = x[i] >= 0 ? 1 : 0;
    return v;
  }

  private static double[] hiddenProb(int[] v, double[][] w, double[] b) {
    double[] result = new double[b.length];
    for (int j = 0; j < result.length; j++) {
      double z = b[j];
      for (int i = 0; i < v.length; i++) z += v[i] * w[i][j];
      result[j] = sigmoid(z);
    }
    return result;
  }

  private static int[] visibleSample(int[] h, double[][] w, double[] b, Random r) {
    int[] result = new int[b.length];
    for (int i = 0; i < result.length; i++) {
      double z = b[i];
      for (int j = 0; j < h.length; j++) z += h[j] * w[i][j];
      result[i] = r.nextDouble() < sigmoid(z) ? 1 : 0;
    }
    return result;
  }

  private static int[] sample(double[] p, Random r) {
    int[] result = new int[p.length];
    for (int i = 0; i < p.length; i++) result[i] = r.nextDouble() < p[i] ? 1 : 0;
    return result;
  }

  private static double sigmoid(double x) {
    if (x >= 0) {
      double z = Math.exp(-x);
      return 1 / (1 + z);
    }
    double z = Math.exp(x);
    return z / (1 + z);
  }
}
