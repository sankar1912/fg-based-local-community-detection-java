package foldedsubgraph;

import java.util.*;

/** One-hidden-layer multilayer perceptron over the shared candidate features. */
public final class MlpCandidateScorer implements CandidateScorer {
  private final FeatureScaling scaling;
  private final double[][] inputHidden;
  private final double[] hiddenBias;
  private final double[] outputHidden;
  private final double outputBias;

  private MlpCandidateScorer(
      FeatureScaling scaling,
      double[][] inputHidden,
      double[] hiddenBias,
      double[] outputHidden,
      double outputBias) {
    this.scaling = scaling;
    this.inputHidden = inputHidden;
    this.hiddenBias = hiddenBias;
    this.outputHidden = outputHidden;
    this.outputBias = outputBias;
  }

  public static MlpCandidateScorer train(
      List<LogisticCandidateScorer.TrainingExample> examples,
      int hidden,
      int epochs,
      double learningRate,
      double l2,
      long seed) {
    FeatureScaling scaling = FeatureScaling.fit(examples);
    int d = CandidateFeatures.NAMES.length;
    Random random = new Random(seed);
    double[][] w = new double[d][hidden];
    double[] hb = new double[hidden];
    double[] ow = new double[hidden];
    for (int i = 0; i < d; i++)
      for (int j = 0; j < hidden; j++) w[i][j] = random.nextGaussian() * .05;
    for (int j = 0; j < hidden; j++) ow[j] = random.nextGaussian() * .05;
    double ob = 0;
    for (int epoch = 0; epoch < epochs; epoch++) {
      for (var example : examples) {
        double[] x = scaling.apply(example.features());
        double[] h = new double[hidden];
        for (int j = 0; j < hidden; j++) {
          double z = hb[j];
          for (int i = 0; i < d; i++) z += x[i] * w[i][j];
          h[j] = sigmoid(z);
        }
        double p = sigmoid(ob + dot(ow, h));
        double error = p - example.label();
        for (int j = 0; j < hidden; j++) {
          double hiddenError = error * ow[j] * h[j] * (1 - h[j]);
          ow[j] -= learningRate * (error * h[j] + l2 * ow[j]);
          hb[j] -= learningRate * hiddenError;
          for (int i = 0; i < d; i++) w[i][j] -= learningRate * (hiddenError * x[i] + l2 * w[i][j]);
        }
        ob -= learningRate * error;
      }
    }
    return new MlpCandidateScorer(scaling, w, hb, ow, ob);
  }

  @Override
  public double probability(double[] features) {
    double[] x = scaling.apply(features);
    double[] h = new double[hiddenBias.length];
    for (int j = 0; j < h.length; j++) {
      double z = hiddenBias[j];
      for (int i = 0; i < x.length; i++) z += x[i] * inputHidden[i][j];
      h[j] = sigmoid(z);
    }
    return sigmoid(outputBias + dot(outputHidden, h));
  }

  @Override
  public String modelType() {
    return "mlp-1-hidden-layer";
  }

  private static double dot(double[] a, double[] b) {
    double result = 0;
    for (int i = 0; i < a.length; i++) result += a[i] * b[i];
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
