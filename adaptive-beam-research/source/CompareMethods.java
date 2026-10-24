package foldedsubgraph;

import java.nio.file.*;
import java.util.*;

/** Compares FG-based with the paper's M-based baseline over every seed. */
public final class CompareMethods {
  public static void main(String[] a) throws Exception {
    if (a.length < 3) {
      System.err.println("Usage: java foldedsubgraph.CompareMethods graph truth oneBased gamma h");
      return;
    }
    boolean one = Boolean.parseBoolean(a[2]);
    double gamma = Double.parseDouble(a[3]), h = Double.parseDouble(a[4]);
    Graph g = DatasetReader.edgeList(Path.of(a[0]), one);
    List<Set<Integer>> t = DatasetReader.communities(Path.of(a[1]), one);
    double[] fg = {0, 0, 0}, m = {0, 0, 0};
    for (int i = 0; i < g.size(); i++) {
      Set<Integer> truth = Metrics.truthFor(i, t);
      Metrics.Score x = Metrics.score(new FGDetector(g).detect(i, gamma, h), truth),
          y = Metrics.score(new MBaseline(g).detect(i), truth);
      fg[0] += x.recall();
      fg[1] += x.precision();
      fg[2] += x.f1();
      m[0] += y.recall();
      m[1] += y.precision();
      m[2] += y.f1();
    }
    for (int j = 0; j < 3; j++) {
      fg[j] /= g.size();
      m[j] /= g.size();
    }
    System.out.printf(
        Locale.ROOT,
        "FG recall=%.6f precision=%.6f f1=%.6f%nM  recall=%.6f precision=%.6f f1=%.6f%n",
        fg[0],
        fg[1],
        fg[2],
        m[0],
        m[1],
        m[2]);
  }
}
