package foldedsubgraph;

import java.nio.file.*;
import java.util.*;

/** Grid-searches gamma and h using the paper's average per-seed F-score. */
public final class ParameterSweep {
  public static void main(String[] a) throws Exception {
    if (a.length < 2) {
      System.err.println("Usage: java foldedsubgraph.ParameterSweep graph truth [oneBased]");
      return;
    }
    boolean one = a.length > 2 && Boolean.parseBoolean(a[2]);
    Graph g = DatasetReader.edgeList(Path.of(a[0]), one);
    List<Set<Integer>> truth = DatasetReader.communities(Path.of(a[1]), one);
    Metrics.Score best = null;
    double bg = 0, bh = 0;
    for (int gi = 5; gi <= 19; gi++) {
      double gamma = gi / 20.0;
      for (int hi = 1; hi <= 15; hi++) {
        double h = hi / 10.0, sr = 0, sp = 0, sf = 0;
        for (int seed = 0; seed < g.size(); seed++) {
          Metrics.Score s =
              Metrics.score(
                  new FGDetector(g).detect(seed, gamma, h), Metrics.truthFor(seed, truth));
          sr += s.recall();
          sp += s.precision();
          sf += s.f1();
        }
        Metrics.Score avg = new Metrics.Score(sr / g.size(), sp / g.size(), sf / g.size());
        if (best == null || avg.f1() > best.f1()) {
          best = avg;
          bg = gamma;
          bh = h;
          System.out.printf(
              Locale.ROOT,
              "best gamma=%.2f h=%.1f recall=%.6f precision=%.6f f1=%.6f%n",
              bg,
              bh,
              best.recall(),
              best.precision(),
              best.f1());
        }
      }
    }
  }
}
