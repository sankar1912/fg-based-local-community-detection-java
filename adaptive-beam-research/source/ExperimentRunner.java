package foldedsubgraph;

import java.nio.file.*;
import java.util.*;

/** Runs Algorithm 5 for every node and writes paper-style averaged metrics. */
public final class ExperimentRunner {
  private ExperimentRunner() {}

  public static void run(Graph g, List<Set<Integer>> truth, double gamma, double h, Path csv)
      throws Exception {
    double sr = 0, sp = 0, sf = 0;
    try (var w = Files.newBufferedWriter(csv)) {
      w.write("seed,found_size,recall,precision,f1\n");
      for (int seed = 0; seed < g.size(); seed++) {
        Set<Integer> found = new FGDetector(g).detect(seed, gamma, h);
        Metrics.Score s = Metrics.score(found, Metrics.truthFor(seed, truth));
        sr += s.recall();
        sp += s.precision();
        sf += s.f1();
        w.write(
            String.format(
                Locale.ROOT,
                "%d,%d,%.12f,%.12f,%.12f%n",
                seed,
                found.size(),
                s.recall(),
                s.precision(),
                s.f1()));
      }
      int n = g.size();
      w.write(String.format(Locale.ROOT, "average,,%.12f,%.12f,%.12f%n", sr / n, sp / n, sf / n));
    }
  }
}
