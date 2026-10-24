package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Runs all seeds for each generated LFR graph and writes a summary CSV. */
public final class LfrExperimentRunner {
  public static void main(String[] a) throws Exception {
    if (a.length < 2) {
      System.err.println(
          "Usage: java foldedsubgraph.LfrExperimentRunner lfr-directory output.csv"
              + " [limit-per-dataset]");
      return;
    }
    int limit = a.length > 2 ? Integer.parseInt(a[2]) : Integer.MAX_VALUE;
    try (BufferedWriter w = Files.newBufferedWriter(Path.of(a[1]))) {
      w.write("dataset,n,edges,recall,precision,f1,seeds\n");
      for (LfrSpec s : LfrSpec.paperTable()) {
        Graph g = DatasetReader.edgeList(Path.of(a[0], s.name() + ".edgelist"), false);
        List<Set<Integer>> t = DatasetReader.communities(Path.of(a[0], s.name() + ".truth"), false);
        int n = Math.min(limit, g.size());
        double r = 0, p = 0, f = 0;
        for (int seed = 0; seed < n; seed++) {
          Metrics.Score z =
              Metrics.score(new FGDetector(g).detect(seed, .45, 1.0), Metrics.truthFor(seed, t));
          r += z.recall();
          p += z.precision();
          f += z.f1();
        }
        w.write(
            String.format(
                Locale.ROOT,
                "%s,%d,%d,%.8f,%.8f,%.8f,%d%n",
                s.name(),
                g.size(),
                g.edgeCount(),
                r / n,
                p / n,
                f / n,
                n));
        System.out.println(s.name() + " completed seeds=" + n);
      }
    }
  }
}
