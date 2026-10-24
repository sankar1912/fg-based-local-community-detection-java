package foldedsubgraph;

import java.nio.file.*;
import java.util.*;

/** CLI: java foldedsubgraph.Main <edge-list> <truth-file> <seed> [gamma] [h]. */
public final class Main {
  public static void main(String[] args) throws Exception {
    if (args.length < 3) {
      System.err.println(
          "Usage: java foldedsubgraph.Main graph.edgelist truth.txt seed [gamma=0.7] [h=0.5]"
              + " [oneBased=false]");
      return;
    }
    boolean one = args.length > 5 && Boolean.parseBoolean(args[5]);
    Graph g = DatasetReader.edgeList(Path.of(args[0]), one);
    List<Set<Integer>> truth = DatasetReader.communities(Path.of(args[1]), one);
    int inputSeed = Integer.parseInt(args[2]), seed = inputSeed - (one ? 1 : 0);
    double ga = args.length > 3 ? Double.parseDouble(args[3]) : .7,
        h = args.length > 4 ? Double.parseDouble(args[4]) : .5;
    Set<Integer> found = new FGDetector(g).detect(seed, ga, h);
    Metrics.Score s = Metrics.score(found, Metrics.truthFor(seed, truth));
    System.out.printf(
        Locale.ROOT,
        "nodes=%d edges=%d seed=%d gamma=%.4f h=%.4f community=%s%nrecall=%.6f precision=%.6f"
            + " f1=%.6f%n",
        g.size(),
        g.edgeCount(),
        inputSeed,
        ga,
        h,
        found,
        s.recall(),
        s.precision(),
        s.f1());
  }
}
