package foldedsubgraph;

import java.nio.file.*;
import java.util.*;

/** Batch CLI: graph truth output gamma h [oneBased]. */
public final class RunExperiments {
  public static void main(String[] a) throws Exception {
    if (a.length < 5) {
      System.err.println(
          "Usage: java foldedsubgraph.RunExperiments graph.edgelist truth.txt output.csv gamma h"
              + " [oneBased]");
      return;
    }
    boolean one = a.length > 5 && Boolean.parseBoolean(a[5]);
    Graph g = DatasetReader.edgeList(Path.of(a[0]), one);
    List<Set<Integer>> truth = DatasetReader.communities(Path.of(a[1]), one);
    ExperimentRunner.run(
        g, truth, Double.parseDouble(a[3]), Double.parseDouble(a[4]), Path.of(a[2]));
  }
}
