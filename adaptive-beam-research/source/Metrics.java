package foldedsubgraph;

import java.util.*;

public final class Metrics {
  private Metrics() {}

  public record Score(double recall, double precision, double f1) {}

  public static Score score(Set<Integer> found, Set<Integer> trueC) {
    Set<Integer> hit = new HashSet<>(found);
    hit.retainAll(trueC);
    double r = trueC.isEmpty() ? 0 : hit.size() / (double) trueC.size(),
        p = found.isEmpty() ? 0 : hit.size() / (double) found.size();
    return new Score(r, p, (r + p) == 0 ? 0 : 2 * r * p / (r + p));
  }

  public static Set<Integer> truthFor(int node, List<Set<Integer>> cs) {
    for (Set<Integer> c : cs) if (c.contains(node)) return c;
    return Set.of(node);
  }
}
