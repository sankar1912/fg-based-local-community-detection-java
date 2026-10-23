package foldedsubgraph;

import java.util.*;

/** Algorithm 1: greedy folded-subgraph construction. */
public final class Folding {
  private final Graph graph;

  public Folding(Graph g) {
    graph = g;
  }

  public Set<Integer> build(int start, double gamma) {
    if (gamma < 0 || gamma > 1) throw new IllegalArgumentException("gamma must be in [0,1]");
    Set<Integer> f = new LinkedHashSet<>();
    f.add(start);
    Set<Integer> r = new LinkedHashSet<>(graph.neighbors(start));
    while (!r.isEmpty()) {
      int w =
          r.stream()
              .min(
                  Comparator.<Integer>comparingInt(v -> -connections(v, f))
                      .thenComparingInt(graph::degree)
                      .thenComparingInt(v -> v))
              .orElseThrow();
      Set<Integer> candidate = new LinkedHashSet<>(f);
      candidate.add(w);
      if (density(candidate) < gamma) break;
      f.add(w);
      r.remove(w);
      for (int u : graph.neighbors(w)) if (!f.contains(u)) r.add(u);
    }
    return f;
  }

  private int connections(int v, Set<Integer> f) {
    int n = 0;
    for (int u : graph.neighbors(v)) if (f.contains(u)) n++;
    return n;
  }

  public double density(Set<Integer> s) {
    if (s.size() < 2) return 1.;
    int e = 0;
    for (int u : s) for (int v : graph.neighbors(u)) if (u < v && s.contains(v)) e++;
    return e / (s.size() * (s.size() - 1) / 2.);
  }
}
