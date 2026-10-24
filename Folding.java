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
    Map<Integer, Integer> connectionCounts = new HashMap<>();
    for (int candidate : r) connectionCounts.put(candidate, 1);
    int internalEdges = 0;
    while (!r.isEmpty()) {
      int w =
          r.stream()
              .min(
                  Comparator.<Integer>comparingInt(v -> -connectionCounts.getOrDefault(v, 0))
                      .thenComparingInt(graph::degree)
                      .thenComparingInt(v -> v))
              .orElseThrow();
      int candidateInternalEdges = internalEdges + connectionCounts.getOrDefault(w, 0);
      int candidateSize = f.size() + 1;
      double candidateDensity =
          candidateSize < 2
              ? 1.
              : candidateInternalEdges / (double) (candidateSize * (candidateSize - 1) / 2);
      if (candidateDensity < gamma) break;
      f.add(w);
      internalEdges = candidateInternalEdges;
      r.remove(w);
      for (int u : graph.neighbors(w)) {
        if (!f.contains(u)) {
          r.add(u);
          connectionCounts.merge(u, 1, Integer::sum);
        }
      }
    }
    return f;
  }

  public double density(Set<Integer> s) {
    if (s.size() < 2) return 1.;
    int e = 0;
    for (int u : s) for (int v : graph.neighbors(u)) if (u < v && s.contains(v)) e++;
    return e / (s.size() * (s.size() - 1) / 2.);
  }
}
