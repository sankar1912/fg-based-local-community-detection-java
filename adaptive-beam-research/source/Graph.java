package foldedsubgraph;

import java.util.*;

/** Immutable-size undirected simple graph used by the paper algorithms. */
public final class Graph {
  private final int n;
  private final List<Set<Integer>> neighbors;

  public Graph(int n) {
    if (n < 1) throw new IllegalArgumentException("n must be positive");
    this.n = n;
    neighbors = new ArrayList<>(n);
    for (int i = 0; i < n; i++) neighbors.add(new LinkedHashSet<>());
  }

  public int size() {
    return n;
  }

  public void addEdge(int a, int b) {
    check(a);
    check(b);
    if (a != b) {
      neighbors.get(a).add(b);
      neighbors.get(b).add(a);
    }
  }

  public Set<Integer> neighbors(int v) {
    check(v);
    return Collections.unmodifiableSet(neighbors.get(v));
  }

  public int degree(int v) {
    return neighbors(v).size();
  }

  public boolean hasEdge(int a, int b) {
    check(a);
    check(b);
    return neighbors.get(a).contains(b);
  }

  public int edgeCount() {
    int e = 0;
    for (Set<Integer> s : neighbors) e += s.size();
    return e / 2;
  }

  private void check(int v) {
    if (v < 0 || v >= n) throw new IndexOutOfBoundsException("node " + v);
  }
}
