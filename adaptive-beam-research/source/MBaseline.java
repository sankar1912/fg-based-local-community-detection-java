package foldedsubgraph;

import java.util.*;

/** Simple M-based baseline from the paper: greedy single-node local modularity expansion. */
public final class MBaseline {
  private final Graph g;

  public MBaseline(Graph g) {
    this.g = g;
  }

  public Set<Integer> detect(int seed) {
    Set<Integer> c = new LinkedHashSet<>();
    c.add(seed);
    while (true) {
      Set<Integer> a = alternative(c), best = null;
      double gain = 0;
      for (int v : a) {
        Set<Integer> f = Set.of(v);
        double d = delta(c, a, f);
        if (d > gain) {
          gain = d;
          best = f;
        }
      }
      if (best == null) break;
      c.addAll(best);
    }
    return c;
  }

  private double delta(Set<Integer> c, Set<Integer> a, Set<Integer> f) {
    int ein = internal(c), eout = external(c);
    int v = f.iterator().next(), x = 0;
    for (int n : g.neighbors(v)) if (c.contains(n)) x++;
    int y = x, z = g.degree(v) - x;
    double old = eout == 0 ? 0 : ein / (double) eout, den = eout + z - y;
    return den <= 0 ? Double.NEGATIVE_INFINITY : (ein + x) / den - old;
  }

  private int internal(Set<Integer> c) {
    int e = 0;
    for (int u : c) for (int v : g.neighbors(u)) if (u < v && c.contains(v)) e++;
    return e;
  }

  private int external(Set<Integer> c) {
    int e = 0;
    for (int u : c) for (int v : g.neighbors(u)) if (!c.contains(v)) e++;
    return e;
  }

  private Set<Integer> alternative(Set<Integer> c) {
    Set<Integer> a = new LinkedHashSet<>();
    for (int u : c) for (int v : g.neighbors(u)) if (!c.contains(v)) a.add(v);
    return a;
  }
}
