package foldedsubgraph;

import java.util.*;

/** Algorithm 3 and equations (1)-(5). */
public final class ModularityExpansion {
  private final Graph g;
  private final Folding folding;
  private final Map<FoldKey, Set<Integer>> foldedCache = new HashMap<>();

  public ModularityExpansion(Graph g) {
    this.g = g;
    folding = new Folding(g);
  }

  public Set<Integer> expand(Set<Integer> initial, double gamma) {
    Set<Integer> c = new LinkedHashSet<>(initial);
    while (true) {
      Set<Integer> a = alternative(c);
      if (a.isEmpty()) return c;
      double best = Double.NEGATIVE_INFINITY;
      Set<Integer> bf = null;
      int ein = internal(c), eout = external(c);
      for (int v : a) {
        Set<Integer> f = folded(v, gamma);
        if (f.stream().noneMatch(x -> !c.contains(x))) continue;
        double d = delta(c, a, f, ein, eout);
        if (d > best) {
          best = d;
          bf = f;
        }
      }
      if (bf == null || best < 0) return c;
      c.addAll(bf);
    }
  }

  public double delta(Set<Integer> c, Set<Integer> a, Set<Integer> f) {
    return delta(c, a, f, internal(c), external(c));
  }

  double delta(Set<Integer> c, Set<Integer> a, Set<Integer> f, int ein, int eout) {
    double x = 0, y = 0, degree = 0, internalF = 0;
    for (int i : f) {
      if (!c.contains(i)) {
        degree += g.degree(i);
        for (int n : g.neighbors(i)) {
          if (c.contains(n)) x++;
          if (f.contains(n) && !c.contains(n)) internalF++;
        }
      }
      if (a.contains(i)) for (int n : g.neighbors(i)) if (c.contains(n)) y++;
    }
    double z = degree - x - .5 * internalF;
    double old = eout == 0 ? 0 : ein / (double) eout;
    double den = eout + z - y;
    if (den <= 0) return Double.NEGATIVE_INFINITY;
    return (ein + x) / den - old;
  }

  int internal(Set<Integer> c) {
    int e = 0;
    for (int u : c) for (int v : g.neighbors(u)) if (u < v && c.contains(v)) e++;
    return e;
  }

  int external(Set<Integer> c) {
    int e = 0;
    for (int u : c) for (int v : g.neighbors(u)) if (!c.contains(v)) e++;
    return e;
  }

  private Set<Integer> alternative(Set<Integer> c) {
    Set<Integer> a = new LinkedHashSet<>();
    for (int u : c) for (int v : g.neighbors(u)) if (!c.contains(v)) a.add(v);
    return a;
  }

  private Set<Integer> folded(int start, double gamma) {
    return foldedCache.computeIfAbsent(new FoldKey(start, gamma), key -> folding.build(start, gamma));
  }

  private record FoldKey(int start, double gamma) {}
}
