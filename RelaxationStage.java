package foldedsubgraph;

import java.util.*;

/** Algorithm 4; beta is fixed to 0.5 as reported in the paper. */
public final class RelaxationStage {
  private final Graph g;
  private final Folding folding;
  private final double beta;
  private final Map<FoldKey, Set<Integer>> foldedCache = new HashMap<>();

  public RelaxationStage(Graph g) {
    this(g, .5);
  }

  public RelaxationStage(Graph g, double b) {
    this.g = g;
    beta = b;
    folding = new Folding(g);
  }

  public Set<Integer> expand(Set<Integer> initial, double gamma, double h) {
    Set<Integer> c = new LinkedHashSet<>(initial), a = alternative(c), bf = null;
    double best = Double.NEGATIVE_INFINITY;
    for (int v : a) {
      Set<Integer> f = folded(v, gamma);
      double s = f1(f, c) + beta * f2(f, c);
      if (s > best) {
        best = s;
        bf = f;
      }
    }
    if (bf != null && best >= h) c.addAll(bf);
    return c;
  }

  public double f1(Set<Integer> f, Set<Integer> c) {
    Set<Integer> hit = new HashSet<>();
    for (int i : f) for (int n : g.neighbors(i)) if (c.contains(n)) hit.add(n);
    return c.isEmpty() ? 0 : hit.size() / (double) c.size();
  }

  public double f2(Set<Integer> f, Set<Integer> c) {
    Set<Integer> hit = new HashSet<>();
    for (int i : f)
      for (int n : g.neighbors(i)) for (int nn : g.neighbors(n)) if (c.contains(nn)) hit.add(nn);
    return c.isEmpty() ? 0 : hit.size() / (double) c.size();
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
