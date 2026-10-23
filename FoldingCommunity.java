package foldedsubgraph;

import java.util.*;

/** Algorithm 2: repeated folding-stage expansion. */
public final class FoldingCommunity {
  private final Graph graph;
  private final Folding folding;

  public FoldingCommunity(Graph g) {
    graph = g;
    folding = new Folding(g);
  }

  public Set<Integer> expand(Set<Integer> initial, double gamma) {
    Set<Integer> c = new LinkedHashSet<>(initial);
    while (true) {
      boolean added = false;
      for (int v : new ArrayList<>(alternative(c))) {
        Set<Integer> f = folding.build(v, gamma);
        int in = 0;
        for (int x : f) if (c.contains(x)) in++;
        if (in >= .5 * f.size() && c.addAll(f)) added = true;
      }
      if (!added) return c;
    }
  }

  public Set<Integer> alternative(Set<Integer> c) {
    Set<Integer> a = new LinkedHashSet<>();
    for (int v : c) for (int u : graph.neighbors(v)) if (!c.contains(u)) a.add(u);
    return a;
  }
}
