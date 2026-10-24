package foldedsubgraph;

import java.util.*;

/** Algorithm 5: complete FG-based local community detector. */
public final class FGDetector {
  private final Graph g;
  private final FoldingCommunity fold;
  private final ModularityExpansion mod;
  private final RelaxationStage relax;

  public FGDetector(Graph g) {
    this.g = g;
    fold = new FoldingCommunity(g);
    mod = new ModularityExpansion(g);
    relax = new RelaxationStage(g);
  }

  public Set<Integer> detect(int seed, double gamma, double h) {
    Set<Integer> c = new LinkedHashSet<>();
    c.add(seed);
    int nums1 = c.size(), nums2 = c.size();
    while (!empty(c)) {
      c = fold.expand(c, gamma);
      c = mod.expand(c, gamma);
      nums1 = c.size();
      if (nums1 == nums2) break;
      c = relax.expand(c, gamma, h);
      nums2 = c.size();
    }
    return c;
  }

  private boolean empty(Set<Integer> c) {
    for (int u : c) for (int v : g.neighbors(u)) if (!c.contains(v)) return false;
    return true;
  }
}
