package foldedsubgraph;

/** Exact parameter rows printed in the paper's Table I. */
public record LfrSpec(
    String name,
    int n,
    int k,
    int maxK,
    double mu,
    int t1,
    int t2,
    int minC,
    int maxC,
    int on,
    int om) {
  public static LfrSpec[] paperTable() {
    return new LfrSpec[] {
      new LfrSpec("LFR1", 200, 7, 15, .1, 2, 1, 10, 80, 0, 0),
      new LfrSpec("LFR2", 200, 7, 15, .2, 2, 1, 10, 80, 0, 0),
      new LfrSpec("LFR3", 200, 7, 15, .3, 2, 1, 10, 80, 0, 0),
      new LfrSpec("LFR4", 200, 7, 15, .4, 2, 1, 10, 80, 0, 0),
      new LfrSpec("LFR5", 200, 7, 15, .5, 2, 1, 10, 80, 0, 0),
      new LfrSpec("LFR6", 1000, 15, 50, .1, 2, 1, 10, 50, 0, 0),
      new LfrSpec("LFR7", 1000, 15, 50, .1, 2, 1, 10, 300, 0, 0),
      new LfrSpec("LFR8", 500, 7, 15, .1, 2, 1, 10, 80, 0, 0),
      new LfrSpec("LFR9", 500, 7, 15, .1, 2, 1, 10, 80, 0, 0)
    };
  }
}
