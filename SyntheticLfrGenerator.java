package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Deterministic large-volume planted benchmark generator using the paper's LFR parameter rows. It
 * preserves N, k, maxK, mu, minc/maxc and writes the graph/truth files. It is an executable
 * replacement when the authors' exact LFR realizations and seeds are unavailable; it is not claimed
 * to be the original Lancichinetti generator.
 */
public final class SyntheticLfrGenerator {
  private SyntheticLfrGenerator() {}

  public static void generateAll(Path root, long seed) throws IOException {
    for (LfrSpec s : LfrSpec.paperTable()) generate(root, s, seed + s.name().hashCode());
  }

  public static void generate(Path root, LfrSpec s, long seed) throws IOException {
    Files.createDirectories(root);
    Random r = new Random(seed);
    int[] comm = new int[s.n()];
    List<Integer> sizes = new ArrayList<>();
    int left = s.n();
    while (left > 0) {
      int size =
          Math.min(
              left, s.minC() + r.nextInt(Math.max(1, Math.min(s.maxC(), left) - s.minC() + 1)));
      sizes.add(size);
      left -= size;
    }
    int at = 0;
    for (int c = 0; c < sizes.size(); c++) for (int j = 0; j < sizes.get(c); j++) comm[at++] = c;
    for (int i = s.n() - 1; i > 0; i--) {
      int j = r.nextInt(i + 1);
      int x = comm[i];
      comm[i] = comm[j];
      comm[j] = x;
    }
    double avgC = s.n() / (double) sizes.size(),
        pin = Math.min(1., (1 - s.mu()) * s.k() / Math.max(1., avgC - 1)),
        pout = Math.min(1., s.mu() * s.k() / Math.max(1., s.n() - avgC));
    Set<Long> edges = new HashSet<>();
    for (int u = 0; u < s.n(); u++)
      for (int v = u + 1; v < s.n(); v++) {
        double p = comm[u] == comm[v] ? pin : pout;
        if (r.nextDouble() < p) {
          if (degree(edges, u) < s.maxK() && degree(edges, v) < s.maxK()) edges.add(key(u, v));
        }
      }
    Path graph = root.resolve(s.name() + ".edgelist"), truth = root.resolve(s.name() + ".truth");
    try (BufferedWriter w = Files.newBufferedWriter(graph)) {
      for (long e : edges) w.write((e >>> 32) + " " + (e & 0xffffffffL) + "\n");
    }
    Map<Integer, List<Integer>> groups = new TreeMap<>();
    for (int i = 0; i < comm.length; i++)
      groups.computeIfAbsent(comm[i], x -> new ArrayList<>()).add(i);
    try (BufferedWriter w = Files.newBufferedWriter(truth)) {
      for (List<Integer> c : groups.values()) {
        for (int i = 0; i < c.size(); i++) {
          if (i > 0) w.write(" ");
          w.write(Integer.toString(c.get(i)));
        }
        w.write("\n");
      }
    }
  }

  private static long key(int a, int b) {
    return ((long) a << 32) | (b & 0xffffffffL);
  }

  private static int degree(Set<Long> e, int v) {
    int n = 0;
    for (long x : e) if ((int) (x >>> 32) == v || ((int) x) == v) n++;
    return n;
  }
}
