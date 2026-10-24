package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/** Readers for edge lists, the project's legacy adjacency format, and truth communities. */
public final class DatasetReader {
  private DatasetReader() {}

  public static Graph edgeList(Path file, boolean oneBased) throws IOException {
    List<int[]> es = new ArrayList<>();
    int max = -1;
    try (BufferedReader r = Files.newBufferedReader(file)) {
      String l;
      while ((l = r.readLine()) != null) {
        l = l.trim();
        if (l.isEmpty() || l.startsWith("#") || l.startsWith("%")) continue;
        String[] p = l.split("\\s+");
        if (p.length < 2 || !p[0].matches("-?\\d+") || !p[1].matches("-?\\d+")) continue;
        int a = Integer.parseInt(p[0]) - (oneBased ? 1 : 0),
            b = Integer.parseInt(p[1]) - (oneBased ? 1 : 0);
        if (a >= 0 && b >= 0) {
          es.add(new int[] {a, b});
          max = Math.max(max, Math.max(a, b));
        }
      }
    }
    Graph g = new Graph(max + 1);
    for (int[] e : es) g.addEdge(e[0], e[1]);
    return g;
  }

  public static Graph adjacencyList(Path file, boolean oneBased) throws IOException {
    try (BufferedReader r = Files.newBufferedReader(file)) {
      String f = r.readLine();
      if (f == null) throw new IOException("empty graph file");
      int n = Integer.parseInt(f.trim().split("\\s+")[0]);
      Graph g = new Graph(n);
      for (int row = 0; row < n; row++) {
        String l = r.readLine();
        if (l == null) break;
        for (String t : l.trim().split("\\s+"))
          if (!t.isEmpty()) {
            int v = Integer.parseInt(t) - (oneBased ? 1 : 0);
            if (v >= 0 && v < n) g.addEdge(row, v);
          }
      }
      return g;
    }
  }

  public static List<Set<Integer>> communities(Path file, boolean oneBased) throws IOException {
    List<Set<Integer>> out = new ArrayList<>();
    try (BufferedReader r = Files.newBufferedReader(file)) {
      String l;
      while ((l = r.readLine()) != null) {
        l = l.trim();
        if (l.isEmpty() || l.startsWith("#")) continue;
        Set<Integer> c = new LinkedHashSet<>();
        for (String t : l.split("\\s+")) c.add(Integer.parseInt(t) - (oneBased ? 1 : 0));
        if (!c.isEmpty()) out.add(c);
      }
    }
    return out;
  }
}
