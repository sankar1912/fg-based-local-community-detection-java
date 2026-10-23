package foldedsubgraph;

import java.awt.*;
import java.awt.image.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import javax.imageio.ImageIO;

/** Generates PNG figures without external Java dependencies. */
public final class ReportGenerator {
  private ReportGenerator() {}

  private static final Color BG = new Color(250, 250, 247),
      INK = new Color(35, 40, 45),
      BLUE = new Color(34, 91, 145),
      ORANGE = new Color(214, 118, 38),
      GREEN = new Color(48, 132, 91),
      GRAY = new Color(125, 130, 135);

  public static void main(String[] a) throws Exception {
    Path out = Path.of(a.length > 0 ? a[0] : "output/figures");
    Files.createDirectories(out);
    paperTables(out);
    if (a.length > 2) karate(out, Path.of(a[1]), Path.of(a[2]));
    System.out.println("PNG reports written to " + out.toAbsolutePath());
  }

  private static void paperTables(Path o) throws IOException {
    String[] methods = {"R-based", "M-based", "LQ-based", "LCDNN", "FSR", "FG-based"};
    String[] ds = {"Karate", "Dolphins", "US politics", "Amazon"};
    double[][] f = {
      {.6474, .7007, .7598, .9723, .6922, 1},
      {.4505, .4685, .6489, .7558, .5868, .9466},
      {.5030, .5257, .6178, .7463, .7543, .7923},
      {.6322, .6465, .7013, .7546, .7601, .7783}
    };
    table(
        o.resolve("paper-table-iii-true-datasets.png"),
        "Paper Table III - F-score on true datasets",
        ds,
        methods,
        f);
    String[] lfr = {"LFR1", "LFR2", "LFR3", "LFR4", "LFR5", "LFR6", "LFR7", "LFR8", "LFR9"};
    double[][] lf = {
      {.7087, .7119, .7302, .8566, .6698, .9500},
      {.4867, .4908, .6249, .7394, .5317, .8775},
      {.3297, .3294, .4614, .4571, .3362, .5781},
      {.2214, .2208, .2696, .4059, .1831, .3326},
      {.2171, .2159, .2819, .2563, .1488, .3009},
      {.8113, .8113, .8147, 1, 1, 1},
      {.6547, .6547, .6646, .9598, .9548, .9929},
      {.5831, .584, .6196, .6411, .9548, .9632},
      {.652, .6537, .6664, .7847, .613, .9396}
    };
    table(
        o.resolve("paper-table-iv-lfr-fscore.png"),
        "Paper Table IV - F-score on LFR datasets",
        lfr,
        methods,
        lf);
    String[] eff = {"Time(s)", "Recall", "Precision", "F-score", "Var"};
    double[][] ev = {
      {.0291, .0038, .0515, .1331, .0702, .0513},
      {.6049, .6063, .6406, .7795, .6957, .9520},
      {.7620, .7620, .7597, .9102, .9528, .8810},
      {.6227, .6234, .6509, .8018, .7163, .8938},
      {.1458, .1185, .1398, .0514, .0764, .0226}
    };
    table(
        o.resolve("paper-table-v-efficiency.png"),
        "Paper Table V - Efficiency and sensitivity",
        eff,
        methods,
        ev);
    String[] ab = {"LFR1", "LFR2", "LFR3", "LFR4", "LFR5", "LFR6", "LFR7", "LFR8", "LFR9"};
    double[][] av = {
      {.9500, .8561, .9713},
      {.8775, .7487, .8365},
      {.5781, .4949, .5637},
      {.3326, .3097, .3326},
      {.3009, .2932, .3009},
      {1, .8296, 1},
      {.9929, .8111, .9929},
      {.9632, .8522, .9632},
      {.9396, .8292, .9396}
    };
    table(
        o.resolve("paper-table-vi-ablation-fscore.png"),
        "Paper Table VI - Three-stage extension F-score",
        ab,
        new String[] {"M-based", "Pure extension", "FG-based"},
        av);
  }

  private static void table(Path file, String title, String[] rows, String[] cols, double[][] v)
      throws IOException {
    int w = 1100, h = 150 + rows.length * 42;
    BufferedImage im = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    Graphics2D g = im.createGraphics();
    base(g, w, h);
    g.setFont(new Font("SansSerif", Font.BOLD, 24));
    g.setColor(INK);
    g.drawString(title, 30, 38);
    int x0 = 250, y0 = 75;
    g.setFont(new Font("SansSerif", Font.BOLD, 16));
    g.drawString("Dataset / metric", 30, y0);
    for (int j = 0; j < cols.length; j++) g.drawString(cols[j], x0 + j * 135, y0);
    g.setFont(new Font("SansSerif", Font.PLAIN, 16));
    for (int i = 0; i < rows.length; i++) {
      int y = y0 + 35 + i * 42;
      g.drawString(rows[i], 30, y);
      for (int j = 0; j < cols.length; j++) {
        if (j == cols.length - 1) g.setColor(GREEN);
        else g.setColor(INK);
        g.drawString(String.format(Locale.ROOT, "%.4f", v[i][j]), x0 + j * 135, y);
      }
    }
    g.dispose();
    ImageIO.write(im, "png", file.toFile());
  }

  private static void karate(Path o, Path graph, Path truth) throws Exception {
    Graph g = DatasetReader.edgeList(graph, true);
    java.util.List<Set<Integer>> t = DatasetReader.communities(truth, true);
    double[][] m = new double[34][3];
    for (int s = 0; s < 34; s++) {
      Metrics.Score z =
          Metrics.score(new FGDetector(g).detect(s, .45, 1.0), Metrics.truthFor(s, t));
      m[s] = new double[] {z.recall(), z.precision(), z.f1()};
    }
    lineChart(
        o.resolve("karate-per-seed-metrics.png"),
        "Karate - FG-based metrics by seed",
        m,
        new String[] {"Recall", "Precision", "F-score"});
    network(o.resolve("karate-detected-community.png"), g, new FGDetector(g).detect(0, .45, 1.0));
  }

  private static void lineChart(Path file, String title, double[][] v, String[] names)
      throws IOException {
    int w = 1200, h = 650;
    BufferedImage im = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    Graphics2D g = im.createGraphics();
    base(g, w, h);
    g.setColor(INK);
    g.setFont(new Font("SansSerif", Font.BOLD, 24));
    g.drawString(title, 35, 40);
    int l = 80, r = 30, t = 80, b = 70, pw = w - l - r, ph = h - t - b;
    g.setFont(new Font("SansSerif", Font.PLAIN, 14));
    for (int k = 0; k <= 10; k++) {
      int y = t + ph - k * ph / 10;
      g.setColor(new Color(220, 220, 215));
      g.drawLine(l, y, w - r, y);
      g.setColor(INK);
      g.drawString(String.format(Locale.ROOT, "%.1f", k / 10.), 35, y + 5);
    }
    Color[] cs = {BLUE, ORANGE, GREEN};
    for (int j = 0; j < names.length; j++) {
      g.setColor(cs[j]);
      for (int i = 1; i < v.length; i++) {
        int x1 = l + (i - 1) * pw / (v.length - 1),
            x2 = l + i * pw / (v.length - 1),
            y1 = t + (int) ((1 - v[i - 1][j]) * ph),
            y2 = t + (int) ((1 - v[i][j]) * ph);
        g.drawLine(x1, y1, x2, y2);
      }
      g.fillRect(l + j * 150, t - 25, 18, 12);
      g.setColor(INK);
      g.drawString(names[j], l + 25 + j * 150, t - 14);
    }
    g.setColor(INK);
    g.drawString("Seed node", w / 2, h - 20);
    g.rotate(-Math.PI / 2);
    g.drawString("Score", -h / 2, 20);
    g.dispose();
    ImageIO.write(im, "png", file.toFile());
  }

  private static void network(Path file, Graph g, Set<Integer> community) throws IOException {
    int w = 900, h = 700;
    BufferedImage im = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    Graphics2D x = im.createGraphics();
    base(x, w, h);
    int cx = w / 2, cy = h / 2, rad = 260;
    int[] px = new int[g.size()], py = new int[g.size()];
    for (int i = 0; i < g.size(); i++) {
      double a = 2 * Math.PI * i / g.size();
      px[i] = cx + (int) (rad * Math.cos(a));
      py[i] = cy + (int) (rad * Math.sin(a));
    }
    x.setStroke(new BasicStroke(1));
    for (int u = 0; u < g.size(); u++)
      for (int v : g.neighbors(u))
        if (u < v) {
          x.setColor(new Color(205, 205, 200));
          x.drawLine(px[u], py[u], px[v], py[v]);
        }
    for (int i = 0; i < g.size(); i++) {
      x.setColor(community.contains(i) ? ORANGE : BLUE);
      x.fillOval(px[i] - 7, py[i] - 7, 14, 14);
      x.setColor(INK);
      x.drawString(Integer.toString(i + 1), px[i] + 8, py[i] + 5);
    }
    x.setColor(INK);
    x.setFont(new Font("SansSerif", Font.BOLD, 22));
    x.drawString("Karate seed 1 detected community", 25, 35);
    x.dispose();
    ImageIO.write(im, "png", file.toFile());
  }

  private static void base(Graphics2D g, int w, int h) {
    g.setColor(BG);
    g.fillRect(0, 0, w, h);
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
  }
}
