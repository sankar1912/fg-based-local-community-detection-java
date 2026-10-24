package foldedsubgraph;

import java.awt.*;
import java.awt.image.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import javax.imageio.ImageIO;

/** Generates base-versus-novelty comparison charts from computed summary.csv. */
public final class NoveltyVisualization {
  private static final Color BG = new Color(250, 250, 247);
  private static final Color INK = new Color(35, 40, 45);
  private static final Color BASE = new Color(34, 91, 145);
  private static final Color NOVELTY = new Color(48, 132, 91);

  public static void main(String[] args) throws Exception {
    Path summary = Path.of(args[0]);
    Path output = Path.of(args.length > 1 ? args[1] : "output/novelty-results/figures");
    Files.createDirectories(output);
    java.util.List<Row> rows = read(summary);
    chart(output.resolve("base-vs-novelty-metrics.png"), rows);
    chart(output.resolve("base-vs-novelty-f1.png"), rows, true);
  }

  private static java.util.List<Row> read(Path file) throws IOException {
    java.util.List<Row> rows = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(file)) {
      String line;
      reader.readLine();
      while ((line = reader.readLine()) != null) {
        String[] p = line.split(",");
        rows.add(new Row(p[0], p[1], Double.parseDouble(p[3]),
            Double.parseDouble(p[4]), Double.parseDouble(p[5])));
      }
    }
    return rows;
  }

  private static void chart(Path file, java.util.List<Row> rows) throws IOException {
    chart(file, rows, false);
  }

  private static void chart(Path file, java.util.List<Row> rows, boolean onlyF1) throws IOException {
    String[] datasets = {"LFR6", "LFR7", "LFR8", "LFR9"};
    int width = onlyF1 ? 1000 : 1250;
    int height = 700;
    BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    Graphics2D g = image.createGraphics();
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
    g.setColor(BG);
    g.fillRect(0, 0, width, height);
    g.setColor(INK);
    g.setFont(new Font("SansSerif", Font.BOLD, 24));
    g.drawString(onlyF1 ? "FG versus novelty F-score" : "FG versus novelty metrics", 35, 40);
    int left = 90;
    int top = 90;
    int plotWidth = width - 130;
    int plotHeight = height - 170;
    for (int tick = 0; tick <= 10; tick++) {
      int y = top + plotHeight - tick * plotHeight / 10;
      g.setColor(new Color(220, 220, 215));
      g.drawLine(left, y, left + plotWidth, y);
      g.setColor(INK);
      g.setFont(new Font("SansSerif", Font.PLAIN, 14));
      g.drawString(String.format(Locale.ROOT, "%.1f", tick / 10.0), 45, y + 5);
    }
    int groupWidth = plotWidth / datasets.length;
    int seriesCount = onlyF1 ? 1 : 3;
    String[] names = onlyF1 ? new String[] {"F-score"}
        : new String[] {"Recall", "Precision", "F-score"};
    Color[] colors = {BASE, NOVELTY};
    for (int i = 0; i < datasets.length; i++) {
      int baseX = left + i * groupWidth + groupWidth / 5;
      for (int method = 0; method < 2; method++) {
        Row row = find(rows, datasets[i], method == 0 ? "FG" : "Novelty");
        for (int metric = 0; metric < seriesCount; metric++) {
          double value = onlyF1 ? row.f1 : metric == 0 ? row.recall : metric == 1 ? row.precision : row.f1;
          int x = baseX + method * 48 + metric * 13;
          int barHeight = (int) Math.round(value * plotHeight);
          g.setColor(colors[method]);
          g.fillRect(x, top + plotHeight - barHeight, 11, barHeight);
        }
      }
      g.setColor(INK);
      g.drawString(datasets[i], baseX + 15, top + plotHeight + 28);
    }
    g.setColor(BASE);
    g.fillRect(left, 55, 18, 12);
    g.setColor(INK);
    g.drawString("FG", left + 25, 66);
    g.setColor(NOVELTY);
    g.fillRect(left + 85, 55, 18, 12);
    g.setColor(INK);
    g.drawString("Novelty", left + 110, 66);
    g.drawString(onlyF1 ? "Dataset" : "Each dataset: Recall, Precision, F-score", width / 2 - 100, height - 25);
    g.dispose();
    ImageIO.write(image, "png", file.toFile());
  }

  private static Row find(java.util.List<Row> rows, String dataset, String method) {
    return rows.stream().filter(row -> row.dataset.equals(dataset) && row.method.equals(method))
        .findFirst().orElseThrow();
  }

  private record Row(String dataset, String method, double recall, double precision, double f1) {}
}
