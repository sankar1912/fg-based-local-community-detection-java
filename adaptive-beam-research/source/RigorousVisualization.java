package foldedsubgraph;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import javax.imageio.*;

/** Generates PNG figures only from the saved rigorous CSV artifacts. */
public final class RigorousVisualization {
  private RigorousVisualization() {}

  public static void main(String[] args) throws IOException {
    Path output = Path.of(args.length == 0 ? "output/rigorous-evaluation" : args[0]);
    List<String[]> summary = read(output.resolve("dataset_summary.csv"));
    List<String[]> pairs = read(output.resolve("paired_seed_results.csv"));
    List<String[]> perSeed = read(output.resolve("per_seed_results.csv"));
    drawBars(
        output.resolve("fg_vs_hybrid_f1_by_dataset.png"), summary, "FG mean F1", "Hybrid mean F1");
    drawDifferences(output.resolve("paired_f1_difference_by_dataset.png"), pairs);
    drawDistribution(output.resolve("f1_difference_distribution.png"), pairs);
    drawModel(output.resolve("model_comparison.png"), read(output.resolve("model_comparison.csv")));
    drawOverrides(output.resolve("override_rate_vs_f1.png"), summary, perSeed);
  }

  private static List<String[]> read(Path file) throws IOException {
    List<String[]> result = new ArrayList<>();
    if (!Files.exists(file)) return result;
    try (BufferedReader r = Files.newBufferedReader(file)) {
      String line;
      while ((line = r.readLine()) != null) result.add(line.split(",", -1));
    }
    return result;
  }

  private static void drawBars(Path file, List<String[]> rows, String left, String right)
      throws IOException {
    List<String[]> data =
        rows.stream()
            .filter(r -> r.length > 3 && !r[0].equals("dataset") && r[1].equals("Logistic"))
            .toList();
    BufferedImage image = canvas(1000, 650);
    Graphics2D g = image.createGraphics();
    axes(g, "FG versus safe-gated logistic F1", 1000, 650);
    for (int i = 0; i < data.size(); i++) {
      String[] r = data.get(i);
      int x = 130 + i * 210;
      bar(g, x, 540, Double.parseDouble(r[3]), Color.BLUE);
      bar(g, x + 65, 540, Double.parseDouble(r[4]), Color.RED);
      g.setColor(Color.BLACK);
      g.drawString(r[0], x, 570);
    }
    legend(g, left, right);
    save(image, file);
  }

  private static void drawDifferences(Path file, List<String[]> rows) throws IOException {
    Map<String, List<Double>> values = new LinkedHashMap<>();
    for (String[] r : rows)
      if (!r[0].equals("dataset") && r[3].equals("Logistic"))
        values.computeIfAbsent(r[0], x -> new ArrayList<>()).add(Double.parseDouble(r[6]));
    BufferedImage image = canvas(1000, 650);
    Graphics2D g = image.createGraphics();
    axes(g, "Paired F1 differences: hybrid - FG", 1000, 650);
    int i = 0;
    for (var e : values.entrySet()) {
      double m = e.getValue().stream().mapToDouble(x -> x).average().orElse(0);
      int x = 150 + i++ * 210;
      g.setColor(m >= 0 ? Color.GREEN.darker() : Color.RED);
      int y = 540 - (int) (m * 300);
      g.fillRect(x, Math.min(y, 540), 80, Math.abs(y - 540));
      g.setColor(Color.BLACK);
      g.drawString(e.getKey(), x, 570);
    }
    save(image, file);
  }

  private static void drawDistribution(Path file, List<String[]> rows) throws IOException {
    List<Double> values = new ArrayList<>();
    for (String[] r : rows)
      if (!r[0].equals("dataset") && r[3].equals("Logistic")) values.add(Double.parseDouble(r[6]));
    BufferedImage image = canvas(1000, 650);
    Graphics2D g = image.createGraphics();
    axes(g, "Distribution of paired F1 differences", 1000, 650);
    int[] bins = new int[20];
    for (double v : values) bins[Math.max(0, Math.min(19, (int) ((v + 1) * 10)))]++;
    int max = Arrays.stream(bins).max().orElse(1);
    for (int i = 0; i < bins.length; i++) {
      int h = bins[i] * 400 / max;
      g.setColor(Color.DARK_GRAY);
      g.fillRect(100 + i * 40, 540 - h, 32, h);
    }
    save(image, file);
  }

  private static void drawModel(Path file, List<String[]> rows) throws IOException {
    BufferedImage image = canvas(1000, 650);
    Graphics2D g = image.createGraphics();
    axes(g, "Model comparison: mean F1 difference", 1000, 650);
    int i = 0;
    for (String[] r : rows)
      if (!r[0].equals("model")) {
        double v = Double.parseDouble(r[2]);
        int x = 150 + i++ * 220;
        int y = 540 - (int) (v * 300);
        g.setColor(v >= 0 ? Color.BLUE : Color.RED);
        g.fillRect(x, Math.min(y, 540), 100, Math.abs(y - 540));
        g.setColor(Color.BLACK);
        g.drawString(r[0], x, 570);
      }
    save(image, file);
  }

  private static void drawOverrides(Path file, List<String[]> rows, List<String[]> perSeed)
      throws IOException {
    BufferedImage image = canvas(1000, 650);
    Graphics2D g = image.createGraphics();
    axes(g, "Logistic override rate versus hybrid F1", 1000, 650);
    int i = 0;
    for (String[] r : rows)
      if (!r[0].equals("dataset") && r[1].equals("Logistic")) {
        String dataset = r[0];
        long total =
            perSeed.stream()
                .filter(
                    x ->
                        x.length > 13
                            && x[0].equals(dataset)
                            && x[3].equals("Logistic")
                            && x[13].equals("SUCCESS"))
                .count();
        long overrides =
            perSeed.stream()
                .filter(
                    x ->
                        x.length > 13
                            && x[0].equals(dataset)
                            && x[3].equals("Logistic")
                            && x[13].equals("SUCCESS")
                            && Boolean.parseBoolean(x[10]))
                .count();
        double rate = total == 0 ? 0 : overrides / (double) total;
        int x = 150 + i++ * 210;
        int y = 540 - (int) (rate * 300);
        g.setColor(Color.ORANGE);
        g.fillOval(x, y, 20, 20);
        g.setColor(Color.BLACK);
        g.drawString(dataset, x, 570);
      }
    save(image, file);
  }

  private static BufferedImage canvas(int w, int h) {
    BufferedImage b = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
    Graphics2D g = b.createGraphics();
    g.setColor(Color.WHITE);
    g.fillRect(0, 0, w, h);
    g.dispose();
    return b;
  }

  private static void axes(Graphics2D g, String title, int w, int h) {
    g.setColor(Color.BLACK);
    g.setFont(new Font("SansSerif", Font.BOLD, 20));
    g.drawString(title, 90, 45);
    g.drawLine(90, 540, 900, 540);
    g.drawLine(90, 80, 90, 540);
  }

  private static void bar(Graphics2D g, int x, int baseline, double value, Color color) {
    int h = (int) (value * 400);
    g.setColor(color);
    g.fillRect(x, baseline - h, 50, h);
  }

  private static void legend(Graphics2D g, String a, String b) {
    g.setColor(Color.BLUE);
    g.fillRect(730, 90, 18, 18);
    g.setColor(Color.BLACK);
    g.drawString(a, 755, 105);
    g.setColor(Color.RED);
    g.fillRect(730, 120, 18, 18);
    g.setColor(Color.BLACK);
    g.drawString(b, 755, 135);
  }

  private static void save(BufferedImage image, Path file) throws IOException {
    Files.createDirectories(file.getParent());
    ImageIO.write(image, "png", file.toFile());
  }
}
