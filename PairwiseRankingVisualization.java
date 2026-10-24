package foldedsubgraph;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import javax.imageio.*;

/** Creates a comparison figure from pairwise experiment summary.csv. */
public final class PairwiseRankingVisualization {
  private PairwiseRankingVisualization() {}

  public static void main(String[] args) throws IOException {
    Path output = Path.of(args.length == 0 ? "output/pairwise-ranking-results-5" : args[0]);
    List<String[]> rows = read(output.resolve("summary.csv"));
    BufferedImage image = new BufferedImage(1000, 650, BufferedImage.TYPE_INT_ARGB);
    Graphics2D g = image.createGraphics();
    g.setColor(Color.WHITE);
    g.fillRect(0, 0, 1000, 650);
    g.setColor(Color.BLACK);
    g.setFont(new Font("SansSerif", Font.BOLD, 20));
    g.drawString("FG versus pointwise and pairwise ranking F1", 90, 45);
    g.drawLine(90, 540, 900, 540);
    g.drawLine(90, 80, 90, 540);
    List<String> methods = List.of("FG", "PointwiseLogistic", "PairwiseRank");
    for (int i = 0; i < rows.size(); i++) {
      String[] r = rows.get(i);
      if (r[0].equals("dataset")) continue;
      int x = 120 + Integer.parseInt(r[0].substring(3)) * 180;
      int method = methods.indexOf(r[1]);
      int h = (int) (Double.parseDouble(r[5]) * 400);
      g.setColor(method == 0 ? Color.BLUE : method == 1 ? Color.RED : Color.GREEN.darker());
      g.fillRect(x + method * 42, 540 - h, 34, h);
      g.setColor(Color.BLACK);
      g.drawString(r[0], x, 570);
    }
    Files.createDirectories(output.resolve("figures"));
    ImageIO.write(image, "png", output.resolve("figures/pairwise-vs-fg-f1.png").toFile());
  }

  private static List<String[]> read(Path file) throws IOException {
    List<String[]> rows = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(file)) {
      String line;
      while ((line = reader.readLine()) != null) rows.add(line.split(","));
    }
    return rows;
  }
}
