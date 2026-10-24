package foldedsubgraph;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import javax.imageio.ImageIO;

/** Creates a PNG from the saved boosted-tree summary CSV. */
public final class BoostedTreeVisualization {
  private BoostedTreeVisualization() {}

  public static void main(String[] args) throws IOException {
    Path output = Path.of(args.length == 0 ? "output/boosted-tree-results-5" : args[0]);
    List<String[]> rows = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(output.resolve("summary.csv"))) {
      String line;
      while ((line = reader.readLine()) != null) rows.add(line.split(","));
    }
    BufferedImage image = new BufferedImage(1000, 650, BufferedImage.TYPE_INT_ARGB);
    Graphics2D g = image.createGraphics();
    g.setColor(Color.WHITE);
    g.fillRect(0, 0, 1000, 650);
    g.setColor(Color.BLACK);
    g.setFont(new Font("SansSerif", Font.BOLD, 20));
    g.drawString("FG versus logistic and boosted-stump F1", 90, 45);
    g.drawLine(90, 540, 900, 540);
    g.drawLine(90, 80, 90, 540);
    List<String> methods = List.of("FG", "PointwiseLogistic", "BoostedStumps");
    for (String[] row : rows)
      if (!row[0].equals("dataset")) {
        int dataset = Integer.parseInt(row[0].substring(3));
        int method = methods.indexOf(row[1]);
        int x = 120 + (dataset - 6) * 210 + method * 50;
        int height = (int) (Double.parseDouble(row[5]) * 400);
        g.setColor(method == 0 ? Color.BLUE : method == 1 ? Color.RED : Color.GREEN.darker());
        g.fillRect(x, 540 - height, 38, height);
        g.setColor(Color.BLACK);
        g.drawString(row[0], 120 + (dataset - 6) * 210, 570);
      }
    Files.createDirectories(output.resolve("figures"));
    ImageIO.write(image, "png", output.resolve("figures/boosted-vs-fg-f1.png").toFile());
  }
}
