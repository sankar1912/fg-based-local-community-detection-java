package foldedsubgraph;

import java.awt.*;
import java.awt.image.*;
import java.nio.file.*;
import java.util.*;
import javax.imageio.ImageIO;

/** Renders all three metrics from the paper's Table VI ablation experiment. */
public final class AblationMetricImages {
  public static void main(String[] a) throws Exception {
    Path o = Path.of(a.length > 0 ? a[0] : "output/figures");
    Files.createDirectories(o);
    String[] d = {"LFR1", "LFR2", "LFR3", "LFR4", "LFR5", "LFR6", "LFR7", "LFR8", "LFR9"},
        m = {"M-based", "Pure extension", "FG-based"};
    double[][]
        r =
            {
              {.6834, .8644, .9713},
              {.4436, .7869, .9097},
              {.3008, .6001, .6708},
              {.2414, .9147, .794},
              {.2287, .8348, .8852},
              {.8109, .8308, 1},
              {.6421, .8108, .9937},
              {.5421, .8637, .9707},
              {.6247, .836, .9588}
            },
        p =
            {
              {.8619, .8565, .9319},
              {.6839, .7406, .8635},
              {.4939, .4642, .5637},
              {.345, .2034, .2559},
              {.3138, .2119, .2045},
              {.8256, .83, 1},
              {.8274, .8236, .9923},
              {.8236, .8448, .956},
              {.8244, .8356, .9302}
            },
        f =
            {
              {.7119, .8561, .95},
              {.4908, .7487, .8775},
              {.3294, .4949, .5781},
              {.2208, .3097, .3326},
              {.2159, .2932, .3009},
              {.8113, .8296, 1},
              {.6547, .8111, .9929},
              {.584, .8522, .9632},
              {.6537, .8292, .9396}
            };
    draw(o.resolve("paper-table-vi-recall.png"), "Paper Table VI - Recall", d, m, r);
    draw(o.resolve("paper-table-vi-precision.png"), "Paper Table VI - Precision", d, m, p);
    draw(o.resolve("paper-table-vi-fscore.png"), "Paper Table VI - F-score", d, m, f);
  }

  private static void draw(Path p, String title, String[] rows, String[] cols, double[][] v)
      throws Exception {
    int w = 850, h = 150 + rows.length * 42;
    BufferedImage im = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    Graphics2D g = im.createGraphics();
    g.setColor(new Color(250, 250, 247));
    g.fillRect(0, 0, w, h);
    g.setColor(new Color(35, 40, 45));
    g.setFont(new Font("SansSerif", Font.BOLD, 24));
    g.drawString(title, 30, 38);
    int x = 250, y = 75;
    g.setFont(new Font("SansSerif", Font.BOLD, 16));
    g.drawString("Dataset", 30, y);
    for (int j = 0; j < cols.length; j++) g.drawString(cols[j], x + j * 170, y);
    g.setFont(new Font("SansSerif", Font.PLAIN, 16));
    for (int i = 0; i < rows.length; i++) {
      g.setColor(new Color(35, 40, 45));
      g.drawString(rows[i], 30, y + 35 + i * 42);
      for (int j = 0; j < cols.length; j++) {
        g.setColor(j == 2 ? new Color(48, 132, 91) : new Color(35, 40, 45));
        g.drawString(String.format(Locale.ROOT, "%.4f", v[i][j]), x + j * 170, y + 35 + i * 42);
      }
    }
    g.dispose();
    ImageIO.write(im, "png", p.toFile());
  }
}
