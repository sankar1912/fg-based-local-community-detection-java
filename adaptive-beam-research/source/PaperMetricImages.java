package foldedsubgraph;

import java.awt.*;
import java.awt.image.*;
import java.nio.file.*;
import java.util.*;
import javax.imageio.ImageIO;

/** Renders the paper's reported recall, precision and F-score tables as PNGs. */
public final class PaperMetricImages {
  private static final Color BG = new Color(250, 250, 247),
      INK = new Color(35, 40, 45),
      GREEN = new Color(48, 132, 91);

  private PaperMetricImages() {}

  public static void main(String[] a) throws Exception {
    Path o = Path.of(a.length > 0 ? a[0] : "output/figures");
    Files.createDirectories(o);
    String[] m = {"R-based", "M-based", "LQ-based", "LCDNN", "FSR", "FG-based"};
    String[] d = {"Karate", "Dolphins", "US politics", "Amazon"};
    double[][][] trueD = {
      {
        {.5527, .6442, .7032, 1, .6091, 1},
        {.9088, .8905, .8905, .9477, 1, 1},
        {.6474, .7007, .7598, .9723, .6922, 1}
      },
      {
        {.323, .3464, .5546, .6594, .4919, .9539},
        {.9648, .9667, .9528, .9693, 1, .9413},
        {.4505, .4685, .6489, .7558, .5868, .9466}
      },
      {
        {.441, .4837, .5938, .7679, .8316, .8848},
        {.7752, .7664, .7729, .7577, .7379, .7359},
        {.503, .5257, .6178, .7463, .7543, .7923}
      },
      {
        {.5688, .5993, .6382, .7424, .6894, .7587},
        {.9281, .9137, .9273, .9133, .9045, .9004},
        {.6322, .6465, .7013, .7546, .7601, .7783}
      }
    };
    for (int k = 0; k < 3; k++)
      draw(
          o.resolve("paper-table-iii-" + name(k) + ".png"),
          "Paper Table III - " + name(k) + " on true datasets",
          d,
          m,
          transpose(trueD, k));
    String[] l = {"LFR1", "LFR2", "LFR3", "LFR4", "LFR5", "LFR6", "LFR7", "LFR8", "LFR9"};
    double[][][] lv = {
      {
        {.6794, .6834, .6952, .8308, .6184, .9713},
        {.8617, .8619, .8536, .9424, 1, .9319},
        {.7087, .7119, .7302, .8566, .6698, .95}
      },
      {
        {.4367, .4436, .6137, .6945, .5472, .9097},
        {.6843, .6839, .6985, .8775, .816, .8635},
        {.4867, .4908, .6249, .7394, .5317, .8775}
      },
      {
        {.3, .3008, .4993, .3936, .3311, .6708},
        {.4947, .4939, .498, .661, .8441, .5637},
        {.3297, .3294, .4614, .4571, .3362, .5781}
      },
      {
        {.2398, .2414, .3478, .4685, .363, .794},
        {.3453, .345, .3163, .4947, .5108, .2559},
        {.2214, .2208, .2696, .4059, .1831, .3326}
      },
      {
        {.2266, .2287, .3414, .304, .1113, .8852},
        {.3148, .3138, .2986, .3527, .7676, .2045},
        {.2171, .2159, .2819, .2563, .1488, .3009}
      },
      {
        {.8109, .8109, .8147, 1, 1, 1},
        {.8256, .8256, .8271, 1, 1, 1},
        {.8113, .8113, .8147, 1, 1, 1}
      },
      {
        {.6421, .6421, .651, .9528, .9419, .9937},
        {.8274, .8274, .8278, .999, 1, .9923},
        {.6547, .6547, .6646, .9598, .9548, .9929}
      },
      {
        {.5408, .5421, .5694, .5619, .4887, .9707},
        {.8235, .8236, .8236, .9469, 1, .956},
        {.5831, .584, .6196, .6411, .9548, .9632}
      },
      {
        {.6221, .6247, .6328, .5631, .5504, .9588},
        {.824, .8244, .8188, .9576, 1, .9302},
        {.652, .6537, .6664, .7847, .613, .9396}
      }
    };
    for (int k = 0; k < 3; k++)
      draw(
          o.resolve("paper-table-iv-" + name(k) + ".png"),
          "Paper Table IV - " + name(k) + " on LFR datasets",
          l,
          m,
          transpose(lv, k));
    System.out.println("All paper metric PNGs written to " + o.toAbsolutePath());
  }

  private static String name(int k) {
    return k == 0 ? "recall" : k == 1 ? "precision" : "fscore";
  }

  private static double[][] transpose(double[][][] a, int k) {
    double[][] r = new double[a.length][];
    for (int i = 0; i < a.length; i++) r[i] = a[i][k];
    return r;
  }

  private static void draw(Path p, String title, String[] rows, String[] cols, double[][] v)
      throws Exception {
    int w = 1100, h = 150 + rows.length * 42;
    BufferedImage im = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    Graphics2D g = im.createGraphics();
    g.setColor(BG);
    g.fillRect(0, 0, w, h);
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
    g.setColor(INK);
    g.setFont(new Font("SansSerif", Font.BOLD, 24));
    g.drawString(title, 30, 38);
    int x0 = 250, y0 = 75;
    g.setFont(new Font("SansSerif", Font.BOLD, 16));
    g.drawString("Dataset", 30, y0);
    for (int j = 0; j < cols.length; j++) g.drawString(cols[j], x0 + j * 135, y0);
    g.setFont(new Font("SansSerif", Font.PLAIN, 16));
    for (int i = 0; i < rows.length; i++) {
      int y = y0 + 35 + i * 42;
      g.setColor(INK);
      g.drawString(rows[i], 30, y);
      for (int j = 0; j < cols.length; j++) {
        g.setColor(j == cols.length - 1 ? GREEN : INK);
        g.drawString(String.format(Locale.ROOT, "%.4f", v[i][j]), x0 + j * 135, y);
      }
    }
    g.dispose();
    ImageIO.write(im, "png", p.toFile());
  }
}
