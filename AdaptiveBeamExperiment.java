package foldedsubgraph;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import javax.imageio.ImageIO;

/** Validation-tuned candidate-budget beam experiment for scalable recovery from greedy errors. */
public final class AdaptiveBeamExperiment {
  private static final double GAMMA = .45;
  private static final double H = 1.0;
  private static final int[] BUDGETS = {16, 32, 64, 128};

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args.length > 0 ? args[0] : "datasets/generated-lfr");
    Path output = Path.of(args.length > 1 ? args[1] : "output/adaptive-beam-results-5");
    int limit = args.length > 2 ? Integer.parseInt(args[2]) : 5;
    Files.createDirectories(output);

    LogisticCandidateScorer scorer =
        LogisticCandidateScorer.train(collect(data, 1, 5), 500, .08, .001);
    int selectedBudget = tune(data, scorer, output);
    scorer.save(output.resolve("logistic-scorer.txt"));
    Files.writeString(
        output.resolve("experiment-config.txt"),
        "training=LFR1-LFR5\nvalidation=LFR3-LFR5 first five eligible seeds\n"
            + "held_out=LFR6-LFR9\nbeam_width=2\nmax_depth=5\nlearned_weight=0.10\n"
            + "candidate_budget=" + selectedBudget + "\nseed_limit=" + limit + "\n");
    run(data, output, scorer, selectedBudget, limit);
    figure(output.resolve("summary.csv"), output.resolve("figures/adaptive-beam-vs-fg-f1.png"));
  }

  private static List<LogisticCandidateScorer.TrainingExample> collect(
      Path data, int first, int last) throws IOException {
    List<LogisticCandidateScorer.TrainingExample> result = new ArrayList<>();
    for (int i = first; i <= last; i++) {
      LfrSpec spec = LfrSpec.paperTable()[i - 1];
      Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
      result.addAll(
          CandidateTrainingData.collect(
              graph, DatasetReader.communities(data.resolve(spec.name() + ".truth"), false), GAMMA));
    }
    return result;
  }

  private static int tune(Path data, CandidateScorer scorer, Path output) throws Exception {
    int bestBudget = BUDGETS[0];
    double best = Double.NEGATIVE_INFINITY;
    try (BufferedWriter writer = Files.newBufferedWriter(output.resolve("validation-sweep.csv"))) {
      writer.write("candidate_budget,mean_validation_f1\n");
      for (int budget : BUDGETS) {
        double total = 0;
        int count = 0;
        for (int index = 2; index < 5; index++) {
          LfrSpec spec = LfrSpec.paperTable()[index];
          Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
          List<Set<Integer>> truth =
              DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
          for (int seed : firstSeeds(graph, truth, 5)) {
            total +=
                Metrics.score(
                        new BeamFGDetector(graph, scorer, 2, 5, .10, budget)
                            .detect(seed, GAMMA, H),
                        Metrics.truthFor(seed, truth))
                    .f1();
            count++;
          }
        }
        double mean = count == 0 ? 0 : total / count;
        writer.write(String.format(Locale.ROOT, "%d,%.10f%n", budget, mean));
        if (mean > best) {
          best = mean;
          bestBudget = budget;
        }
      }
    }
    return bestBudget;
  }

  private static void run(Path data, Path output, CandidateScorer scorer, int budget, int limit)
      throws Exception {
    Path file = output.resolve("per-seed.csv");
    canonicalize(file);
    Set<String> completed = loadCompleted(file);
    boolean append = Files.exists(file) && Files.size(file) > 0;
    try (BufferedWriter writer =
        append
            ? Files.newBufferedWriter(file, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
            : Files.newBufferedWriter(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
      if (!append) writer.write("dataset,seed,method,recall,precision,f1,runtime_ms\n");
      for (int index = 5; index < 9; index++) {
        LfrSpec spec = LfrSpec.paperTable()[index];
        Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
        List<Set<Integer>> truth =
            DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
        List<Integer> seeds = firstSeeds(graph, truth, limit);
        ThreadLocal<DetectorContext> contexts =
            ThreadLocal.withInitial(() -> new DetectorContext(graph, scorer, budget));
        ExecutorService pool = Executors.newFixedThreadPool(Math.min(4, seeds.size()));
        List<Future<List<String>>> jobs = new ArrayList<>();
        for (int seed : seeds) {
          boolean complete = true;
          for (String method : List.of("FG", "BeamLogistic", "AdaptiveBeam"))
            if (!completed.contains(key(spec.name(), seed, method))) complete = false;
          if (!complete)
            jobs.add(
                pool.submit(
                    () -> evaluate(spec.name(), truth, seed, contexts.get())));
        }
        for (Future<List<String>> job : jobs)
          for (String row : job.get()) {
            writer.write(row);
            String[] parts = row.split(",", -1);
            completed.add(key(parts[0], Integer.parseInt(parts[1]), parts[2]));
          }
        writer.flush();
        pool.shutdown();
        System.out.println(spec.name() + ": processed " + jobs.size() + " seed jobs");
      }
    }
    canonicalize(file);
    summarize(file, output.resolve("summary.csv"));
  }

  private static List<String> evaluate(
      String dataset, List<Set<Integer>> truth, int seed, DetectorContext context) {
    Set<Integer> expected = Metrics.truthFor(seed, truth);
    List<String> rows = new ArrayList<>();
    long start = System.nanoTime();
    rows.add(row(dataset, seed, "FG", Metrics.score(context.fg.detect(seed, GAMMA, H), expected), start));
    start = System.nanoTime();
    rows.add(row(dataset, seed, "BeamLogistic", Metrics.score(
        context.fullBeam.detect(seed, GAMMA, H), expected), start));
    start = System.nanoTime();
    rows.add(row(dataset, seed, "AdaptiveBeam", Metrics.score(
        context.adaptiveBeam.detect(seed, GAMMA, H), expected), start));
    return rows;
  }

  private static String key(String dataset, int seed, String method) {
    return dataset + "\t" + seed + "\t" + method;
  }

  private static Set<String> loadCompleted(Path source) throws IOException {
    Set<String> completed = new HashSet<>();
    if (!Files.exists(source)) return completed;
    try (BufferedReader reader = Files.newBufferedReader(source)) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] parts = line.split(",", -1);
        if (parts.length != 7 || parts[0].equals("dataset")) continue;
        try {
          completed.add(key(parts[0], Integer.parseInt(parts[1]), parts[2]));
          Double.parseDouble(parts[6]);
        } catch (NumberFormatException ignored) {
          // Ignore an incomplete row left by an interrupted process.
        }
      }
    }
    return completed;
  }

  private static void canonicalize(Path source) throws IOException {
    if (!Files.exists(source) || Files.size(source) == 0) return;
    LinkedHashMap<String, String> rows = new LinkedHashMap<>();
    try (BufferedReader reader = Files.newBufferedReader(source)) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] parts = line.split(",", -1);
        if (parts.length != 7 || parts[0].equals("dataset")) continue;
        try {
          int seed = Integer.parseInt(parts[1]);
          for (int i = 3; i < 7; i++) Double.parseDouble(parts[i]);
          rows.put(key(parts[0], seed, parts[2]), line);
        } catch (NumberFormatException ignored) {
          // Ignore malformed or partially flushed rows.
        }
      }
    }
    Path cleaned = source.resolveSibling(source.getFileName() + ".cleaned");
    try (BufferedWriter writer = Files.newBufferedWriter(cleaned)) {
      writer.write("dataset,seed,method,recall,precision,f1,runtime_ms\n");
      for (String row : rows.values()) writer.write(row + "\n");
    }
    Files.move(cleaned, source, StandardCopyOption.REPLACE_EXISTING);
  }

  private static final class DetectorContext {
    private final FGDetector fg;
    private final BeamFGDetector fullBeam;
    private final BeamFGDetector adaptiveBeam;

    private DetectorContext(Graph graph, CandidateScorer scorer, int budget) {
      fg = new FGDetector(graph);
      fullBeam = new BeamFGDetector(graph, scorer, 2, 5, .10);
      adaptiveBeam = new BeamFGDetector(graph, scorer, 2, 5, .10, budget);
    }
  }

  private static List<Integer> firstSeeds(Graph graph, List<Set<Integer>> truth, int limit) {
    Set<Integer> present = new HashSet<>();
    for (Set<Integer> community : truth) present.addAll(community);
    List<Integer> result = new ArrayList<>();
    for (int node = 0; node < graph.size() && (limit <= 0 || result.size() < limit); node++)
      if (present.contains(node)) result.add(node);
    return result;
  }

  private static String row(String d, int seed, String method, Metrics.Score score, long start) {
    return String.format(Locale.ROOT, "%s,%d,%s,%.10f,%.10f,%.10f,%.3f%n", d, seed, method,
        score.recall(), score.precision(), score.f1(), (System.nanoTime() - start) / 1_000_000.0);
  }

  private static void summarize(Path source, Path target) throws IOException {
    Map<String, List<double[]>> grouped = new LinkedHashMap<>();
    try (BufferedReader reader = Files.newBufferedReader(source)) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] p = line.split(",");
        if (p[0].equals("dataset")) continue;
        grouped.computeIfAbsent(p[0] + "," + p[2], k -> new ArrayList<>())
            .add(new double[] {Double.parseDouble(p[3]), Double.parseDouble(p[4]),
                Double.parseDouble(p[5]), Double.parseDouble(p[6])});
      }
    }
    try (BufferedWriter writer = Files.newBufferedWriter(target)) {
      writer.write("dataset,method,seeds,recall,precision,f1,f1_std,runtime_ms\n");
      for (var entry : grouped.entrySet()) {
        String[] key = entry.getKey().split(",");
        List<double[]> values = entry.getValue();
        double recall = values.stream().mapToDouble(v -> v[0]).average().orElse(0);
        double precision = values.stream().mapToDouble(v -> v[1]).average().orElse(0);
        double f1 = values.stream().mapToDouble(v -> v[2]).average().orElse(0);
        double runtime = values.stream().mapToDouble(v -> v[3]).average().orElse(0);
        double variance = values.stream().mapToDouble(v -> (v[2] - f1) * (v[2] - f1)).average().orElse(0);
        writer.write(String.format(Locale.ROOT, "%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.3f%n",
            key[0], key[1], values.size(), recall, precision, f1, Math.sqrt(variance), runtime));
      }
    }
  }

  private static void figure(Path summary, Path target) throws IOException {
    Map<String, Integer> colors = Map.of("FG", 0, "BeamLogistic", 1, "AdaptiveBeam", 2);
    BufferedImage image = new BufferedImage(1100, 650, BufferedImage.TYPE_INT_ARGB);
    Graphics2D g = image.createGraphics();
    g.setColor(Color.WHITE); g.fillRect(0, 0, image.getWidth(), image.getHeight());
    g.setColor(Color.BLACK); g.setFont(new Font("SansSerif", Font.BOLD, 20));
    g.drawString("FG, full beam, and adaptive-budget beam F1", 100, 45);
    g.drawLine(90, 540, 1000, 540); g.drawLine(90, 80, 90, 540);
    Color[] palette = {Color.BLUE, Color.GREEN.darker(), Color.ORANGE.darker()};
    try (BufferedReader reader = Files.newBufferedReader(summary)) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] p = line.split(","); if (p[0].equals("dataset")) continue;
        int dataset = Integer.parseInt(p[0].substring(3));
        int method = colors.get(p[1]); int x = 120 + (dataset - 6) * 220 + method * 55;
        int height = (int) (Double.parseDouble(p[5]) * 400);
        g.setColor(palette[method]); g.fillRect(x, 540 - height, 42, height);
      }
    }
    Files.createDirectories(target.getParent()); ImageIO.write(image, "png", target.toFile());
    g.dispose();
  }
}
