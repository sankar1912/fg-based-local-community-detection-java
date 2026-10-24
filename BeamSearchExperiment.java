package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/** Validation-tuned beam-search experiment for the greedy-expansion flaw. */
public final class BeamSearchExperiment {
  private static final double GAMMA = .45;
  private static final double H = 1.0;
  private static final double[] LEARNED_WEIGHTS = {0.0, .05, .10};
  private static final int[] BEAM_WIDTHS = {2, 3};
  private static final int[] DEPTHS = {3, 5};

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args.length > 0 ? args[0] : "datasets/generated-lfr");
    Path output = Path.of(args.length > 1 ? args[1] : "output/beam-search-results-5");
    int limit = args.length > 2 ? Integer.parseInt(args[2]) : 5;
    Files.createDirectories(output);
    List<LogisticCandidateScorer.TrainingExample> train = collect(data, 1, 5);
    LogisticCandidateScorer scorer = LogisticCandidateScorer.train(train, 500, .08, .001);
    Config config = tune(data, scorer, output);
    scorer.save(output.resolve("logistic-scorer.txt"));
    Files.writeString(
        output.resolve("experiment-config.txt"),
        "training=LFR1-LFR5\n"
            + "validation=LFR3-LFR5 first five eligible seeds\n"
            + "held_out=LFR6-LFR9\n"
            + "beam_width="
            + config.width
            + "\nmax_depth="
            + config.depth
            + "\nlearned_weight="
            + config.learnedWeight
            + "\nvalidation_f1="
            + config.validationF1
            + "\nseed_limit="
            + limit
            + "\n");
    run(data, output, scorer, config, limit);
    BeamSearchVisualization.main(new String[] {output.toString()});
  }

  private static List<LogisticCandidateScorer.TrainingExample> collect(
      Path data, int first, int last) throws IOException {
    List<LogisticCandidateScorer.TrainingExample> result = new ArrayList<>();
    LfrSpec[] table = LfrSpec.paperTable();
    for (int i = first; i <= last; i++) {
      LfrSpec spec = table[i - 1];
      Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
      result.addAll(
          CandidateTrainingData.collect(
              graph,
              DatasetReader.communities(data.resolve(spec.name() + ".truth"), false),
              GAMMA));
    }
    return result;
  }

  private static Config tune(Path data, CandidateScorer scorer, Path output) throws Exception {
    Config best = null;
    try (BufferedWriter writer = Files.newBufferedWriter(output.resolve("validation-sweep.csv"))) {
      writer.write("beam_width,max_depth,learned_weight,mean_validation_f1\n");
      for (int width : BEAM_WIDTHS)
        for (int depth : DEPTHS)
          for (double learnedWeight : LEARNED_WEIGHTS) {
            final int w = width, d = depth;
            final double lw = learnedWeight;
            List<Callable<Double>> jobs = new ArrayList<>();
            for (int index = 2; index < 5; index++) {
              LfrSpec spec = LfrSpec.paperTable()[index];
              Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
              List<Set<Integer>> truth =
                  DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
              Set<Integer> present = new HashSet<>();
              for (Set<Integer> c : truth) present.addAll(c);
              int count = 0;
              for (int seed = 0; seed < graph.size() && count < 5; seed++)
                if (present.contains(seed)) {
                  int current = seed;
                  jobs.add(
                      () ->
                          Metrics.score(
                                  new BeamFGDetector(graph, scorer, w, d, lw)
                                      .detect(current, GAMMA, H),
                                  Metrics.truthFor(current, truth))
                              .f1());
                  count++;
                }
            }
            ExecutorService pool =
                Executors.newFixedThreadPool(
                    Math.max(1, Math.min(12, Runtime.getRuntime().availableProcessors())));
            double total = 0;
            for (Future<Double> job : pool.invokeAll(jobs)) total += job.get();
            pool.shutdown();
            double mean = jobs.isEmpty() ? 0 : total / jobs.size();
            writer.write(
                String.format(
                    Locale.ROOT, "%d,%d,%.4f,%.10f%n", width, depth, learnedWeight, mean));
            Config candidate = new Config(width, depth, learnedWeight, mean);
            if (best == null || candidate.betterThan(best)) best = candidate;
          }
    }
    return best;
  }

  private static void run(Path data, Path output, CandidateScorer scorer, Config config, int limit)
      throws Exception {
    Path perSeed = output.resolve("per-seed.csv");
    canonicalize(perSeed);
    Set<String> completed = loadCompleted(perSeed);
    boolean append = Files.exists(perSeed) && Files.size(perSeed) > 0;
    try (BufferedWriter writer =
        append
            ? Files.newBufferedWriter(perSeed, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
            : Files.newBufferedWriter(perSeed, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
      if (!append) writer.write("dataset,seed,method,recall,precision,f1,runtime_ms\n");
      for (int index = 5; index < 9; index++) {
        LfrSpec spec = LfrSpec.paperTable()[index];
        Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
        List<Set<Integer>> truth =
            DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
        List<Integer> seeds = eligible(graph, truth);
        if (limit > 0 && seeds.size() > limit) seeds = new ArrayList<>(seeds.subList(0, limit));
        ExecutorService pool =
            Executors.newFixedThreadPool(
                Math.max(1, Math.min(12, Runtime.getRuntime().availableProcessors())));
        List<Future<List<String>>> futures = new ArrayList<>();
        for (int seed : seeds) {
          boolean complete = true;
          for (String method : List.of("FG", "GreedyLogistic", "BeamLogistic"))
            if (!completed.contains(key(spec.name(), seed, method))) complete = false;
          if (!complete)
            futures.add(
                pool.submit(() -> evaluate(spec.name(), graph, truth, seed, scorer, config, completed)));
        }
        for (Future<List<String>> future : futures)
          for (String row : future.get()) {
            writer.write(row);
            String[] parts = row.split(",", -1);
            completed.add(key(parts[0], Integer.parseInt(parts[1]), parts[2]));
          }
        writer.flush();
        pool.shutdown();
        System.out.println(spec.name() + ": processed " + futures.size() + " seed jobs");
      }
    }
    canonicalize(perSeed);
    summarize(output.resolve("per-seed.csv"), output.resolve("summary.csv"));
  }

  private static List<String> evaluate(
      String dataset,
      Graph graph,
      List<Set<Integer>> truth,
      int seed,
      CandidateScorer scorer,
      Config config,
      Set<String> completed) {
    Set<Integer> expected = Metrics.truthFor(seed, truth);
    List<String> rows = new ArrayList<>();
    if (!completed.contains(key(dataset, seed, "FG"))) {
      long start = System.nanoTime();
      Metrics.Score fg = Metrics.score(new FGDetector(graph).detect(seed, GAMMA, H), expected);
      rows.add(row(dataset, seed, "FG", fg, elapsed(start)));
    }
    if (!completed.contains(key(dataset, seed, "GreedyLogistic"))) {
      long start = System.nanoTime();
      Metrics.Score greedy =
          Metrics.score(
              new HybridFGDetector(graph, scorer, .2, 0, .6, .05).detect(seed, GAMMA, H), expected);
      rows.add(row(dataset, seed, "GreedyLogistic", greedy, elapsed(start)));
    }
    if (!completed.contains(key(dataset, seed, "BeamLogistic"))) {
      long start = System.nanoTime();
      Metrics.Score beam =
          Metrics.score(
              new BeamFGDetector(graph, scorer, config.width, config.depth, config.learnedWeight)
                  .detect(seed, GAMMA, H),
              expected);
      rows.add(row(dataset, seed, "BeamLogistic", beam, elapsed(start)));
    }
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
          Integer.parseInt(parts[1]);
          Double.parseDouble(parts[5]);
          completed.add(key(parts[0], Integer.parseInt(parts[1]), parts[2]));
        } catch (NumberFormatException ignored) {
          // Ignore an incomplete line left by an interrupted run.
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
          Double.parseDouble(parts[3]);
          Double.parseDouble(parts[4]);
          Double.parseDouble(parts[5]);
          Double.parseDouble(parts[6]);
          rows.put(key(parts[0], seed, parts[2]), line);
        } catch (NumberFormatException ignored) {
          // Ignore incomplete or malformed rows from an interrupted process.
        }
      }
    }
    Path cleaned = source.resolveSibling(source.getFileName() + ".cleaned");
    try (BufferedWriter writer = Files.newBufferedWriter(cleaned)) {
      writer.write("dataset,seed,method,recall,precision,f1,runtime_ms\n");
      for (String line : rows.values()) writer.write(line + "\n");
    }
    Files.move(cleaned, source, StandardCopyOption.REPLACE_EXISTING);
  }

  private static List<Integer> eligible(Graph graph, List<Set<Integer>> truth) {
    Set<Integer> present = new HashSet<>();
    for (Set<Integer> c : truth) present.addAll(c);
    List<Integer> result = new ArrayList<>();
    for (int i = 0; i < graph.size(); i++) if (present.contains(i)) result.add(i);
    return result;
  }

  private static String row(
      String d, int seed, String method, Metrics.Score score, double runtime) {
    return String.format(
        Locale.ROOT,
        "%s,%d,%s,%.10f,%.10f,%.10f,%.3f%n",
        d,
        seed,
        method,
        score.recall(),
        score.precision(),
        score.f1(),
        runtime);
  }

  private static double elapsed(long start) {
    return (System.nanoTime() - start) / 1_000_000.0;
  }

  private static void summarize(Path source, Path target) throws IOException {
    Map<String, List<double[]>> grouped = new LinkedHashMap<>();
    try (BufferedReader reader = Files.newBufferedReader(source)) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] p = line.split(",");
        if (!p[0].equals("dataset"))
          grouped
              .computeIfAbsent(p[0] + "," + p[2], k -> new ArrayList<>())
              .add(
                  new double[] {
                    Double.parseDouble(p[3]),
                    Double.parseDouble(p[4]),
                    Double.parseDouble(p[5]),
                    Double.parseDouble(p[6])
                  });
      }
    }
    try (BufferedWriter writer = Files.newBufferedWriter(target)) {
      writer.write("dataset,method,seeds,recall,precision,f1,f1_std,runtime_ms\n");
      for (var e : grouped.entrySet()) {
        String[] key = e.getKey().split(",");
        List<double[]> v = e.getValue();
        double recall = v.stream().mapToDouble(x -> x[0]).average().orElse(0),
            precision = v.stream().mapToDouble(x -> x[1]).average().orElse(0),
            f1 = v.stream().mapToDouble(x -> x[2]).average().orElse(0),
            runtime = v.stream().mapToDouble(x -> x[3]).average().orElse(0);
        double variance =
            v.size() < 2
                ? 0
                : v.stream().mapToDouble(x -> (x[2] - f1) * (x[2] - f1)).sum() / v.size();
        writer.write(
            String.format(
                Locale.ROOT,
                "%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.3f%n",
                key[0],
                key[1],
                v.size(),
                recall,
                precision,
                f1,
                Math.sqrt(variance),
                runtime));
      }
    }
  }

  private record Config(int width, int depth, double learnedWeight, double validationF1) {
    boolean betterThan(Config other) {
      return validationF1 > other.validationF1 + 1e-12
          || (Math.abs(validationF1 - other.validationF1) < 1e-12
              && (width < other.width
                  || (width == other.width
                      && (depth < other.depth
                          || (depth == other.depth && learnedWeight < other.learnedWeight)))));
    }
  }
}
