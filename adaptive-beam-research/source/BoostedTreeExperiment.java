package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/** Validation-only tuning and held-out comparison for the nonlinear boosted-stump selector. */
public final class BoostedTreeExperiment {
  private static final double GAMMA = .45;
  private static final double H = 1.0;
  private static final double PENALTY = 0;
  private static final double[] WEIGHTS = {.2, .5, .8};
  private static final double[] SLACKS = {0, .05, .10};
  private static final double[] THRESHOLDS = {.6, .8};

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args.length > 0 ? args[0] : "datasets/generated-lfr");
    Path output = Path.of(args.length > 1 ? args[1] : "output/boosted-tree-results-5");
    int limit = args.length > 2 ? Integer.parseInt(args[2]) : 5;
    Files.createDirectories(output);
    List<LogisticCandidateScorer.TrainingExample> train = collect(data, 1, 2);
    Config config = tune(data, train, output);
    List<LogisticCandidateScorer.TrainingExample> finalData = collect(data, 1, 5);
    LogisticCandidateScorer logistic = LogisticCandidateScorer.train(finalData, 500, .08, .001);
    GradientBoostedCandidateScorer boosted =
        GradientBoostedCandidateScorer.train(finalData, 100, .15, .001);
    logistic.save(output.resolve("pointwise-logistic.txt"));
    boosted.save(output.resolve("gradient-boosted-model.txt"));
    Files.writeString(
        output.resolve("experiment-config.txt"),
        "training=LFR1-LFR5\n"
            + "validation=LFR3-LFR5 first five eligible seeds\n"
            + "held_out=LFR6-LFR9\n"
            + "selected_weight="
            + config.weight
            + "\nselected_slack="
            + config.slack
            + "\nselected_threshold="
            + config.threshold
            + "\nvalidation_f1="
            + config.validationF1
            + "\nmodel=gradient-boosted-stumps\nseed_limit="
            + limit
            + "\n");
    run(data, output, logistic, boosted, config, limit);
    BoostedTreeVisualization.main(new String[] {output.toString()});
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

  private static Config tune(
      Path data, List<LogisticCandidateScorer.TrainingExample> train, Path output)
      throws Exception {
    GradientBoostedCandidateScorer scorer =
        GradientBoostedCandidateScorer.train(train, 100, .15, .001);
    Config best = null;
    try (BufferedWriter writer = Files.newBufferedWriter(output.resolve("validation-sweep.csv"))) {
      writer.write("membership_weight,modularity_slack,minimum_membership,mean_validation_f1\n");
      for (double weight : WEIGHTS)
        for (double slack : SLACKS)
          for (double threshold : THRESHOLDS) {
            final double w = weight, s = slack, t = threshold;
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
                      () -> {
                        HybridFGDetector detector =
                            new HybridFGDetector(graph, scorer, w, PENALTY, t, s);
                        return Metrics.score(
                                detector.detect(current, GAMMA, H),
                                Metrics.truthFor(current, truth))
                            .f1();
                      });
                  count++;
                }
            }
            ExecutorService pool =
                Executors.newFixedThreadPool(
                    Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
            double total = 0;
            for (Future<Double> job : pool.invokeAll(jobs)) total += job.get();
            pool.shutdown();
            double mean = jobs.isEmpty() ? 0 : total / jobs.size();
            writer.write(
                String.format(
                    Locale.ROOT, "%.4f,%.4f,%.4f,%.10f%n", weight, slack, threshold, mean));
            Config candidate = new Config(weight, slack, threshold, mean);
            if (best == null || candidate.betterThan(best)) best = candidate;
          }
    }
    return best;
  }

  private static void run(
      Path data,
      Path output,
      CandidateScorer logistic,
      CandidateScorer boosted,
      Config config,
      int limit)
      throws Exception {
    try (BufferedWriter writer = Files.newBufferedWriter(output.resolve("per-seed.csv"))) {
      writer.write("dataset,seed,method,recall,precision,f1,runtime_ms,overrides\n");
      for (int index = 5; index < 9; index++) {
        LfrSpec spec = LfrSpec.paperTable()[index];
        Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
        List<Set<Integer>> truth =
            DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
        List<Integer> seeds = eligible(graph, truth);
        if (limit > 0 && seeds.size() > limit) seeds = new ArrayList<>(seeds.subList(0, limit));
        ExecutorService pool =
            Executors.newFixedThreadPool(
                Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
        List<Future<List<String>>> futures = new ArrayList<>();
        for (int seed : seeds)
          futures.add(
              pool.submit(
                  () -> evaluate(spec.name(), graph, truth, seed, logistic, boosted, config)));
        for (Future<List<String>> future : futures)
          for (String row : future.get()) writer.write(row);
        writer.flush();
        pool.shutdown();
        System.out.println(spec.name() + ": " + seeds.size() + " seeds");
      }
    }
    summarize(output.resolve("per-seed.csv"), output.resolve("summary.csv"));
  }

  private static List<String> evaluate(
      String dataset,
      Graph graph,
      List<Set<Integer>> truth,
      int seed,
      CandidateScorer logistic,
      CandidateScorer boosted,
      Config config) {
    Set<Integer> expected = Metrics.truthFor(seed, truth);
    List<String> rows = new ArrayList<>();
    long start = System.nanoTime();
    Metrics.Score fg = Metrics.score(new FGDetector(graph).detect(seed, GAMMA, H), expected);
    rows.add(row(dataset, seed, "FG", fg, elapsed(start), 0));
    start = System.nanoTime();
    HybridFGDetector logDetector =
        new HybridFGDetector(
            graph, logistic, config.weight, PENALTY, config.threshold, config.slack);
    Metrics.Score log = Metrics.score(logDetector.detect(seed, GAMMA, H), expected);
    rows.add(row(dataset, seed, "PointwiseLogistic", log, elapsed(start), overrides(logDetector)));
    start = System.nanoTime();
    HybridFGDetector treeDetector =
        new HybridFGDetector(
            graph, boosted, config.weight, PENALTY, config.threshold, config.slack);
    Metrics.Score tree = Metrics.score(treeDetector.detect(seed, GAMMA, H), expected);
    rows.add(row(dataset, seed, "BoostedStumps", tree, elapsed(start), overrides(treeDetector)));
    return rows;
  }

  private static List<Integer> eligible(Graph graph, List<Set<Integer>> truth) {
    Set<Integer> present = new HashSet<>();
    for (Set<Integer> c : truth) present.addAll(c);
    List<Integer> result = new ArrayList<>();
    for (int i = 0; i < graph.size(); i++) if (present.contains(i)) result.add(i);
    return result;
  }

  private static int overrides(HybridFGDetector detector) {
    return (int)
        detector.decisions().stream().filter(HybridModularityExpansion.Decision::accepted).count();
  }

  private static String row(
      String d, int seed, String method, Metrics.Score score, double runtime, int overrides) {
    return String.format(
        Locale.ROOT,
        "%s,%d,%s,%.10f,%.10f,%.10f,%.3f,%d%n",
        d,
        seed,
        method,
        score.recall(),
        score.precision(),
        score.f1(),
        runtime,
        overrides);
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
                    Double.parseDouble(p[6]),
                    Double.parseDouble(p[7])
                  });
      }
    }
    try (BufferedWriter writer = Files.newBufferedWriter(target)) {
      writer.write("dataset,method,seeds,recall,precision,f1,f1_std,runtime_ms,overrides\n");
      for (var e : grouped.entrySet()) {
        String[] key = e.getKey().split(",");
        List<double[]> v = e.getValue();
        double recall = v.stream().mapToDouble(x -> x[0]).average().orElse(0),
            precision = v.stream().mapToDouble(x -> x[1]).average().orElse(0),
            f1 = v.stream().mapToDouble(x -> x[2]).average().orElse(0),
            runtime = v.stream().mapToDouble(x -> x[3]).average().orElse(0),
            over = v.stream().mapToDouble(x -> x[4]).sum();
        double variance =
            v.size() < 2
                ? 0
                : v.stream().mapToDouble(x -> (x[2] - f1) * (x[2] - f1)).sum() / v.size();
        writer.write(
            String.format(
                Locale.ROOT,
                "%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.3f,%.0f%n",
                key[0],
                key[1],
                v.size(),
                recall,
                precision,
                f1,
                Math.sqrt(variance),
                runtime,
                over));
      }
    }
  }

  private record Config(double weight, double slack, double threshold, double validationF1) {
    boolean betterThan(Config other) {
      return validationF1 > other.validationF1 + 1e-12
          || (Math.abs(validationF1 - other.validationF1) < 1e-12
              && (weight < other.weight
                  || (weight == other.weight
                      && (slack < other.slack
                          || (slack == other.slack && threshold < other.threshold)))));
    }
  }
}
