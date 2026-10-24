package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/** Held-out comparison for the pairwise query-conditioned ranking novelty. */
public final class PairwiseRankingExperiment {
  private static final double GAMMA = .45;
  private static final double H = 1.0;
  private static final double PENALTY = 0.0;
  private static final double[] WEIGHTS = {.2, .5, .8};
  private static final double[] SLACKS = {0.0, .05, .10};
  private static final double[] THRESHOLDS = {.6, .8};

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args.length > 0 ? args[0] : "datasets/generated-lfr");
    Path output = Path.of(args.length > 1 ? args[1] : "output/pairwise-ranking-results-5");
    int seedLimit = args.length > 2 ? Integer.parseInt(args[2]) : 5;
    Files.createDirectories(output);
    List<List<LogisticCandidateScorer.TrainingExample>> groups = collectGroups(data, 1, 5);
    List<LogisticCandidateScorer.TrainingExample> examples = new ArrayList<>();
    for (List<LogisticCandidateScorer.TrainingExample> group : groups) examples.addAll(group);
    LogisticCandidateScorer pointwise = LogisticCandidateScorer.train(examples, 500, .08, .001);
    PairwiseCandidateScorer pairwise = PairwiseCandidateScorer.train(groups, 500, .04, .001);
    Config config = tune(data, pairwise, output);
    pointwise.save(output.resolve("pointwise-logistic.txt"));
    pairwise.save(output.resolve("pairwise-ranking-model.txt"));
    Files.writeString(
        output.resolve("experiment-config.txt"),
        "training=LFR1-LFR5\n"
            + "validation=LFR3-LFR5 first five eligible seeds\n"
            + "held_out=LFR6-LFR9\n"
            + "model=pairwise-logistic-ranking\n"
            + "membership_weight="
            + config.weight
            + "\nexternal_penalty="
            + PENALTY
            + "\nminimum_membership="
            + config.threshold
            + "\nmodularity_slack="
            + config.slack
            + "\nvalidation_f1="
            + config.validationF1
            + "\nseed_limit="
            + seedLimit
            + "\n");
    run(data, output, pointwise, pairwise, config, seedLimit);
    PairwiseRankingVisualization.main(new String[] {output.toString()});
    System.out.println("Pairwise ranking results written to " + output.toAbsolutePath());
  }

  private static Config tune(Path data, CandidateScorer pairwise, Path output) throws Exception {
    Config best = null;
    try (BufferedWriter writer = Files.newBufferedWriter(output.resolve("validation-sweep.csv"))) {
      writer.write("membership_weight,modularity_slack,minimum_membership,mean_validation_f1\n");
      for (double weight : WEIGHTS)
        for (double slack : SLACKS)
          for (double threshold : THRESHOLDS) {
            final double selectedWeight = weight;
            final double selectedSlack = slack;
            final double selectedThreshold = threshold;
            List<Callable<Double>> jobs = new ArrayList<>();
            for (int index = 2; index < 5; index++) {
              LfrSpec spec = LfrSpec.paperTable()[index];
              Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
              List<Set<Integer>> truth =
                  DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
              Set<Integer> present = new HashSet<>();
              for (Set<Integer> community : truth) present.addAll(community);
              int count = 0;
              for (int seed = 0; seed < graph.size() && count < 5; seed++) {
                if (!present.contains(seed)) continue;
                int current = seed;
                jobs.add(
                    () -> {
                      HybridFGDetector detector =
                          new HybridFGDetector(
                              graph,
                              pairwise,
                              selectedWeight,
                              PENALTY,
                              selectedThreshold,
                              selectedSlack);
                      return Metrics.score(
                              detector.detect(current, GAMMA, H), Metrics.truthFor(current, truth))
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

  private static List<List<LogisticCandidateScorer.TrainingExample>> collectGroups(
      Path data, int first, int last) throws IOException {
    List<List<LogisticCandidateScorer.TrainingExample>> result = new ArrayList<>();
    LfrSpec[] table = LfrSpec.paperTable();
    for (int i = first; i <= last; i++) {
      LfrSpec spec = table[i - 1];
      Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
      List<Set<Integer>> truth =
          DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
      result.addAll(CandidateTrainingData.collectGroups(graph, truth, GAMMA));
    }
    return result;
  }

  private static void run(
      Path data,
      Path output,
      CandidateScorer pointwise,
      CandidateScorer pairwise,
      Config config,
      int seedLimit)
      throws Exception {
    try (BufferedWriter rows = Files.newBufferedWriter(output.resolve("per-seed.csv"))) {
      rows.write("dataset,seed,method,recall,precision,f1,runtime_ms,overrides\n");
      for (int index = 5; index < 9; index++) {
        LfrSpec spec = LfrSpec.paperTable()[index];
        Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
        List<Set<Integer>> truth =
            DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
        List<Integer> seeds = new ArrayList<>();
        Set<Integer> present = new HashSet<>();
        for (Set<Integer> community : truth) present.addAll(community);
        for (int seed = 0; seed < graph.size(); seed++) if (present.contains(seed)) seeds.add(seed);
        if (seedLimit > 0 && seeds.size() > seedLimit)
          seeds = new ArrayList<>(seeds.subList(0, seedLimit));
        ExecutorService pool =
            Executors.newFixedThreadPool(
                Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
        List<Future<List<String>>> futures = new ArrayList<>();
        for (int seed : seeds)
          futures.add(
              pool.submit(
                  () -> evaluate(spec.name(), graph, truth, seed, pointwise, pairwise, config)));
        for (Future<List<String>> future : futures)
          for (String line : future.get()) rows.write(line);
        rows.flush();
        pool.shutdown();
        System.out.println(spec.name() + ": " + seeds.size() + " seeds");
      }
    }
    writeSummary(output.resolve("per-seed.csv"), output.resolve("summary.csv"));
  }

  private static List<String> evaluate(
      String dataset,
      Graph graph,
      List<Set<Integer>> truth,
      int seed,
      CandidateScorer pointwise,
      CandidateScorer pairwise,
      Config config) {
    Set<Integer> expected = Metrics.truthFor(seed, truth);
    List<String> result = new ArrayList<>();
    long start = System.nanoTime();
    Metrics.Score base = Metrics.score(new FGDetector(graph).detect(seed, GAMMA, H), expected);
    result.add(row(dataset, seed, "FG", base, elapsed(start), 0));
    start = System.nanoTime();
    HybridFGDetector pointwiseDetector =
        new HybridFGDetector(
            graph, pointwise, config.weight, PENALTY, config.threshold, config.slack);
    Metrics.Score pointwiseScore =
        Metrics.score(pointwiseDetector.detect(seed, GAMMA, H), expected);
    result.add(
        row(
            dataset,
            seed,
            "PointwiseLogistic",
            pointwiseScore,
            elapsed(start),
            countOverrides(pointwiseDetector)));
    start = System.nanoTime();
    HybridFGDetector pairwiseDetector =
        new HybridFGDetector(
            graph, pairwise, config.weight, PENALTY, config.threshold, config.slack);
    Metrics.Score pairwiseScore = Metrics.score(pairwiseDetector.detect(seed, GAMMA, H), expected);
    result.add(
        row(
            dataset,
            seed,
            "PairwiseRank",
            pairwiseScore,
            elapsed(start),
            countOverrides(pairwiseDetector)));
    return result;
  }

  private static int countOverrides(HybridFGDetector detector) {
    return (int)
        detector.decisions().stream().filter(HybridModularityExpansion.Decision::accepted).count();
  }

  private static String row(
      String dataset, int seed, String method, Metrics.Score score, double runtime, int overrides) {
    return String.format(
        Locale.ROOT,
        "%s,%d,%s,%.10f,%.10f,%.10f,%.3f,%d%n",
        dataset,
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

  private static void writeSummary(Path perSeed, Path summary) throws IOException {
    Map<String, List<double[]>> values = new LinkedHashMap<>();
    try (BufferedReader reader = Files.newBufferedReader(perSeed)) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] p = line.split(",");
        if (!p[0].equals("dataset"))
          values
              .computeIfAbsent(p[0] + "," + p[2], ignored -> new ArrayList<>())
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
    try (BufferedWriter writer = Files.newBufferedWriter(summary)) {
      writer.write("dataset,method,seeds,recall,precision,f1,f1_std,runtime_ms,overrides\n");
      for (Map.Entry<String, List<double[]>> entry : values.entrySet()) {
        String[] key = entry.getKey().split(",");
        List<double[]> rows = entry.getValue();
        double recall = rows.stream().mapToDouble(x -> x[0]).average().orElse(0),
            precision = rows.stream().mapToDouble(x -> x[1]).average().orElse(0),
            f1 = rows.stream().mapToDouble(x -> x[2]).average().orElse(0),
            runtime = rows.stream().mapToDouble(x -> x[3]).average().orElse(0),
            overrides = rows.stream().mapToDouble(x -> x[4]).sum();
        double variance =
            rows.size() < 2
                ? 0
                : rows.stream().mapToDouble(x -> (x[2] - f1) * (x[2] - f1)).sum() / rows.size();
        writer.write(
            String.format(
                Locale.ROOT,
                "%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.3f,%.0f%n",
                key[0],
                key[1],
                rows.size(),
                recall,
                precision,
                f1,
                Math.sqrt(variance),
                runtime,
                overrides));
      }
    }
  }
}
