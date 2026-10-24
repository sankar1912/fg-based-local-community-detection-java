package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/** Trains on LFR1-LFR5 and evaluates held-out LFR6-LFR9. */
public final class NoveltyExperiment {
  private static final double GAMMA = 0.45;
  private static final double H = 1.0;
  private static final double MODULARITY_SLACK = 0.05;

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args.length > 0 ? args[0] : "datasets/generated-lfr");
    Path output = Path.of(args.length > 1 ? args[1] : "output/novelty-results");
    int limit = args.length > 2 ? Integer.parseInt(args[2]) : 0;
    double membershipWeight = args.length > 3 ? Double.parseDouble(args[3]) : 0.5;
    double externalPenalty = args.length > 4 ? Double.parseDouble(args[4]) : 0.1;
    double minimumMembership = args.length > 5 ? Double.parseDouble(args[5]) : 0.60;
    double modularitySlack = args.length > 6 ? Double.parseDouble(args[6]) : MODULARITY_SLACK;
    Files.createDirectories(output);

    LogisticCandidateScorer scorer = train(data, output);
    runHeldOut(
        data,
        output,
        scorer,
        limit,
        membershipWeight,
        externalPenalty,
        minimumMembership,
        modularitySlack);
    NoveltyVisualization.main(
        new String[] {
          output.resolve("summary.csv").toString(), output.resolve("figures").toString()
        });
    System.out.println("Novelty results written to " + output.toAbsolutePath());
  }

  private static LogisticCandidateScorer train(Path data, Path output) throws Exception {
    List<LogisticCandidateScorer.TrainingExample> examples = new ArrayList<>();
    for (int index = 1; index <= 5; index++) {
      LfrSpec spec = LfrSpec.paperTable()[index - 1];
      Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
      List<Set<Integer>> truth =
          DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
      examples.addAll(CandidateTrainingData.collect(graph, truth, GAMMA));
    }
    LogisticCandidateScorer scorer = LogisticCandidateScorer.train(examples, 500, .08, .001);
    scorer.save(output.resolve("candidate-scorer.txt"));
    try (BufferedWriter writer = Files.newBufferedWriter(output.resolve("training-metadata.txt"))) {
      writer.write("training_graphs=LFR1,LFR2,LFR3,LFR4,LFR5\n");
      writer.write("training_examples=" + examples.size() + "\n");
      writer.write("label=majority of folded candidate nodes belongs to seed truth community\n");
      writer.write("gamma=" + GAMMA + "\n");
    }
    return scorer;
  }

  private static void runHeldOut(
      Path data,
      Path output,
      LogisticCandidateScorer scorer,
      int limit,
      double membershipWeight,
      double externalPenalty,
      double minimumMembership,
      double modularitySlack)
      throws Exception {
    Map<String, Summary> summaries = new LinkedHashMap<>();
    try (BufferedWriter rows = Files.newBufferedWriter(output.resolve("per-seed.csv"))) {
      rows.write("dataset,seed,method,found_size,recall,precision,f1\n");
      for (int index = 5; index < 9; index++) {
        LfrSpec spec = LfrSpec.paperTable()[index];
        Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
        List<Set<Integer>> truth =
            DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
        HybridFGDetector hybrid =
            new HybridFGDetector(
                graph,
                scorer,
                membershipWeight,
                externalPenalty,
                minimumMembership,
                modularitySlack);
        int count = limit <= 0 ? graph.size() : Math.min(limit, graph.size());
        ExecutorService executor =
            Executors.newFixedThreadPool(
                Math.max(1, Math.min(Runtime.getRuntime().availableProcessors(), 4)));
        List<Future<SeedResult>> futures = new ArrayList<>();
        for (int seed = 0; seed < count; seed++) {
          final int currentSeed = seed;
          futures.add(
              executor.submit(
                  () ->
                      evaluateSeed(
                          graph,
                          truth,
                          scorer,
                          currentSeed,
                          membershipWeight,
                          externalPenalty,
                          minimumMembership,
                          modularitySlack)));
        }
        Summary baseSummary = new Summary();
        Summary hybridSummary = new Summary();
        for (Future<SeedResult> future : futures) {
          SeedResult result = future.get();
          write(rows, spec.name(), result.seed(), "FG", result.base());
          write(rows, spec.name(), result.seed(), "Novelty", result.novelty());
          baseSummary.add(result.base());
          hybridSummary.add(result.novelty());
          rows.flush();
        }
        executor.shutdown();
        summaries.put(spec.name() + ",FG", baseSummary);
        summaries.put(spec.name() + ",Novelty", hybridSummary);
      }
    }
    try (BufferedWriter writer = Files.newBufferedWriter(output.resolve("novelty-config.txt"))) {
      writer.write("gamma=" + GAMMA + "\n");
      writer.write("h=" + H + "\n");
      writer.write("membership_weight=" + membershipWeight + "\n");
      writer.write("external_penalty=" + externalPenalty + "\n");
      writer.write("minimum_membership=" + minimumMembership + "\n");
      writer.write("modularity_slack=" + modularitySlack + "\n");
    }
    try (BufferedWriter summary = Files.newBufferedWriter(output.resolve("summary.csv"))) {
      summary.write("dataset,method,seeds,recall,precision,f1,f1_std\n");
      for (Map.Entry<String, Summary> entry : summaries.entrySet()) {
        String[] key = entry.getKey().split(",");
        Summary value = entry.getValue();
        summary.write(
            String.format(
                Locale.ROOT,
                "%s,%s,%d,%.8f,%.8f,%.8f,%.8f%n",
                key[0],
                key[1],
                value.count,
                value.recall(),
                value.precision(),
                value.f1(),
                value.f1Std()));
      }
    }
  }

  private static SeedResult evaluateSeed(
      Graph graph,
      List<Set<Integer>> truth,
      LogisticCandidateScorer scorer,
      int seed,
      double membershipWeight,
      double externalPenalty,
      double minimumMembership,
      double modularitySlack) {
    Set<Integer> expected = Metrics.truthFor(seed, truth);
    Metrics.Score base = Metrics.score(new FGDetector(graph).detect(seed, GAMMA, H), expected);
    HybridFGDetector hybrid =
        new HybridFGDetector(
            graph, scorer, membershipWeight, externalPenalty, minimumMembership, modularitySlack);
    Metrics.Score novelty = Metrics.score(hybrid.detect(seed, GAMMA, H), expected);
    return new SeedResult(seed, base, novelty);
  }

  private static void write(
      BufferedWriter writer, String dataset, int seed, String method, Metrics.Score score)
      throws IOException {
    writer.write(
        String.format(
            Locale.ROOT,
            "%s,%d,%s,,%.8f,%.8f,%.8f%n",
            dataset,
            seed,
            method,
            score.recall(),
            score.precision(),
            score.f1()));
  }

  private static final class Summary {
    private int count;
    private double recall;
    private double precision;
    private double f1;
    private double f1Squared;

    private void add(Metrics.Score score) {
      count++;
      recall += score.recall();
      precision += score.precision();
      f1 += score.f1();
      f1Squared += score.f1() * score.f1();
    }

    private double recall() {
      return recall / count;
    }

    private double precision() {
      return precision / count;
    }

    private double f1() {
      return f1 / count;
    }

    private double f1Std() {
      if (count < 2) return 0.0;
      return Math.sqrt(Math.max(0.0, f1Squared / count - f1() * f1()));
    }
  }

  private record SeedResult(int seed, Metrics.Score base, Metrics.Score novelty) {}
}
