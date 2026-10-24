package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;

/** Reproducible validation-only tuning and full held-out comparison. */
public final class RigorousEvaluation {
  private static final double GAMMA = .45;
  private static final double H = 1.0;
  private static final long RANDOM_SEED = 20250721L;
  private static final int VALIDATION_SEEDS_PER_DATASET = 5;
  private static final String[] MODELS = {"FG", "Logistic", "MLP", "RBM"};
  private static final double[] WEIGHTS = {.2, .5, .8};
  private static final double[] SLACKS = {0.0, .05, .10};
  private static final double[] THRESHOLDS = {.6, .8};

  public static void main(String[] args) throws Exception {
    Path data = Path.of(args.length > 0 ? args[0] : "datasets/generated-lfr");
    Path output = Path.of(args.length > 1 ? args[1] : "output/rigorous-evaluation");
    int seedLimit = args.length > 2 ? Integer.parseInt(args[2]) : 0;
    Files.createDirectories(output);
    long start = System.currentTimeMillis();
    writeManifest(output, data, "STARTED", null, null);

    List<LogisticCandidateScorer.TrainingExample> trainExamples = collect(data, 1, 2);
    List<LogisticCandidateScorer.TrainingExample> validationExamples = collect(data, 3, 5);
    Config selected = tune(data, output, trainExamples, validationExamples);
    writeSelectedConfig(output, selected);

    List<LogisticCandidateScorer.TrainingExample> finalExamples = collect(data, 1, 5);
    Map<String, CandidateScorer> models = trainModels(finalExamples);
    writeTrainingMetadata(output, trainExamples, validationExamples, finalExamples, models);
    runHeldOut(data, output, models, selected, seedLimit);
    deriveArtifacts(data, output, selected, models);
    RigorousVisualization.main(new String[] {output.toString()});
    writeManifest(output, data, "COMPLETED", selected, models.keySet());
    System.out.printf(
        Locale.ROOT,
        "Rigorous evaluation completed in %.1f minutes: %s%n",
        (System.currentTimeMillis() - start) / 60000.0,
        output.toAbsolutePath());
  }

  private static List<LogisticCandidateScorer.TrainingExample> collect(
      Path data, int first, int last) throws IOException {
    List<LogisticCandidateScorer.TrainingExample> result = new ArrayList<>();
    LfrSpec[] table = LfrSpec.paperTable();
    for (int i = first; i <= last; i++) {
      LfrSpec spec = table[i - 1];
      Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
      List<Set<Integer>> truth =
          DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
      result.addAll(CandidateTrainingData.collect(graph, truth, GAMMA));
    }
    return result;
  }

  private static Config tune(
      Path data,
      Path output,
      List<LogisticCandidateScorer.TrainingExample> train,
      List<LogisticCandidateScorer.TrainingExample> ignoredValidationExamples)
      throws Exception {
    Path sweep = output.resolve("validation_sweep.csv");
    try (BufferedWriter writer = Files.newBufferedWriter(sweep)) {
      writer.write("membership_weight,modularity_slack,minimum_membership,mean_validation_f1\n");
      LogisticCandidateScorer scorer = LogisticCandidateScorer.train(train, 500, .08, .001);
      Config best = null;
      for (double weight : WEIGHTS)
        for (double slack : SLACKS)
          for (double threshold : THRESHOLDS) {
            List<Callable<Double>> jobs = new ArrayList<>();
            final double selectedWeight = weight;
            final double selectedSlack = slack;
            final double selectedThreshold = threshold;
            for (int i = 3; i <= 5; i++) {
              LfrSpec spec = LfrSpec.paperTable()[i - 1];
              Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
              List<Set<Integer>> truth =
                  DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
              List<Integer> seeds = eligibleSeeds(graph, truth);
              int amount = Math.min(VALIDATION_SEEDS_PER_DATASET, seeds.size());
              for (int j = 0; j < amount; j++) {
                int seed = seeds.get(j);
                jobs.add(
                    () -> {
                      HybridFGDetector detector =
                          new HybridFGDetector(
                              graph, scorer, selectedWeight, 0, selectedThreshold, selectedSlack);
                      return Metrics.score(
                              detector.detect(seed, GAMMA, H), Metrics.truthFor(seed, truth))
                          .f1();
                    });
              }
            }
            ExecutorService validationPool =
                Executors.newFixedThreadPool(
                    Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
            double total = 0;
            for (Future<Double> job : validationPool.invokeAll(jobs)) total += job.get();
            validationPool.shutdown();
            double mean = jobs.isEmpty() ? 0 : total / jobs.size();
            writer.write(
                String.format(
                    Locale.ROOT, "%.4f,%.4f,%.4f,%.10f%n", weight, slack, threshold, mean));
            Config candidate = new Config(weight, 0, threshold, slack, mean);
            if (best == null || candidate.betterThan(best)) best = candidate;
          }
      return best;
    }
  }

  private static Map<String, CandidateScorer> trainModels(
      List<LogisticCandidateScorer.TrainingExample> examples) {
    Map<String, CandidateScorer> models = new LinkedHashMap<>();
    models.put("Logistic", LogisticCandidateScorer.train(examples, 500, .08, .001));
    models.put("MLP", MlpCandidateScorer.train(examples, 12, 150, .02, .001, RANDOM_SEED));
    models.put("RBM", RbmCandidateScorer.train(examples, 12, 40, .01, .0001, RANDOM_SEED));
    return models;
  }

  private static void runHeldOut(
      Path data, Path output, Map<String, CandidateScorer> models, Config config, int seedLimit)
      throws Exception {
    Path perSeed = output.resolve("per_seed_results.csv");
    Path audit = output.resolve("override_audit.csv");
    Set<String> completed = completedRows(perSeed);
    boolean newRows = !Files.exists(perSeed) || Files.size(perSeed) == 0;
    try (BufferedWriter rows =
            Files.newBufferedWriter(perSeed, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        BufferedWriter auditWriter =
            Files.newBufferedWriter(audit, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
      if (newRows)
        rows.write(
            "dataset,graph_id,seed_id,algorithm,model_type,F1,precision,recall,modularity_gain,selected_candidate_id,learned_override,override_reason,runtime_ms,status,failure_reason\n");
      if (!Files.exists(audit) || Files.size(audit) == 0)
        auditWriter.write(
            "dataset,graph_id,seed_id,algorithm,step,original_fg_candidate,proposed_candidate,original_delta,proposed_delta,compatibility,threshold,slack,accepted,reason\n");
      for (int index = 5; index < 9; index++) {
        LfrSpec spec = LfrSpec.paperTable()[index];
        Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
        List<Set<Integer>> truth =
            DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
        List<Integer> seeds = eligibleSeeds(graph, truth);
        if (seedLimit > 0 && seeds.size() > seedLimit)
          seeds = new ArrayList<>(seeds.subList(0, seedLimit));
        ExecutorService pool =
            Executors.newFixedThreadPool(
                Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors())));
        List<Future<List<Result>>> futures = new ArrayList<>();
        for (int seed : seeds)
          futures.add(
              pool.submit(() -> evaluateSeed(spec.name(), graph, truth, seed, models, config)));
        int done = 0;
        for (Future<List<Result>> future : futures) {
          for (Result result : future.get()) {
            String key = result.dataset + "|" + result.seed + "|" + result.algorithm;
            if (!completed.contains(key)) {
              rows.write(result.csv());
              completed.add(key);
            }
            for (String line : result.auditLines) auditWriter.write(line);
          }
          done++;
          if (done % 25 == 0 || done == futures.size()) {
            System.out.printf("%s: %d/%d eligible seeds%n", spec.name(), done, futures.size());
            rows.flush();
            auditWriter.flush();
          }
        }
        pool.shutdown();
      }
    }
  }

  private static List<Result> evaluateSeed(
      String dataset,
      Graph graph,
      List<Set<Integer>> truth,
      int seed,
      Map<String, CandidateScorer> models,
      Config config) {
    List<Result> result = new ArrayList<>();
    Set<Integer> expected = Metrics.truthFor(seed, truth);
    long start = System.nanoTime();
    try {
      Set<Integer> found = new FGDetector(graph).detect(seed, GAMMA, H);
      Metrics.Score score = Metrics.score(found, expected);
      result.add(
          Result.success(
              dataset, seed, "FG", "none", score, 0, "", false, "baseline", elapsed(start)));
    } catch (Exception e) {
      result.add(Result.failure(dataset, seed, "FG", "none", e));
    }
    for (Map.Entry<String, CandidateScorer> entry : models.entrySet()) {
      start = System.nanoTime();
      try {
        HybridFGDetector detector =
            new HybridFGDetector(
                graph,
                entry.getValue(),
                config.weight,
                config.penalty,
                config.threshold,
                config.slack);
        Set<Integer> found = detector.detect(seed, GAMMA, H);
        Metrics.Score score = Metrics.score(found, expected);
        List<HybridModularityExpansion.Decision> decisions = detector.decisions();
        String candidates =
            decisions.stream()
                .map(HybridModularityExpansion.Decision::proposedCandidate)
                .collect(java.util.stream.Collectors.joining("|"));
        boolean override =
            decisions.stream().anyMatch(HybridModularityExpansion.Decision::accepted);
        String reason = override ? "accepted_safe_override" : "fg_candidate_retained";
        List<String> audit = new ArrayList<>();
        int step = 0;
        for (HybridModularityExpansion.Decision d : decisions) {
          audit.add(
              String.format(
                  Locale.ROOT,
                  "%s,%s,%d,%s,%d,%s,%s,%.12f,%.12f,%.12f,%.4f,%.4f,%s,%s%n",
                  dataset,
                  dataset,
                  seed,
                  entry.getKey(),
                  step++,
                  d.originalFgCandidate(),
                  d.proposedCandidate(),
                  d.originalDelta(),
                  d.proposedDelta(),
                  d.compatibility(),
                  d.threshold(),
                  d.slack(),
                  d.accepted(),
                  d.reason()));
        }
        result.add(
            Result.success(
                dataset,
                seed,
                entry.getKey(),
                entry.getValue().modelType(),
                score,
                detector.selectedGainTotal(),
                candidates,
                override,
                reason,
                elapsed(start),
                audit));
      } catch (Exception e) {
        result.add(Result.failure(dataset, seed, entry.getKey(), entry.getValue().modelType(), e));
      }
    }
    return result;
  }

  private static List<Integer> eligibleSeeds(Graph graph, List<Set<Integer>> truth) {
    Set<Integer> present = new HashSet<>();
    for (Set<Integer> community : truth) present.addAll(community);
    List<Integer> result = new ArrayList<>();
    for (int node = 0; node < graph.size(); node++) if (present.contains(node)) result.add(node);
    return result;
  }

  private static Set<String> completedRows(Path file) throws IOException {
    Set<String> result = new HashSet<>();
    if (!Files.exists(file)) return result;
    try (BufferedReader reader = Files.newBufferedReader(file)) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] p = line.split(",", -1);
        if (p.length >= 5 && !p[0].equals("dataset") && "SUCCESS".equals(p[13]))
          result.add(p[0] + "|" + p[2] + "|" + p[3]);
      }
    }
    return result;
  }

  private static void deriveArtifacts(
      Path data, Path output, Config config, Map<String, CandidateScorer> models)
      throws IOException {
    List<Row> rows = readRows(output.resolve("per_seed_results.csv"));
    List<String> algorithms = List.of("Logistic", "MLP", "RBM");
    try (BufferedWriter paired =
        Files.newBufferedWriter(output.resolve("paired_seed_results.csv"))) {
      paired.write(
          "dataset,graph_id,seed_id,model,FG_F1,hybrid_F1,paired_F1_difference,FG_precision,hybrid_precision,FG_recall,hybrid_recall,FG_runtime_ms,hybrid_runtime_ms\n");
      for (String dataset : new TreeSet<>(rows.stream().map(r -> r.dataset).toList()))
        for (int seed :
            new TreeSet<>(
                rows.stream().filter(r -> r.dataset.equals(dataset)).map(r -> r.seed).toList())) {
          Row fg = find(rows, dataset, seed, "FG");
          if (fg == null || !fg.success) continue;
          for (String model : algorithms) {
            Row h = find(rows, dataset, seed, model);
            if (h != null && h.success)
              paired.write(
                  String.format(
                      Locale.ROOT,
                      "%s,%s,%d,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.3f,%.3f%n",
                      dataset,
                      dataset,
                      seed,
                      model,
                      fg.f1,
                      h.f1,
                      h.f1 - fg.f1,
                      fg.precision,
                      h.precision,
                      fg.recall,
                      h.recall,
                      fg.runtime,
                      h.runtime));
          }
        }
    }
    List<Pair> pairs = readPairs(output.resolve("paired_seed_results.csv"));
    writeSummaries(output, pairs);
    writeStatisticalTests(output, pairs);
    writeModelComparison(output, pairs);
    writeCandidateRankingMetrics(data, output, models);
    writeReport(output, config, rows, pairs);
  }

  private static void writeSummaries(Path output, List<Pair> pairs) throws IOException {
    try (BufferedWriter w = Files.newBufferedWriter(output.resolve("dataset_summary.csv"))) {
      w.write(
          "dataset,model,n,FG_mean_F1,hybrid_mean_F1,mean_paired_F1_difference,FG_precision_mean,hybrid_precision_mean,FG_recall_mean,hybrid_recall_mean,FG_runtime_mean,hybrid_runtime_mean\n");
      for (String d : new TreeSet<>(pairs.stream().map(p -> p.dataset).toList()))
        for (String m : List.of("Logistic", "MLP", "RBM"))
          writeSummary(
              w,
              d,
              m,
              pairs.stream().filter(p -> p.dataset.equals(d) && p.model.equals(m)).toList());
    }
    try (BufferedWriter w = Files.newBufferedWriter(output.resolve("overall_summary.csv"))) {
      w.write(
          "model,n,FG_mean_F1,hybrid_mean_F1,mean_paired_F1_difference,FG_precision_mean,hybrid_precision_mean,FG_recall_mean,hybrid_recall_mean,FG_runtime_mean,hybrid_runtime_mean\n");
      for (String m : List.of("Logistic", "MLP", "RBM"))
        writeSummary(w, "ALL", m, pairs.stream().filter(p -> p.model.equals(m)).toList());
    }
  }

  private static void writeSummary(BufferedWriter w, String d, String m, List<Pair> ps)
      throws IOException {
    double[] fg = ps.stream().mapToDouble(p -> p.fgF1).toArray(),
        hy = ps.stream().mapToDouble(p -> p.hyF1).toArray();
    double difference = mean(ps.stream().mapToDouble(p -> p.diff).toArray());
    double fgPrecision = mean(ps.stream().mapToDouble(p -> p.fgPrecision).toArray());
    double hyPrecision = mean(ps.stream().mapToDouble(p -> p.hyPrecision).toArray());
    double fgRecall = mean(ps.stream().mapToDouble(p -> p.fgRecall).toArray());
    double hyRecall = mean(ps.stream().mapToDouble(p -> p.hyRecall).toArray());
    double fgRuntime = mean(ps.stream().mapToDouble(p -> p.fgRuntime).toArray());
    double hyRuntime = mean(ps.stream().mapToDouble(p -> p.hyRuntime).toArray());
    if (d.equals("ALL")) {
      w.write(
          String.format(
              Locale.ROOT,
              "%s,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.3f,%.3f%n",
              m,
              ps.size(),
              mean(fg),
              mean(hy),
              difference,
              fgPrecision,
              hyPrecision,
              fgRecall,
              hyRecall,
              fgRuntime,
              hyRuntime));
    } else {
      w.write(
          String.format(
              Locale.ROOT,
              "%s,%s,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.3f,%.3f%n",
              d,
              m,
              ps.size(),
              mean(fg),
              mean(hy),
              difference,
              fgPrecision,
              hyPrecision,
              fgRecall,
              hyRecall,
              fgRuntime,
              hyRuntime));
    }
  }

  private static void writeStatisticalTests(Path output, List<Pair> pairs) throws IOException {
    List<Test> tests = new ArrayList<>();
    for (String scope : new ArrayList<>(List.of("ALL", "LFR6", "LFR7", "LFR8", "LFR9")))
      for (String model : List.of("Logistic", "MLP", "RBM")) {
        List<Pair> ps =
            pairs.stream()
                .filter(
                    p -> p.model.equals(model) && (scope.equals("ALL") || p.dataset.equals(scope)))
                .toList();
        double[] d = ps.stream().mapToDouble(p -> p.diff).toArray();
        Stats stats = Stats.of(d, RANDOM_SEED + model.hashCode() + scope.hashCode());
        tests.add(new Test(scope, model, stats));
      }
    tests.sort(Comparator.comparingDouble(t -> t.stats.pValue));
    for (int i = 0; i < tests.size(); i++)
      tests.get(i).holm = Math.min(1, tests.get(i).stats.pValue * (tests.size() - i));
    try (BufferedWriter w = Files.newBufferedWriter(output.resolve("statistical_tests.csv"))) {
      w.write(
          "scope,model,n,mean_difference,sd_difference,median_difference,ci95_low,ci95_high,improved,unchanged,regressed,permutation_p,holm_p,cohen_dz,effect_direction\n");
      for (Test t : tests) w.write(t.csv());
    }
  }

  private static void writeModelComparison(Path output, List<Pair> pairs) throws IOException {
    try (BufferedWriter w = Files.newBufferedWriter(output.resolve("model_comparison.csv"))) {
      w.write(
          "model,n,mean_F1_difference,sd_F1_difference,mean_hybrid_F1,mean_runtime_ms,overrides\n");
      List<Row> allRows = readRows(output.resolve("per_seed_results.csv"));
      for (String model : List.of("Logistic", "MLP", "RBM")) {
        List<Pair> ps = pairs.stream().filter(p -> p.model.equals(model)).toList();
        long overrides =
            allRows.stream()
                .filter(r -> r.algorithm.equals(model) && r.success && r.override)
                .count();
        w.write(
            String.format(
                Locale.ROOT,
                "%s,%d,%.10f,%.10f,%.10f,%.3f,%d%n",
                model,
                ps.size(),
                mean(ps.stream().mapToDouble(p -> p.diff).toArray()),
                sd(ps.stream().mapToDouble(p -> p.diff).toArray()),
                mean(ps.stream().mapToDouble(p -> p.hyF1).toArray()),
                mean(ps.stream().mapToDouble(p -> p.hyRuntime).toArray()),
                overrides));
      }
    }
  }

  private static void writeCandidateRankingMetrics(
      Path data, Path output, Map<String, CandidateScorer> models) throws IOException {
    try (BufferedWriter w =
        Files.newBufferedWriter(output.resolve("candidate_ranking_metrics.csv"))) {
      w.write("dataset,model,n,accuracy,positive_rate\n");
      for (int i = 5; i < 9; i++) {
        LfrSpec spec = LfrSpec.paperTable()[i];
        Graph graph = DatasetReader.edgeList(data.resolve(spec.name() + ".edgelist"), false);
        List<Set<Integer>> truth =
            DatasetReader.communities(data.resolve(spec.name() + ".truth"), false);
        List<LogisticCandidateScorer.TrainingExample> examples =
            CandidateTrainingData.collect(graph, truth, GAMMA);
        for (Map.Entry<String, CandidateScorer> model : models.entrySet()) {
          int correct = 0, positive = 0;
          for (var e : examples) {
            int actual = e.label();
            int predicted = model.getValue().probability(e.features()) >= .5 ? 1 : 0;
            if (actual == predicted) correct++;
            positive += actual;
          }
          w.write(
              String.format(
                  Locale.ROOT,
                  "%s,%s,%d,%.10f,%.10f%n",
                  spec.name(),
                  model.getKey(),
                  examples.size(),
                  examples.isEmpty() ? 0 : correct / (double) examples.size(),
                  examples.isEmpty() ? 0 : positive / (double) examples.size()));
        }
      }
    }
  }

  private static void writeReport(Path output, Config config, List<Row> rows, List<Pair> pairs)
      throws IOException {
    try (BufferedWriter w = Files.newBufferedWriter(output.resolve("experiment_report.md"))) {
      w.write("# Rigorous hybrid FG evaluation\n\n");
      w.write(
          "## Research question and protocol\n\n"
              + "The untouched FG detector is compared with safe-gated logistic regression, MLP,"
              + " and a genuine Bernoulli-Bernoulli RBM. Training uses LFR1-LFR2 for validation"
              + " tuning, validation uses the first five deterministic eligible seeds of LFR3-LFR5,"
              + " and final models are retrained on LFR1-LFR5"
              + " before one-time evaluation on every eligible seed in LFR6-LFR9. Test outcomes do"
              + " not select settings.\n\n");
      w.write("## Selected configuration\n\n```text\n" + config + "\n```\n\n");
      w.write(
          "## Dataset and seed accounting\n\n"
              + "The eligible-seed rule is every node present in the truth file and representable"
              + " in the graph. Failed evaluations remain in `per_seed_results.csv`; only"
              + " successful matching FG/model rows enter paired analysis.\n\n");
      w.write(
          "## Models\n\n"
              + "- FG: original implementation.\n"
              + "- Logistic: standardized logistic regression.\n"
              + "- MLP: one-hidden-layer neural network.\n"
              + "- RBM: Bernoulli-Bernoulli RBM with feature bits plus a label visible unit,"
              + " trained with CD-1.\n\n");
      w.write(
          "## Statistical interpretation\n\n"
              + "See `statistical_tests.csv` for paired permutation p-values, Holm-adjusted values,"
              + " bootstrap confidence intervals, and Cohen's dz. These are paired seed analyses;"
              + " seeds within a generated graph may be correlated, so graph-level dependence"
              + " remains a limitation. No positive claim is made without a confidence interval and"
              + " consistency supporting it.\n\n");
      w.write(
          "## Artifacts\n\n"
              + "All required CSV files, manifest, audit log, model metadata, and PNG figures are"
              + " generated from saved evaluation rows.\n");
    }
  }

  private static void writeSelectedConfig(Path output, Config c) throws IOException {
    Files.writeString(
        output.resolve("selected_config.json"),
        String.format(
            Locale.ROOT,
            "{\n"
                + "  \"membership_weight\": %.4f,\n"
                + "  \"external_penalty\": %.4f,\n"
                + "  \"minimum_membership\": %.4f,\n"
                + "  \"modularity_slack\": %.4f,\n"
                + "  \"selection_metric\": \"mean_validation_f1\",\n"
                + "  \"validation_f1\": %.10f,\n"
                + "  \"tie_breaking\": \"higher F1, then lower weight, lower slack, lower"
                + " threshold\"\n"
                + "}\n",
            c.weight,
            c.penalty,
            c.threshold,
            c.slack,
            c.validationF1));
  }

  private static void writeTrainingMetadata(
      Path output,
      List<?> train,
      List<?> validation,
      List<?> finalExamples,
      Map<String, CandidateScorer> models)
      throws IOException {
    try (BufferedWriter w = Files.newBufferedWriter(output.resolve("training_metadata.txt"))) {
      w.write(
          "training_split=LFR1,LFR2\n"
              + "validation_split=LFR3,LFR4,LFR5\n"
              + "final_training_split=LFR1,LFR2,LFR3,LFR4,LFR5\n");
      w.write(
          "train_examples="
              + train.size()
              + "\nvalidation_examples="
              + validation.size()
              + "\nfinal_examples="
              + finalExamples.size()
              + "\n");
      w.write(
          "feature_names="
              + String.join(",", CandidateFeatures.NAMES)
              + "\n"
              + "label=at least half of folded candidate nodes in seed truth community\n"
              + "random_seed="
              + RANDOM_SEED
              + "\n");
      for (Map.Entry<String, CandidateScorer> e : models.entrySet())
        w.write(e.getKey() + "=" + e.getValue().modelType() + "\n");
    }
  }

  private static void writeManifest(
      Path output, Path data, String status, Config selected, Set<String> models)
      throws IOException {
    String selectedText =
        selected == null
            ? "null"
            : String.format(
                Locale.ROOT,
                "{\"membership_weight\":%.4f,\"modularity_slack\":%.4f,\"minimum_membership\":%.4f}",
                selected.weight,
                selected.slack,
                selected.threshold);
    Files.writeString(
        output.resolve("experiment_manifest.json"),
        String.format(
            Locale.ROOT,
            "{\n"
                + "  \"status\": \"%s\",\n"
                + "  \"data_directory\": \"%s\",\n"
                + "  \"training\": [\"LFR1\",\"LFR2\"],\n"
                + "  \"validation\": [\"LFR3\",\"LFR4\",\"LFR5\"],\n"
                + "  \"held_out_test\": [\"LFR6\",\"LFR7\",\"LFR8\",\"LFR9\"],\n"
                + "  \"eligible_seed_rule\": \"all node identifiers present in truth and graph\",\n"
                + "  \"gamma\": %.4f,\n"
                + "  \"h\": %.4f,\n"
                + "  \"random_seed\": %d,\n"
                + "  \"selected_config\": %s,\n"
                + "  \"models\": \"%s\",\n"
                + "  \"updated_at\": \"%s\"\n"
                + "}\n",
            status,
            data.toAbsolutePath().toString().replace("\\", "\\\\"),
            GAMMA,
            H,
            RANDOM_SEED,
            selectedText,
            models == null ? "pending" : String.join(",", models),
            Instant.now()));
  }

  private static List<Row> readRows(Path file) throws IOException {
    List<Row> result = new ArrayList<>();
    try (BufferedReader r = Files.newBufferedReader(file)) {
      String line;
      while ((line = r.readLine()) != null) {
        String[] p = line.split(",", -1);
        if (p.length >= 15 && !p[0].equals("dataset")) result.add(Row.parse(p));
      }
    }
    return result;
  }

  private static List<Pair> readPairs(Path file) throws IOException {
    List<Pair> result = new ArrayList<>();
    try (BufferedReader r = Files.newBufferedReader(file)) {
      String line;
      while ((line = r.readLine()) != null) {
        String[] p = line.split(",", -1);
        if (p.length >= 13 && !p[0].equals("dataset")) result.add(Pair.parse(p));
      }
    }
    return result;
  }

  private static Row find(List<Row> rows, String dataset, int seed, String model) {
    return rows.stream()
        .filter(
            r ->
                r.dataset.equals(dataset)
                    && r.seed == seed
                    && r.algorithm.equals(model)
                    && r.success)
        .findFirst()
        .orElse(null);
  }

  private static double elapsed(long start) {
    return (System.nanoTime() - start) / 1_000_000.0;
  }

  private static double mean(double[] a) {
    return a.length == 0 ? 0 : Arrays.stream(a).average().orElse(0);
  }

  private static double sd(double[] a) {
    if (a.length < 2) return 0;
    double m = mean(a);
    return Math.sqrt(Arrays.stream(a).map(x -> (x - m) * (x - m)).sum() / (a.length - 1));
  }

  private record Config(
      double weight, double penalty, double threshold, double slack, double validationF1) {
    boolean betterThan(Config o) {
      return validationF1 > o.validationF1 + 1e-12
          || (Math.abs(validationF1 - o.validationF1) < 1e-12
              && (weight < o.weight
                  || (weight == o.weight
                      && (slack < o.slack || (slack == o.slack && threshold < o.threshold)))));
    }

    @Override
    public String toString() {
      return "membership_weight="
          + weight
          + "\nexternal_penalty="
          + penalty
          + "\nminimum_membership="
          + threshold
          + "\nmodularity_slack="
          + slack
          + "\nvalidation_f1="
          + validationF1;
    }
  }

  private record Result(
      String dataset,
      int seed,
      String algorithm,
      String modelType,
      Metrics.Score score,
      double gain,
      String candidate,
      boolean override,
      String reason,
      double runtime,
      List<String> auditLines,
      boolean success,
      String failure) {
    static Result success(
        String d,
        int s,
        String a,
        String mt,
        Metrics.Score sc,
        double g,
        String c,
        boolean o,
        String r,
        double rt) {
      return success(d, s, a, mt, sc, g, c, o, r, rt, List.of());
    }

    static Result success(
        String d,
        int s,
        String a,
        String mt,
        Metrics.Score sc,
        double g,
        String c,
        boolean o,
        String r,
        double rt,
        List<String> audit) {
      return new Result(d, s, a, mt, sc, g, c, o, r, rt, audit, true, "");
    }

    static Result failure(String d, int s, String a, String mt, Exception e) {
      return new Result(
          d,
          s,
          a,
          mt,
          new Metrics.Score(0, 0, 0),
          0,
          "",
          false,
          "failure",
          0,
          List.of(),
          false,
          sanitize(e));
    }

    String csv() {
      return String.format(
          Locale.ROOT,
          "%s,%s,%d,%s,%s,%.10f,%.10f,%.10f,%.10f,%s,%s,%s,%.3f,%s,%s%n",
          dataset,
          dataset,
          seed,
          algorithm,
          modelType,
          score.f1(),
          score.precision(),
          score.recall(),
          gain,
          candidate,
          override,
          reason,
          runtime,
          success ? "SUCCESS" : "FAILED",
          failure);
    }

    private static String sanitize(Exception e) {
      return e.getClass().getSimpleName()
          + ":"
          + String.valueOf(e.getMessage()).replace(',', ';').replace('\n', ' ');
    }
  }

  private record Row(
      String dataset,
      int seed,
      String algorithm,
      double f1,
      double precision,
      double recall,
      double runtime,
      boolean success,
      boolean override) {
    static Row parse(String[] p) {
      return new Row(
          p[0],
          Integer.parseInt(p[2]),
          p[3],
          Double.parseDouble(p[5]),
          Double.parseDouble(p[6]),
          Double.parseDouble(p[7]),
          Double.parseDouble(p[12]),
          p[13].equals("SUCCESS"),
          Boolean.parseBoolean(p[10]));
    }
  }

  private record Pair(
      String dataset,
      int seed,
      String model,
      double fgF1,
      double hyF1,
      double diff,
      double fgPrecision,
      double hyPrecision,
      double fgRecall,
      double hyRecall,
      double fgRuntime,
      double hyRuntime) {
    static Pair parse(String[] p) {
      return new Pair(
          p[0],
          Integer.parseInt(p[2]),
          p[3],
          Double.parseDouble(p[4]),
          Double.parseDouble(p[5]),
          Double.parseDouble(p[6]),
          Double.parseDouble(p[7]),
          Double.parseDouble(p[8]),
          Double.parseDouble(p[9]),
          Double.parseDouble(p[10]),
          Double.parseDouble(p[11]),
          Double.parseDouble(p[12]));
    }
  }

  private static final class Test {
    private final String scope;
    private final String model;
    private final Stats stats;
    private double holm;

    private Test(String scope, String model, Stats stats) {
      this.scope = scope;
      this.model = model;
      this.stats = stats;
    }

    private String csv() {
      return String.format(
          Locale.ROOT,
          "%s,%s,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%d,%d,%d,%.10f,%.10f,%.10f,%s%n",
          scope,
          model,
          stats.n,
          stats.mean,
          stats.sd,
          stats.median,
          stats.low,
          stats.high,
          stats.improved,
          stats.unchanged,
          stats.regressed,
          stats.pValue,
          holm,
          stats.effect,
          stats.mean > 0 ? "positive" : stats.mean < 0 ? "negative" : "zero");
    }
  }

  private record Stats(
      int n,
      double mean,
      double sd,
      double median,
      double low,
      double high,
      int improved,
      int unchanged,
      int regressed,
      double pValue,
      double effect) {
    static Stats of(double[] d, long seed) {
      if (d.length == 0) return new Stats(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0);
      double m = RigorousEvaluation.mean(d), s = RigorousEvaluation.sd(d);
      double[] sorted = d.clone();
      Arrays.sort(sorted);
      int up = 0, same = 0;
      for (double x : d) {
        if (x > 1e-12) up++;
        else if (x < -1e-12) {
        } else same++;
      }
      int down = d.length - up - same;
      double[] boot = new double[10000];
      Random r = new Random(seed);
      for (int b = 0; b < boot.length; b++) {
        double sum = 0;
        for (int i = 0; i < d.length; i++) sum += d[r.nextInt(d.length)];
        boot[b] = sum / d.length;
      }
      Arrays.sort(boot);
      int extreme = 0;
      for (int b = 0; b < 10000; b++) if (Math.abs(boot[b]) >= Math.abs(m)) extreme++;
      int permutationExtreme = 0;
      Random signs = new Random(seed ^ 0x5DEECE66DL);
      for (int b = 0; b < 10000; b++) {
        double signedMean = 0;
        for (double value : d) signedMean += signs.nextBoolean() ? value : -value;
        signedMean /= d.length;
        if (Math.abs(signedMean) >= Math.abs(m)) permutationExtreme++;
      }
      double p = (permutationExtreme + 1.0) / 10001.0;
      return new Stats(
          d.length,
          m,
          s,
          sorted[d.length / 2],
          boot[250],
          boot[9750],
          up,
          same,
          down,
          Math.min(1, 2 * p),
          s == 0 ? 0 : m / s);
    }
  }
}
