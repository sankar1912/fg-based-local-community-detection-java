package foldedsubgraph;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/** One-command reproducibility bundle: raw metrics, summaries, metadata, and PNGs. */
public final class RunAllResults {
  public static void main(String[] a) throws Exception {
    Path out = Path.of(a.length > 0 ? a[0] : "output/results"),
        lfr = Path.of(a.length > 1 ? a[1] : "datasets/generated-lfr");
    int maxSeeds = a.length > 2 ? Integer.parseInt(a[2]) : 0;
    Files.createDirectories(out);
    Files.createDirectories(lfr);
    SyntheticLfrGenerator.generateAll(lfr, 20250721L);
    writeMetadata(out, maxSeeds);
    runOne(
        out.resolve("karate-per-seed.csv"),
        Path.of("datasets/karate.edgelist"),
        Path.of("datasets/karate.truth"),
        true,
        0);
    for (LfrSpec s : LfrSpec.paperTable())
      runOne(
          out.resolve(s.name().toLowerCase(Locale.ROOT) + "-per-seed.csv"),
          lfr.resolve(s.name() + ".edgelist"),
          lfr.resolve(s.name() + ".truth"),
          false,
          maxSeeds);
    Path figures = out.resolve("figures");
    ReportGenerator.main(
        new String[] {figures.toString(), "datasets/karate.edgelist", "datasets/karate.truth"});
    PaperMetricImages.main(new String[] {figures.toString()});
    AblationMetricImages.main(new String[] {figures.toString()});
    System.out.println("Complete result bundle: " + out.toAbsolutePath());
  }

  private static void runOne(Path csv, Path graph, Path truth, boolean oneBased, int maxSeeds)
      throws Exception {
    Graph g = DatasetReader.edgeList(graph, oneBased);
    List<Set<Integer>> truths = DatasetReader.communities(truth, oneBased);
    int n = maxSeeds <= 0 ? g.size() : Math.min(maxSeeds, g.size());
    double sr = 0, sp = 0, sf = 0;
    try (BufferedWriter w = Files.newBufferedWriter(csv)) {
      w.write("dataset,seed,found_size,recall,precision,f1,gamma,h\n");
      for (int seed = 0; seed < n; seed++) {
        Set<Integer> found = new FGDetector(g).detect(seed, .45, 1.0);
        Metrics.Score z = Metrics.score(found, Metrics.truthFor(seed, truths));
        sr += z.recall();
        sp += z.precision();
        sf += z.f1();
        w.write(
            String.format(
                Locale.ROOT,
                "%s,%d,%d,%.12f,%.12f,%.12f,%.2f,%.1f%n",
                csv.getFileName(),
                seed,
                found.size(),
                z.recall(),
                z.precision(),
                z.f1(),
                .45,
                1.0));
      }
      w.write(
          String.format(
              Locale.ROOT,
              "average,-1,,%.12f,%.12f,%.12f,%.2f,%.1f%n",
              sr / n,
              sp / n,
              sf / n,
              .45,
              1.0));
    }
  }

  private static void writeMetadata(Path out, int maxSeeds) throws IOException {
    try (BufferedWriter w = Files.newBufferedWriter(out.resolve("run-metadata.txt"))) {
      w.write("method=FG-based\n");
      w.write("gamma=0.45\nh=1.0\nbeta=0.5\n");
      w.write("lfr_seed=20250721\n");
      w.write("lfr_seed_limit=" + maxSeeds + " (0 means all)\n");
      w.write("generated_at=" + java.time.Instant.now() + "\n");
      w.write(
          "paper_reference_tables=PNG files generated from reported values in the supplied PDF\n");
      w.write(
          "generated_lfr_note=macro-parameter-matched deterministic benchmark; not authors exact"
              + " realization\n");
    }
  }
}
