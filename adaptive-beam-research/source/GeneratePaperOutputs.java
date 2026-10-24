package foldedsubgraph;

import java.nio.file.*;

/** Generates all nine paper-sized benchmark files and PNG report figures. */
public final class GeneratePaperOutputs {
  public static void main(String[] a) throws Exception {
    Path root = Path.of(a.length > 0 ? a[0] : "datasets/generated-lfr");
    Path figures = Path.of(a.length > 1 ? a[1] : "output/figures");
    SyntheticLfrGenerator.generateAll(root, 20250721L);
    ReportGenerator.main(
        new String[] {figures.toString(), "datasets/karate.edgelist", "datasets/karate.truth"});
    System.out.println(
        "Generated " + LfrSpec.paperTable().length + " deterministic benchmark datasets.");
  }
}
