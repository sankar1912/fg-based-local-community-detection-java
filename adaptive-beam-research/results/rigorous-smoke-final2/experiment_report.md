# Rigorous hybrid FG evaluation

## Research question and protocol

The untouched FG detector is compared with safe-gated logistic regression, MLP, and a genuine Bernoulli-Bernoulli RBM. Training uses LFR1-LFR2 for validation tuning, validation uses the first five deterministic eligible seeds of LFR3-LFR5, and final models are retrained on LFR1-LFR5 before one-time evaluation on every eligible seed in LFR6-LFR9. Test outcomes do not select settings.

## Selected configuration

```text
membership_weight=0.2
external_penalty=0.0
minimum_membership=0.6
modularity_slack=0.0
validation_f1=0.28034549513387697
```

## Dataset and seed accounting

The eligible-seed rule is every node present in the truth file and representable in the graph. Failed evaluations remain in `per_seed_results.csv`; only successful matching FG/model rows enter paired analysis.

## Models

- FG: original implementation.
- Logistic: standardized logistic regression.
- MLP: one-hidden-layer neural network.
- RBM: Bernoulli-Bernoulli RBM with feature bits plus a label visible unit, trained with CD-1.

## Statistical interpretation

See `statistical_tests.csv` for paired permutation p-values, Holm-adjusted values, bootstrap confidence intervals, and Cohen's dz. These are paired seed analyses; seeds within a generated graph may be correlated, so graph-level dependence remains a limitation. No positive claim is made without a confidence interval and consistency supporting it.

## Artifacts

All required CSV files, manifest, audit log, model metadata, and PNG figures are generated from saved evaluation rows.
