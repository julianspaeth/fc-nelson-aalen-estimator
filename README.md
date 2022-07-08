# Nelson-Aalen FeatureCloud App

## Description

This app calculates the hazard rate for time-to-event data using the Nelson-Aalen estimator.

## Input

- `input`: containing the local training data (columns: features; rows: samples)

## Output

- `cum_hazard_function`: CSV file containing the survival function data
- `hazard_plot`: PNG image showing the survival plot (Kaplan-Meier plot)
- `logrank_test`: CSV file containing the pairwise logrank-test results

## Workflows

This app is not compatible with other FeatureCloud apps.

## Config

Use the config file to customize your training. Just upload it together with your training data as `config.yml`

```yml
fc_nelson_aalen:
  files: # file names
    input: lymphoma1.csv # name of the input CSV/TSV/sas7bdat file
    output: # name of the output files
      cum_hazard_function: "cum_hazard_function" # name of the CSV file containing the survival function data
      hazard_plot: "hazard_plot" # name of the PNG image showing the survival plot (Kaplan-Meier plot)
      logrank_test: logrank_test # If a category column is given: name of CSV file containing the pairwise logrank-test results

  # parameters
  parameters:
    duration_col: "Time" # name of the column containing the time values
    event_col: "Censor" # name of the column containing the event values (1=event occurred, 0=censored)
    category_col: "Stage_group" # name of the column containing the categories that shall be analysed separately (e.g. treatment A vs. treatment B)
    differential_privacy: "none" # amount of differential privacy added to the computation (none, low, middle or high)
    multipletesting_method: "bonferroni" # Method used for testing and adjustment of pvalues in the pairwise logrank test
      #bonferroni : one-step correction
      #sidak : one-step correction
      #holm-sidak : step down method using Sidak adjustments
      #holm : step-down method using Bonferroni adjustments
      #simes-hochberg : step-up method (independent)
      #hommel : closed method based on Simes tests (non-negative)
      #fdr_bh : Benjamini/Hochberg (non-negative)
      #fdr_by : Benjamini/Yekutieli (negative)
      #fdr_tsbh : two stage fdr correction (non-negative)
      #fdr_tsbky : two stage fdr correction (non-negative)
```

## Privacy

- Event times and counts are exchanged
- Differential privacy can be applied to make the resulting hazard rates private.
