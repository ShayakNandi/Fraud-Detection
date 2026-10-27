# Regenerated run, October 27, 2026

The reconstructed implementation was trained again on the recovered original
CSVs, using seed 42 and the same dependency pins. It reproduced the previously
reported best round, threshold, and held-out classification counts.

| Item | Result |
| --- | --- |
| Members, train / validation / test | 75 / 19 / 24 |
| Payments, train / validation / test | 451 / 132 / 181 |
| Best boosting rounds | 133 |
| Validation-selected fraud threshold | 0.82 |
| Fraudulent test payments flagged | 22 / 31 |
| Legitimate test payments flagged | 2 / 150 |
| Fraud precision | 91.67% |
| Fraud recall | 70.97% |
| Fraud F1 | 0.80 |
| Lookalike test payments falsely flagged | 0 / 36 |
| First scam payments detected | 8 / 10 affected test members |

The multiclass confusion matrix (true rows, predicted columns):

| True / predicted | Legitimate | Authorized Scam | Account Takeover |
| --- | ---: | ---: | ---: |
| Legitimate | 146 | 3 | 1 |
| Authorized Scam | 4 | 21 | 0 |
| Account Takeover | 0 | 0 | 6 |

The multiclass argmax decision differs from the binary review cutoff decision.
Validation fraud recall was 100%; test recall was 70.97%, illustrating cutoff
instability in small samples. No parameters were changed to improve test results.
These are synthetic-data benchmark results, not real-world performance estimates.
See example_artifacts/training_metrics.json for the full metrics and individual
prediction CSVs for error inspection.
