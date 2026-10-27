# Event-driven fraud detection

Rebuilt on October 27, 2026 from the implementation in this conversation after
the original temporary ZIP expired. The original CSVs were recovered unchanged.
The archive and extracted project folder now use `Event_driven_fraud_detection`.

This XGBoost baseline scores each outgoing `Payment Initiated` event as
0 = Legitimate, 1 = Authorized Scam, or 2 = Account Takeover. It includes training,
prediction, regression tests, CSVs, and regenerated trained-model artifacts.

## Setup and run

Use Python 3.11 or 3.12. Open a terminal in the extracted project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `python3` instead of `python3.12` if that is your Python 3.11/3.12 command.
On Windows use `py -3.12 -m venv .venv` then `.venv\Scripts\activate`.
Enter the commands below on one line in Windows terminals.

Train a new model:

```bash
python train_xgboost.py \
  --events data/events.csv \
  --members data/members.csv \
  --output-dir artifacts
```

Score with that model:

```bash
python predict_payments.py \
  --events data/events.csv \
  --model-dir artifacts \
  --output predictions.csv
```

Or skip training and use the included model:

```bash
python predict_payments.py --events data/events.csv --model-dir example_artifacts --output predictions.csv
```

Your original CSVs are bundled as `data/events.csv` and `data/members.csv`.
You may instead pass quoted paths to the original long filenames. No API key,
GPU, cloud service or external data is required after installing dependencies.
If macOS reports a missing `libomp.dylib`, install OpenMP with `brew install libomp`.
The pinned versions require Python 3.11+; Python 3.9 is unsupported.

Output directories must be new or empty, and prediction output files must not
already exist. Use different names for each experiment to preserve results.

## Files

| File | Purpose |
| --- | --- |
| fraud_features.py | Shared input validation, labels and historical features |
| train_xgboost.py | Splits, encoding, training, threshold selection, evaluation and saving |
| predict_payments.py | Load model and score payments without member labels |
| test_pipeline.py | Seven regression tests for leakage, splits, aggregates and reload |
| requirements.txt | Tested dependency versions |
| data/ | Original supplied CSVs with shorter filenames |
| example_artifacts/ | Regenerated model, metrics, predictions and feature export |
| RUN_RESULTS.md | Results of the regenerated default run |

## Tuning

Run `python train_xgboost.py --help` for all arguments.

| Parameter | Default | Effect |
| --- | --- | --- |
| --max-depth | 3 | Tree complexity; start with 2–4 |
| --learning-rate | 0.05 | Smaller updates often require more rounds |
| --n-estimators | 1000 | Maximum boosting rounds |
| --early-stopping-rounds | 50 | Patience on validation multiclass log loss |
| --min-child-weight | 3 | Larger values make splits more conservative |
| --subsample | 0.8 | Training-row sampling per round |
| --colsample-bytree | 0.8 | Feature sampling per tree |
| --reg-lambda | 1.0 | L2 regularization |
| --reg-alpha | 0.0 | L1 regularization |
| --gamma | 0.0 | Minimum split improvement |
| --class-weight | balanced | Use balanced sample weights or none |
| --threshold-method | f1 | Maximize validation fraud F1, or use precision |
| --minimum-precision | 0.7 | Validation precision target for precision method |
| --threshold | unset | Fixed fraud-review cutoff, overriding tuning |
| --short-hours | 24 | Short historical window, requires retraining |
| --long-days | 7 | Long historical window, requires retraining |
| --no-interventions | off | Exclude direct intervention/trusted-contact signals |
| --refit | off | Refit on train+validation using best number of rounds |
| --test-size | 0.2 | Fraction of eligible members reserved for testing |
| --validation-size | 0.2 | Fraction of remaining members for validation |
| --seed | 42 | Split and model random seed |
| --n-jobs | 2 | CPU threads |
| --export-features | off | Write raw features and metadata for auditing |
| --output-dir | artifacts | New or empty destination |

Example:

```bash
python train_xgboost.py \
  --events data/events.csv --members data/members.csv \
  --output-dir artifacts_conservative \
  --max-depth 2 --min-child-weight 5 \
  --threshold-method precision --minimum-precision 0.80
```

Precision mode selects the lowest qualifying threshold from the validation grid,
0.05 through 0.95 in 0.01 steps. It fails clearly if no threshold qualifies.
F1 ties favor the threshold closest to 0.5, then the higher threshold.
Do not repeatedly tune against held-out test metrics.

## Prediction and integration

Prediction outputs include three class probabilities, their argmax class,
`fraud_probability` (scam + takeover), `fraud_threshold`, and `flag_for_review`.
The binary flag uses fraud probability >= threshold. It can differ from the
multiclass decision, since the latter chooses the largest single class score.

Change the cutoff without retraining:

```bash
python predict_payments.py --events data/events.csv --model-dir artifacts \
  --threshold 0.70 --output predictions_070.csv
```

Higher cutoffs generally flag fewer payments. Select the tradeoff using validation
and review capacity. These are uncalibrated model scores, not verified real-world
fraud probabilities.

Use `--event-id` to score one outgoing initiation while retaining all prior events:

```bash
python predict_payments.py --events data/events.csv --model-dir artifacts \
  --event-id YOUR_INITIATION_EVENT_ID --output one_prediction.csv
```

Replace the placeholder with an event ID from the event CSV or test predictions.
An event ID is different from a payment ID.

Programmatic use:

```python
from fraud_features import load_events
from predict_payments import FraudScorer

scorer = FraudScorer('artifacts')
events = load_events('data/events.csv')
result = scorer.score(events, event_id='YOUR_INITIATION_EVENT_ID')
```

Load the scorer once and reuse it. `score()` expects cleaned events as returned
by `load_events`. Supply the new initiation and known prior history. No member
label file is needed. Missing history produces missing/zero historical features
and changes the input context. A full replay CSV is supported because future
rows are filtered separately for each payment. This is a small offline baseline,
not a high-volume streaming service. It only writes scores and review flags.

## Labels, features and leakage

The dataset contains 118 members, 7,471 events, and 764 outgoing initiations.
A payment is fraudulent only if listed in that member's `scam_payment_ids`.
The member's scenario determines scam versus takeover. Other payments by a
victim remain legitimate. This assumes the listed scam IDs are exhaustive.

The seed-42 split is 75/19/24 members (train/validation/test), with 451/132/181
payments. Members never cross splits. Splitting is stratified by member scenario
class. Missing classes or invalid split sizes raise errors. The one-hot encoder
is fitted on training only; unknown categories are accepted and numeric missing
values remain NaN. No scaling is needed. Weights come only from training rows.

Features include current initiation amount, recipient age/status, baseline and
velocity bands, channel, rail, time-of-day/week, and supplied current flags.
Those flags and baseline summaries must themselves have been computed causally
upstream; the CSV cannot prove that assumption. Historical features summarize
authentication, profile, recipient, money movement, contact and intervention
activity, plus time since key events and recipient-specific payment history.
All-history, short-window and long-window summaries are included.

- History requires a smaller event sequence and timestamp no later than the
  initiation. Equal timestamps require smaller sequence. Duplicate keys and
  backwards timestamps are rejected.
- Case & Recovery history is excluded. Current outcome/response/warning/case
  fields, notes, IDs and member outcome columns are not model features.
- The latest prior balance snapshot replaces the ambiguous current
  `available_balance_after`. Missing prior balance remains missing.
- Settled money uses explicit completed event types. Lifecycle events such as
  released/held/cancelled cannot count the same transfer again. Repeated completed
  payment IDs are deduplicated; standalone ledger entries retain their event IDs.
- Recipient and risk-intensity counts use initiation rows once per payment.
  Their amounts represent attempts; outgoing/incoming settled aggregates are separate.
- No narrative keyword or embedding features are used.
- Prior interventions are allowed by default. `--no-interventions` removes direct
  intervention/trusted-contact events and signals for a sensitivity check. It does
  not eliminate every possible indirect effect of earlier bank actions.
- Unmatched scam IDs, unknown labels, missing critical fields and duplicate
  payment initiations fail instead of being silently discarded.

The default retains the early-stopped model on which the validation cutoff was
selected. `--refit` follows the optional train-plus-validation refit from the guide,
but transferring the cutoff to a changed model can change its precision/recall.
The encoder remains training-only. Validation metrics in that mode describe the
pre-refit model; test metrics describe the final saved model.

## Outputs and tests

Model files: `xgboost_model.json`, `preprocessor.joblib`, `feature_columns.json`,
`label_mapping.json`, `threshold.json`, `metadata.json`. Metadata records versions,
input hashes, parameters, feature windows and best boosting round. Keep these
together. Only load trusted joblib files because they use pickle.

Evaluation: `training_metrics.json`, `test_predictions.csv`,
`validation_predictions.csv`, `validation_thresholds.csv`, `learning_curve.json`,
`member_splits.csv`, and `feature_importance.csv`. Optional `payment_features.csv`
contains metadata and labels for inspection, but these were excluded from model
inputs. Importance is global model importance, not a causal or per-case explanation.

```bash
python -m unittest -v test_pipeline.py
```

Tests cover future-event invariance, current-outcome exclusion, case-history
exclusion, lifecycle deduplication, labels/member splits, missing history and
unknown categories, and saved-model consistency. The final test uses the bundled
example artifacts and skips if they are removed.

This is a synthetic hackathon benchmark with few independent members. Member
separation does not prevent repeated scenario templates across splits. The test
set has only six takeover payments. High scores do not establish real-world
performance; later evaluation needs independent scenario families and genuinely
unseen operational data. Class weighting and synthetic prevalence also mean
probability calibration requires separate evaluation.
