"""Train/evaluate the payment classifier. Run with --help for options."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost
from xgboost import XGBClassifier
from fraud_features import (CATEGORICAL, CLASS_NAMES, FeatureConfig, build_dataset,
                            load_events, load_members)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def probability(value):
    value = float(value)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError('Must be strictly between 0 and 1.')
    return value


def threshold_value(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError('Must be between 0 and 1.')
    return value


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be a positive integer.')
    return value


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Must be a finite positive number.')
    return value


def nonnegative_float(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('Must be a finite nonnegative number.')
    return value


def sampling_fraction(value):
    value = float(value)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError('Must be greater than 0 and at most 1.')
    return value


def split_members(meta, members, test_size, validation_size, seed):
    eligible = members[members.synthetic_member_id.isin(meta.synthetic_member_id)].copy()
    try:
        trainval, test = train_test_split(eligible, test_size=test_size,
                                         stratify=eligible.member_target, random_state=seed)
        train, valid = train_test_split(trainval, test_size=validation_size,
                                       stratify=trainval.member_target, random_state=seed)
    except ValueError as exc:
        raise ValueError('Cannot make stratified member splits. Use more members or larger split fractions. ' + str(exc)) from exc
    groups = {'train': train, 'validation': valid, 'test': test}
    sets = [set(g.synthetic_member_id) for g in groups.values()]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3))
    indices = {name: meta.index[meta.synthetic_member_id.isin(g.synthetic_member_id)].to_numpy()
               for name, g in groups.items()}
    for name, idx in indices.items():
        if set(meta.loc[idx, 'target']) != {0, 1, 2}:
            raise ValueError(f'{name} payments do not contain all three classes; change split fractions or seed.')
    return groups, indices


def select_threshold(y, fraud_probability, method='f1', minimum_precision=0.7):
    """Search validation only. Tie-break F1 by proximity to 0.5, then higher cutoff."""
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = fraud_probability >= threshold
        rows.append({'threshold': float(threshold), 'precision': precision_score(y, pred, zero_division=0),
                     'recall': recall_score(y, pred, zero_division=0),
                     'f1': f1_score(y, pred, zero_division=0), 'flagged_count': int(pred.sum())})
    table = pd.DataFrame(rows)
    if method == 'precision':
        eligible = table[(table.precision >= minimum_precision) & (table.flagged_count > 0)]
        if eligible.empty:
            raise ValueError('No validation threshold meets --minimum-precision. Lower it or use --threshold-method f1.')
        chosen = eligible.sort_values('threshold').iloc[0]
    else:
        candidates = table[np.isclose(table.f1, table.f1.max())].copy()
        candidates['distance'] = (candidates.threshold - 0.5).abs()
        chosen = candidates.sort_values(['distance', 'threshold'], ascending=[True, False]).iloc[0]
    return float(chosen.threshold), table


def evaluate(meta, proba, threshold):
    y = meta.target.to_numpy()
    pred = proba.argmax(axis=1)
    binary = y > 0
    fraud = proba[:, 1] + proba[:, 2]
    flag = fraud >= threshold
    look = meta.is_legitimate_lookalike.to_numpy() & (y == 0)
    first = (meta[meta.target > 0].sort_values(['synthetic_member_id', 'event_sequence'])
             .groupby('synthetic_member_id').head(1).index)
    first_positions = meta.index.get_indexer(first)
    tn, fp, fn, tp = confusion_matrix(binary, flag, labels=[False, True]).ravel()
    return {
        'payment_count': len(y), 'member_count': int(meta.synthetic_member_id.nunique()),
        'class_counts': {CLASS_NAMES[i]: int((y == i).sum()) for i in range(3)},
        'multiclass_report': classification_report(y, pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                                  output_dict=True, zero_division=0),
        'confusion_matrix_true_rows_predicted_columns': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        'macro_ovr_roc_auc': float(roc_auc_score(y, proba, multi_class='ovr', average='macro')),
        'multiclass_log_loss': float(log_loss(y, proba, labels=[0, 1, 2])),
        'binary_fraud': {'threshold': threshold, 'precision': float(precision_score(binary, flag, zero_division=0)),
                         'recall': float(recall_score(binary, flag, zero_division=0)),
                         'f1': float(f1_score(binary, flag, zero_division=0)),
                         'roc_auc': float(roc_auc_score(binary, fraud)),
                         'average_precision': float(average_precision_score(binary, fraud)),
                         'false_positive_rate': float(fp / (fp + tn)) if fp + tn else None,
                         'true_positive': int(tp), 'false_positive': int(fp),
                         'true_negative': int(tn), 'false_negative': int(fn)},
        'legitimate_lookalikes': {'payment_count': int(look.sum()), 'incorrectly_flagged': int(flag[look].sum()),
                                 'false_positive_rate': float(flag[look].mean()) if look.any() else None},
        'first_scam_payment': {'member_count': len(first_positions),
                               'detected': int(flag[first_positions].sum()),
                               'recall': float(flag[first_positions].mean()) if len(first_positions) else None},
    }


def prediction_table(meta, proba, threshold):
    result = meta.copy()
    for i, label in enumerate(['legitimate_probability', 'authorized_scam_probability', 'account_takeover_probability']):
        result[label] = proba[:, i]
    result['fraud_probability'] = proba[:, 1] + proba[:, 2]
    result['predicted_class_id'] = proba.argmax(axis=1)
    result['predicted_class'] = [CLASS_NAMES[i] for i in proba.argmax(axis=1)]
    result['fraud_threshold'] = threshold
    result['flag_for_review'] = result.fraud_probability >= threshold
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--events', required=True, help='Event CSV path')
    p.add_argument('--members', required=True, help='Member labels CSV path')
    p.add_argument('--output-dir', default='artifacts', help='New or empty directory')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--test-size', type=probability, default=0.2)
    p.add_argument('--validation-size', type=probability, default=0.2, help='Fraction of remaining non-test members')
    p.add_argument('--n-estimators', type=positive_int, default=1000)
    p.add_argument('--early-stopping-rounds', type=positive_int, default=50)
    p.add_argument('--max-depth', type=positive_int, default=3)
    p.add_argument('--learning-rate', type=positive_float, default=0.05)
    p.add_argument('--min-child-weight', type=nonnegative_float, default=3)
    p.add_argument('--subsample', type=sampling_fraction, default=0.8)
    p.add_argument('--colsample-bytree', type=sampling_fraction, default=0.8)
    p.add_argument('--reg-lambda', type=nonnegative_float, default=1.0)
    p.add_argument('--reg-alpha', type=nonnegative_float, default=0.0)
    p.add_argument('--gamma', type=nonnegative_float, default=0.0)
    p.add_argument('--n-jobs', type=positive_int, default=2)
    p.add_argument('--class-weight', choices=['balanced', 'none'], default='balanced')
    p.add_argument('--short-hours', type=positive_float, default=24)
    p.add_argument('--long-days', type=positive_float, default=7)
    p.add_argument('--no-interventions', action='store_true', help='Exclude intervention/trusted-contact signals for an ablation')
    p.add_argument('--threshold-method', choices=['f1', 'precision'], default='f1')
    p.add_argument('--minimum-precision', type=threshold_value, default=0.7)
    p.add_argument('--threshold', type=threshold_value, help='Fixed cutoff instead of validation tuning')
    p.add_argument('--refit', action='store_true', help='Refit on train+validation at best round; cutoff may transfer imperfectly')
    p.add_argument('--export-features', action='store_true', help='Save raw feature table with metadata for debugging')
    return p


def main():
    args = parser().parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty. Choose a new --output-dir to preserve prior results.')
    config = FeatureConfig(args.short_hours, args.long_days, not args.no_interventions)
    events, members = load_events(args.events), load_members(args.members)
    print(f'Loaded {len(events)} events and {len(members)} members. Building point-in-time features...', flush=True)
    meta, features = build_dataset(events, members, config)
    groups, idx = split_members(meta, members, args.test_size, args.validation_size, args.seed)
    numeric = [c for c in features if c not in CATEGORICAL]
    encoder = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ('numeric', 'passthrough', numeric),
    ], verbose_feature_names_out=False)
    matrices = {'train': encoder.fit_transform(features.loc[idx['train']])}
    for name in ['validation', 'test']:
        matrices[name] = encoder.transform(features.loc[idx[name]])
    for name, indices in idx.items():
        print(name, 'members=', len(groups[name]), 'payments=', len(indices),
              'classes=', meta.loc[indices, 'target'].value_counts().sort_index().to_dict(), flush=True)
    y = {name: meta.loc[indices, 'target'].to_numpy() for name, indices in idx.items()}
    params = dict(objective='multi:softprob', num_class=3, eval_metric='mlogloss', tree_method='hist',
                  random_state=args.seed, n_jobs=args.n_jobs, n_estimators=args.n_estimators,
                  learning_rate=args.learning_rate, max_depth=args.max_depth, min_child_weight=args.min_child_weight,
                  subsample=args.subsample, colsample_bytree=args.colsample_bytree, reg_lambda=args.reg_lambda,
                  reg_alpha=args.reg_alpha, gamma=args.gamma)
    def weights(labels):
        return compute_sample_weight('balanced', labels) if args.class_weight == 'balanced' else np.ones(len(labels))
    model = XGBClassifier(**params, early_stopping_rounds=args.early_stopping_rounds)
    model.fit(matrices['train'], y['train'], sample_weight=weights(y['train']),
              eval_set=[(matrices['validation'], y['validation'])], verbose=False)
    best_rounds = model.best_iteration + 1
    val_proba = model.predict_proba(matrices['validation'])
    tuned_threshold, curve = select_threshold(y['validation'] > 0, val_proba[:, 1:].sum(axis=1),
                                              args.threshold_method if args.threshold is None else 'f1', args.minimum_precision)
    threshold = tuned_threshold if args.threshold is None else args.threshold
    validation_metrics = evaluate(meta.loc[idx['validation']], val_proba, threshold)
    learning_curve = model.evals_result()
    if args.refit:
        full = np.concatenate([matrices['train'], matrices['validation']])
        labels = np.concatenate([y['train'], y['validation']])
        model = XGBClassifier(**{**params, 'n_estimators': best_rounds})
        model.fit(full, labels, sample_weight=weights(labels), verbose=False)
    test_proba = model.predict_proba(matrices['test'])
    metrics = evaluate(meta.loc[idx['test']], test_proba, threshold)
    majority = int(pd.Series(y['train']).mode().iloc[0])
    metrics['majority_class_baseline'] = {
        'class': CLASS_NAMES[majority], 'accuracy': float((y['test'] == majority).mean()),
        'macro_f1': float(f1_score(y['test'], np.full_like(y['test'], majority), average='macro', zero_division=0)),
    }
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / 'xgboost_model.json')
    joblib.dump(encoder, out / 'preprocessor.joblib')
    names = encoder.get_feature_names_out().tolist()
    save_json(out / 'feature_columns.json', {'raw': features.columns.tolist(), 'encoded': names})
    save_json(out / 'label_mapping.json', {name: i for i, name in enumerate(CLASS_NAMES)})
    save_json(out / 'threshold.json', {'fraud_threshold': threshold, 'method': 'fixed' if args.threshold is not None else args.threshold_method,
                                      'source': 'user' if args.threshold is not None else 'validation'})
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'scikit-learn': sklearn.__version__, 'xgboost': xgboost.__version__, 'joblib': joblib.__version__}
    metadata = {'feature_config': config.to_dict(), 'class_names': CLASS_NAMES, 'best_boosting_rounds': best_rounds,
                'refit': args.refit, 'arguments': vars(args), 'versions': versions,
                'input_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for name, path in [('events', args.events), ('members', args.members)]},
                'prediction_point': 'Outgoing Payment Initiated', 'probabilities_calibrated': False}
    save_json(out / 'metadata.json', metadata)
    save_json(out / 'training_metrics.json', {'validation_before_optional_refit': validation_metrics, 'test': metrics})
    save_json(out / 'learning_curve.json', learning_curve)
    curve.to_csv(out / 'validation_thresholds.csv', index=False)
    pd.DataFrame({'feature': names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).to_csv(out / 'feature_importance.csv', index=False)
    prediction_table(meta.loc[idx['test']], test_proba, threshold).to_csv(out / 'test_predictions.csv', index=False)
    prediction_table(meta.loc[idx['validation']], val_proba, threshold).to_csv(out / 'validation_predictions.csv', index=False)
    pd.concat([g[['synthetic_member_id', 'member_target']].assign(split=name) for name, g in groups.items()]).to_csv(out / 'member_splits.csv', index=False)
    if args.export_features:
        pd.concat([meta, features], axis=1).to_csv(out / 'payment_features.csv', index=False)
    print(f'Best rounds: {best_rounds}; fraud threshold: {threshold:.2f}')
    print(classification_report(y['test'], test_proba.argmax(axis=1), labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0))
    print(json.dumps({'binary_fraud': metrics['binary_fraud'], 'legitimate_lookalikes': metrics['legitimate_lookalikes'],
                      'first_scam_payment': metrics['first_scam_payment']}, indent=2))
    print(f'Saved model, preprocessing, metrics and predictions to {out.resolve()}')


if __name__ == '__main__':
    main()
