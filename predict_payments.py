"""Score payments with a saved model; member labels are not required."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings
import joblib
import sklearn
from xgboost import XGBClassifier
from fraud_features import FeatureConfig, build_dataset, load_events
from train_xgboost import prediction_table, threshold_value


class FraudScorer:
    """Reusable scorer. Pass cleaned events including history and the new initiation."""
    def __init__(self, model_dir):
        folder = Path(model_dir)
        self.metadata = json.loads((folder / 'metadata.json').read_text())
        if self.metadata['versions']['scikit-learn'] != sklearn.__version__:
            warnings.warn('scikit-learn version differs from training. Use the recorded version for reproducibility.')
        self.config = FeatureConfig(**self.metadata['feature_config'])
        self.threshold = json.loads((folder / 'threshold.json').read_text())['fraud_threshold']
        self.columns = json.loads((folder / 'feature_columns.json').read_text())['raw']
        # Load only trusted artifacts: joblib uses Python pickle.
        self.preprocessor = joblib.load(folder / 'preprocessor.joblib')
        self.model = XGBClassifier()
        self.model.load_model(folder / 'xgboost_model.json')

    def score(self, events, event_id=None, threshold=None):
        threshold = self.threshold if threshold is None else threshold_value(threshold)
        metadata, features = build_dataset(events, config=self.config, event_id=event_id)
        if set(features.columns) != set(self.columns):
            raise ValueError('Feature schema differs from training. Use the matching fraud_features.py.')
        matrix = self.preprocessor.transform(features[self.columns])
        return prediction_table(metadata, self.model.predict_proba(matrix), threshold)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events', required=True, help='Event CSV containing payment initiations and prior history')
    p.add_argument('--model-dir', default='artifacts')
    p.add_argument('--output', default='predictions.csv')
    p.add_argument('--event-id', help='Score only this outgoing initiation event, using its prior history')
    p.add_argument('--threshold', type=threshold_value, help='Override saved review cutoff without retraining')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError(f'{output} exists. Choose a new --output path.')
    result = FraudScorer(args.model_dir).score(load_events(args.events), args.event_id, args.threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f'Scored {len(result)} payments; flagged {int(result.flag_for_review.sum())}. Saved {output.resolve()}')


if __name__ == '__main__':
    main()
