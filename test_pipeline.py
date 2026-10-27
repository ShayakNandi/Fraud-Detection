"""Run: python -m unittest -v test_pipeline.py"""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from fraud_features import (FeatureConfig, build_dataset, build_feature_row,
                            completed_movements, load_events, load_members, payment_rows)
from train_xgboost import split_members
ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = load_events(ROOT / 'data/events.csv')
        cls.members = load_members(ROOT / 'data/members.csv')
        cls.history = cls.events[cls.events.synthetic_member_id.eq('M1001')].copy()
        cls.payment = payment_rows(cls.history).iloc[-1].copy()

    def test_future_events_cannot_change_features(self):
        p, h = self.payment, self.history
        original = pd.Series(build_feature_row(p, h))
        truncated = h[h.event_sequence <= p.event_sequence]
        assert_series_equal(original, pd.Series(build_feature_row(p, truncated)))
        changed = h.copy()
        future = changed.event_sequence > p.event_sequence
        changed.loc[future, 'amount'] = 1e12
        changed.loc[future, 'available_balance_after'] = 1e12
        changed.loc[future, 'member_response'] = 'Reported Not Me'
        changed.loc[future, 'event_type'] = 'Password Reset'
        assert_series_equal(original, pd.Series(build_feature_row(p, changed)))

    def test_current_outcomes_and_notes_cannot_change_features(self):
        altered = self.payment.copy()
        for c in ['payment_status', 'member_response', 'warning_displayed', 'case_open', 'event_note']:
            altered[c] = 'arbitrary outcome text'
        altered['available_balance_after'] = 1e12
        assert_series_equal(pd.Series(build_feature_row(self.payment, self.history)),
                            pd.Series(build_feature_row(altered, self.history)))

    def test_case_recovery_history_excluded(self):
        extra = self.history.iloc[-1].copy()
        extra['event_sequence'] = self.payment.event_sequence - 0.5
        extra['event_timestamp'] = self.payment.event_timestamp - pd.Timedelta(seconds=1)
        extra['event_category'] = 'Case & Recovery'
        extra['event_type'] = 'Scam Reported'
        extra['available_balance_after'] = 1e12
        extra['member_response'] = 'Reported Not Me'
        changed = pd.concat([self.history, pd.DataFrame([extra])], ignore_index=True)
        assert_series_equal(pd.Series(build_feature_row(self.payment, self.history)),
                            pd.Series(build_feature_row(self.payment, changed)))

    def test_completed_amounts_not_double_counted(self):
        completed = self.events[self.events.event_type.eq('Payment Completed')].iloc[0].copy()
        released = completed.copy()
        released['event_type'] = 'Payment Released'
        initiated = completed.copy()
        initiated['event_type'] = 'Payment Initiated'
        duplicate = completed.copy()
        duplicate['event_id'] = 'duplicate-record'
        movements = completed_movements(pd.DataFrame([initiated, released, completed, duplicate]))
        self.assertEqual(len(movements), 1)
        self.assertEqual(float(movements.amount.sum()), completed.amount)

    def test_labels_and_member_split(self):
        p = payment_rows(self.events)
        info = self.members.set_index('synthetic_member_id')
        meta = p[['synthetic_member_id']].reset_index(drop=True)
        meta['target'] = [int(info.loc[r.synthetic_member_id, 'member_target'])
                          if r.payment_id in info.loc[r.synthetic_member_id, 'scam_payment_set'] else 0
                          for r in p.itertuples()]
        groups, indices = split_members(meta, self.members, .2, .2, 42)
        assigned = [set(g.synthetic_member_id) for g in groups.values()]
        self.assertTrue(assigned[0].isdisjoint(assigned[1] | assigned[2]))
        self.assertTrue(assigned[1].isdisjoint(assigned[2]))
        self.assertEqual(sum(map(len, indices.values())), len(meta))
        mid = 'M1001'
        metadata, _ = build_dataset(self.history, self.members[self.members.synthetic_member_id.eq(mid)])
        for r in metadata.itertuples():
            self.assertEqual(r.target, 1 if r.payment_id in info.loc[mid, 'scam_payment_set'] else 0)

    def test_empty_history_and_unknown_categories(self):
        p = self.payment.copy()
        p['channel'] = 'New Channel'
        p['recipient_id'] = np.nan
        f = build_feature_row(p, self.history.iloc[:0], FeatureConfig())
        self.assertEqual(f['event_count_all'], 0)
        self.assertEqual(f['current_recipient_previous_payment_count'], 0)
        self.assertTrue(np.isnan(f['hours_since_password_reset']))
        self.assertEqual(f['channel'], 'New Channel')

    def test_saved_model_reproduces_test_prediction(self):
        folder = ROOT / 'example_artifacts'
        if not (folder / 'metadata.json').exists():
            self.skipTest('Train example_artifacts first to test serialization.')
        from predict_payments import FraudScorer
        saved = pd.read_csv(folder / 'test_predictions.csv').iloc[0]
        scorer = FraudScorer(folder)
        actual = scorer.score(self.events, event_id=saved.event_id).iloc[0]
        cols = ['legitimate_probability', 'authorized_scam_probability', 'account_takeover_probability']
        np.testing.assert_allclose(actual[cols].astype(float), saved[cols].astype(float), rtol=1e-6)
        self.assertAlmostEqual(float(actual[cols].astype(float).sum()), 1, places=5)
        self.assertEqual(actual.flag_for_review, saved.flag_for_review)
        altered = self.events.copy()
        altered.loc[altered.event_id.eq(saved.event_id), 'channel'] = 'Never Seen Channel'
        self.assertEqual(len(scorer.score(altered, event_id=saved.event_id)), 1)


if __name__ == '__main__':
    unittest.main()
