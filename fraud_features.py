"""Shared, point-in-time feature construction for training and prediction."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd

CLASS_NAMES = ['Legitimate', 'Authorized Scam', 'Account Takeover']
SCENARIO_CLASS = {
    'Normal Activity': 0, 'Family Emergency': 1, 'Romance / Investment': 1,
    'Tech Support / Government Impersonation': 1, 'Sound Impersonation + ATO': 2,
}
META = ['event_id', 'synthetic_member_id', 'payment_id', 'event_sequence', 'event_timestamp']
CATEGORICAL = ['channel', 'payment_rail', 'day_of_week', 'last_member_response']
NUMERIC_INPUT = ['amount', 'amount_to_baseline_ratio', 'recipient_age_days',
                 'days_since_last_same_rail', 'available_balance_after']
FLAGS = ['profile_change_recent', 'password_reset_recent', 'trusted_contact_enrolled',
         'employee_interaction']
EVENT_COUNTS = [
    'Login', 'Failed Login', 'One-Time Code Sent', 'Password Reset', 'Email Change',
    'Phone Change', 'Address Change', 'Profile Change Alert Sent', 'Recipient Added',
    'Warning Displayed', 'Confirmation Requested', 'Employee Outreach',
    'Employee Escalation', 'Account Restricted', 'Access Restored', 'Payment Held',
    'Payment Released', 'Payment Cancelled', 'Payment Stopped', 'Trusted Contact Offered',
    'Trusted Contact Enrolled', 'Trusted Contact Notified', 'Branch Visit', 'Contact Center Call',
]
DELTA_EVENTS = ['Failed Login', 'Password Reset', 'One-Time Code Sent', 'Email Change',
                'Phone Change', 'Address Change', 'Recipient Added', 'Branch Visit',
                'Contact Center Call', 'Warning Displayed', 'Employee Outreach']
RESPONSES = ['Confirmed Intended', 'Requested Help', 'Cancelled Payment',
             'Reported Not Me', 'Accepted Offer', 'Declined Offer']
COMPLETED_TYPES = ['Payment Completed', 'Debit Purchase', 'Bill Payment',
                   'Automatic Payment', 'ATM Withdrawal', 'Deposit']

@dataclass(frozen=True)
class FeatureConfig:
    short_hours: float = 24.0
    long_days: float = 7.0
    include_interventions: bool = True

    def __post_init__(self):
        if self.short_hours <= 0 or self.long_days <= 0:
            raise ValueError('History windows must be positive.')

    def to_dict(self):
        return asdict(self)


def slug(value):
    return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')


def require_columns(df, columns, kind):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f'{kind} is missing columns: {missing}')


def load_events(path: str | Path) -> pd.DataFrame:
    """Fail on malformed critical fields instead of silently dropping payments."""
    e = pd.read_csv(path, dtype=str)
    required = META + ['event_type', 'event_category', 'transaction_direction',
                       'recipient_id', 'recipient_status', 'amount_vs_member_baseline',
                       'payment_velocity_band', 'channel', 'payment_rail', 'member_response']
    require_columns(e, required + NUMERIC_INPUT + FLAGS, 'Events CSV')
    for c in e.columns:
        e[c] = e[c].str.strip().replace('', np.nan)
    for c in ['event_id', 'synthetic_member_id', 'event_sequence', 'event_timestamp',
              'event_type', 'event_category', 'transaction_direction']:
        if e[c].isna().any():
            raise ValueError(f'Missing required event value: {c}')
    e['event_sequence'] = pd.to_numeric(e.event_sequence, errors='raise')
    if (e.event_sequence < 0).any() or (e.event_sequence % 1 != 0).any():
        raise ValueError('event_sequence must be a nonnegative integer.')
    e['event_timestamp'] = pd.to_datetime(e.event_timestamp, format='mixed', errors='raise')
    if not pd.api.types.is_datetime64_any_dtype(e.event_timestamp):
        raise ValueError('Use a consistent timestamp timezone throughout the events CSV.')
    if e.event_id.duplicated().any() or e.duplicated(['synthetic_member_id', 'event_sequence']).any():
        raise ValueError('Duplicate event IDs or member/event_sequence keys.')
    for c in NUMERIC_INPUT:
        e[c] = pd.to_numeric(e[c], errors='raise')
        if np.isinf(e[c]).any():
            raise ValueError(f'Infinite numeric input: {c}')
    for c in FLAGS:
        bad = e[c].dropna()[~e[c].dropna().isin(['Yes', 'No', '1', '0', 'True', 'False'])]
        if len(bad):
            raise ValueError(f'Unexpected boolean values in {c}: {bad.unique()}')
        e[c] = e[c].map({'Yes': 1, 'No': 0, '1': 1, '0': 0, 'True': 1, 'False': 0})
    e = e.sort_values(['synthetic_member_id', 'event_sequence']).reset_index(drop=True)
    for mid, g in e.groupby('synthetic_member_id', sort=False):
        if not g.event_timestamp.is_monotonic_increasing:
            raise ValueError(f'Timestamps run backwards in event_sequence order for {mid}.')
    p = payment_rows(e)
    if p[['payment_id', 'amount']].isna().any().any() or (p.amount < 0).any():
        raise ValueError('Outgoing Payment Initiated rows need payment_id and nonnegative amount.')
    if p.duplicated(['synthetic_member_id', 'payment_id']).any():
        raise ValueError('Duplicate outgoing payment initiations for the same member/payment_id.')
    return e


def payment_rows(e):
    return e[e.event_type.eq('Payment Initiated') & e.transaction_direction.eq('Outgoing')]


def load_members(path):
    m = pd.read_csv(path, dtype=str).fillna('')
    require_columns(m, ['synthetic_member_id', 'scenario_type', 'scam_payment_ids',
                        'is_legitimate_lookalike'], 'Members CSV')
    for c in m.columns:
        m[c] = m[c].str.strip()
    if m.synthetic_member_id.eq('').any() or m.synthetic_member_id.duplicated().any():
        raise ValueError('Member IDs must be present and unique.')
    unknown = set(m.scenario_type) - set(SCENARIO_CLASS)
    if unknown:
        raise ValueError(f'Unknown scenarios: {sorted(unknown)}')
    if not m.is_legitimate_lookalike.isin(['Yes', 'No']).all():
        raise ValueError('is_legitimate_lookalike must be Yes or No.')
    m['member_target'] = m.scenario_type.map(SCENARIO_CLASS)
    m['scam_payment_set'] = m.scam_payment_ids.map(lambda s: set(p.strip() for p in s.split(';') if p.strip()))
    for r in m.itertuples():
        if bool(r.scam_payment_set) != bool(r.member_target):
            raise ValueError(f'Scenario/scam-payment label inconsistency for {r.synthetic_member_id}.')
        if r.is_legitimate_lookalike == 'Yes' and r.member_target != 0:
            raise ValueError('A legitimate lookalike cannot have a fraud scenario.')
    return m


def completed_movements(h):
    """Deduplicate payment IDs, keeping standalone ledger entries by event ID."""
    money = h[h.event_type.isin(COMPLETED_TYPES) & h.event_category.eq('Money Movement')
              & h.amount.notna()].copy()
    money['_movement_key'] = money.payment_id.fillna('event:' + money.event_id)
    return money.drop_duplicates('_movement_key', keep='first')


def history_summary(h, include_interventions):
    f = {'event_count': len(h)}
    for event in EVENT_COUNTS:
        if include_interventions or event not in INTERVENTION_TYPES:
            f[slug(event) + '_count'] = int(h.event_type.eq(event).sum())
    f['profile_change_count'] = int(h.event_type.isin(['Email Change', 'Phone Change', 'Address Change']).sum())
    initiated = payment_rows(h)
    for status in ['New', 'Existing']:
        payments = initiated[initiated.recipient_status.eq(status)]
        f[status.lower() + '_recipient_payment_count'] = len(payments)
        f[status.lower() + '_recipient_payment_amount_sum'] = float(payments.amount.sum())
    f['distinct_recipient_count'] = int(initiated.recipient_id.nunique())
    for band in ['Above Typical', 'Significantly Above Typical']:
        f[slug(band) + '_payment_count'] = int(initiated.amount_vs_member_baseline.eq(band).sum())
    f['high_velocity_payment_count'] = int(initiated.payment_velocity_band.eq('High').sum())
    money = completed_movements(h)
    for direction in ['Outgoing', 'Incoming']:
        amounts = money.loc[money.transaction_direction.eq(direction), 'amount']
        prefix = direction.lower()
        f[prefix + '_payment_count'] = len(amounts)
        f[prefix + '_amount_sum'] = float(amounts.sum())
        f[prefix + '_amount_max'] = float(amounts.max()) if len(amounts) else np.nan
        f[prefix + '_amount_mean'] = float(amounts.mean()) if len(amounts) else np.nan
    outgoing = money[money.transaction_direction.eq('Outgoing')]
    for rail in ['P2P', 'Wire', 'ACH', 'Cash', 'External Transfer']:
        f[slug(rail) + '_amount_sum'] = float(outgoing.loc[outgoing.payment_rail.eq(rail), 'amount'].sum())
    if include_interventions:
        for response in RESPONSES:
            f['response_' + slug(response) + '_count'] = int(h.member_response.eq(response).sum())
        trusted = h[h.event_category.eq('Trusted Contact')]
        for response in ['Accepted Offer', 'Declined Offer']:
            f['trusted_contact_' + slug(response) + '_count'] = int(trusted.member_response.eq(response).sum())
    return f


INTERVENTION_TYPES = {
    'Warning Displayed', 'Confirmation Requested', 'Employee Outreach', 'Employee Escalation',
    'Account Restricted', 'Access Restored', 'Payment Held', 'Payment Released',
    'Payment Cancelled', 'Payment Stopped', 'Trusted Contact Offered',
    'Trusted Contact Enrolled', 'Trusted Contact Notified',
}


def build_feature_row(p, member_events, config=FeatureConfig()):
    """Only smaller sequences at or before the scoring timestamp may contribute."""
    h = member_events[(member_events.event_sequence < p.event_sequence)
                      & (member_events.event_timestamp <= p.event_timestamp)
                      & ~member_events.event_category.eq('Case & Recovery')].copy()
    h = h.sort_values('event_sequence')
    if not config.include_interventions:
        h = h[~h.event_type.isin(INTERVENTION_TYPES)
              & ~h.event_category.isin(['Intervention', 'Trusted Contact'])]
    f = {c: p[c] for c in NUMERIC_INPUT if c != 'available_balance_after'}
    balances = h.available_balance_after.dropna()
    f['prior_available_balance'] = float(balances.iloc[-1]) if len(balances) else np.nan
    balance = f['prior_available_balance']
    f['amount_to_prior_balance_ratio'] = float(p.amount / (balance + 1)) if balance >= 0 else np.nan
    for c in ['channel', 'payment_rail']:
        f[c] = str(p[c]) if pd.notna(p[c]) else 'Unknown'
    f['day_of_week'] = str(p.event_timestamp.dayofweek)
    f['hour'] = p.event_timestamp.hour
    f['is_weekend'] = int(p.event_timestamp.dayofweek >= 5)
    for c, mapping in {
        'recipient_status': {'Existing': 0, 'New': 1},
        'amount_vs_member_baseline': {'Typical': 0, 'Above Typical': 1, 'Significantly Above Typical': 2},
        'payment_velocity_band': {'Low': 0, 'Medium': 1, 'High': 2},
    }.items():
        f[c] = mapping.get(p[c], np.nan)
    for c in FLAGS:
        if config.include_interventions or c not in ['employee_interaction', 'trusted_contact_enrolled']:
            f[c] = p[c]
    windows = {
        'all': h,
        'short': h[h.event_timestamp >= p.event_timestamp - pd.Timedelta(hours=config.short_hours)],
        'long': h[h.event_timestamp >= p.event_timestamp - pd.Timedelta(days=config.long_days)],
    }
    for name, window in windows.items():
        f.update({f'{k}_{name}': v for k, v in history_summary(window, config.include_interventions).items()})
    for event in DELTA_EVENTS:
        if not config.include_interventions and event in INTERVENTION_TYPES:
            continue
        last = h.loc[h.event_type.eq(event), 'event_timestamp'].max()
        f['hours_since_' + slug(event)] = (p.event_timestamp - last).total_seconds() / 3600 if pd.notna(last) else np.nan
    same = payment_rows(h)
    same = same[same.recipient_id.eq(p.recipient_id)] if pd.notna(p.recipient_id) else same.iloc[:0]
    f['current_recipient_previous_payment_count'] = len(same)
    f['current_recipient_previous_amount_sum'] = float(same.amount.sum())
    for label, stamp in [('first', same.event_timestamp.min()), ('last', same.event_timestamp.max())]:
        f[f'days_since_{label}_payment_to_current_recipient'] = (p.event_timestamp - stamp).total_seconds() / 86400 if pd.notna(stamp) else np.nan
    responses = h.member_response[h.member_response.isin(RESPONSES)]
    f['last_member_response'] = str(responses.iloc[-1]) if config.include_interventions and len(responses) else 'None'
    return f


def build_dataset(events, members=None, config=FeatureConfig(), event_id=None):
    """Return metadata and features separately, keeping IDs/labels out of X."""
    payments = payment_rows(events)
    if event_id is not None:
        payments = payments[payments.event_id.eq(event_id)]
    if payments.empty:
        raise ValueError('No matching outgoing Payment Initiated events to score.')
    member_info = members.set_index('synthetic_member_id') if members is not None else None
    if member_info is not None:
        unknown = set(events.synthetic_member_id) - set(member_info.index)
        if unknown:
            raise ValueError(f'Events have members absent from labels: {sorted(unknown)}')
        known = set(zip(payments.synthetic_member_id, payments.payment_id))
        if event_id is None:
            missing = [(mid, pid) for mid, r in member_info.iterrows()
                       for pid in r.scam_payment_set if (mid, pid) not in known]
            if missing:
                raise ValueError(f'Scam labels have no matching outgoing initiation: {missing}')
    histories = dict(tuple(events.groupby('synthetic_member_id', sort=False)))
    metadata, features = [], []
    for _, p in payments.iterrows():
        info = {c: p[c] for c in META}
        if member_info is not None:
            member = member_info.loc[p.synthetic_member_id]
            info['target'] = int(member.member_target) if p.payment_id in member.scam_payment_set else 0
            info['is_legitimate_lookalike'] = member.is_legitimate_lookalike == 'Yes'
        metadata.append(info)
        features.append(build_feature_row(p, histories[p.synthetic_member_id], config))
    return pd.DataFrame(metadata), pd.DataFrame(features)
