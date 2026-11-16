"""Immutable global ledger snapshot, binding dispersion to recorded outcomes."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math

@dataclass(frozen=True)
class LedgerSnapshot:
    global_count: int
    family_counts: tuple[tuple[str, int], ...]
    dispersion_receipt: tuple[tuple[str, float], ...]
    ledger_hash: str
    taken_at: str

    @property
    def sharpes(self):
        return tuple(value for _,value in self.dispersion_receipt)


def take_snapshot(ledger):
    with ledger._connect() as con:
        con.execute('BEGIN')
        tables = {name: [dict(r) for r in con.execute(f'SELECT * FROM {name} ORDER BY rowid')]
                  for name in ('trials','trial_events','legacy_imports','ledger_imports','candidate_return_series','evidence')}
    families = {}
    for row in tables['trials']:
        families[row['family']] = families.get(row['family'],0)+1
    for row in tables['legacy_imports']:
        key = 'UNCLASSIFIED_LEGACY:' + row['campaign']
        families[key] = families.get(key,0)+row['attempt_count']
    receipt = {}
    for row in tables['trial_events']:
        if row['event_type'] in ('outcome','correction'):
            payload = json.loads(row['payload_json'])
            # A correction replaces the outcome's dispersion, including removal.
            receipt.pop(row['trial_id'],None)
            value = payload.get('sharpe')
            if value is not None:
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise ValueError('invalid ledger Sharpe')
                receipt[row['trial_id']] = float(value)
    count = len(tables['trials'])+sum(r['attempt_count'] for r in tables['legacy_imports'])
    digest = hashlib.sha256(json.dumps(tables,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return LedgerSnapshot(count,tuple(sorted(families.items())),tuple(sorted(receipt.items())),digest,datetime.now(timezone.utc).isoformat())
