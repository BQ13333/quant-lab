"""Fail-closed, point-in-time market rules. Currency is market-local."""
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
import json
import math

class MarketRules:
    def __init__(self, entries=None):
        self.entries=json.loads(Path(__file__).with_name('rules_v1.json').read_text()) if entries is None else entries
        for market,fields in self.entries.items():
            if market not in {'MY','CN','HK'}: raise ValueError('unsupported market')
            for records in fields.values():
                for r in records:
                    if not {'value','effective_from','source_url','source_kind'} <= r.keys() or r['source_kind'] not in {'PRIMARY','SECONDARY','NONE'} or (r['source_kind']=='PRIMARY' and not r['source_url']):
                        raise ValueError('rule provenance required')
                    Date.fromisoformat(r['effective_from'])

    def for_date(self, market, date, *, admission=True):
        date=Date.fromisoformat(str(date)[:10])
        if market not in self.entries: raise ValueError('unknown market')
        result={}
        for field,records in self.entries[market].items():
            eligible=[r for r in records if Date.fromisoformat(r['effective_from']) <= date]
            if not eligible: continue
            r=max(eligible,key=lambda r:r['effective_from'])
            # Do not fall back to an older PRIMARY when a newer change is unverified.
            if r.get('effective_until') and date > Date.fromisoformat(r['effective_until']): continue
            if not admission or r['source_kind']=='PRIMARY': result[field]=r['value']
        return result

    def eligible_name(self, market, name, date):
        if market not in self.entries: raise ValueError('unknown market')
        if market=='CN' and 'ST' in name.upper():
            return 'st_band' in self.for_date(market,date)
        return True


def explicit_cost(market, side, notional, date, *, rules=None):
    if side not in {'BUY','SELL'} or not math.isfinite(notional) or notional <= 0:
        raise ValueError('invalid cost order')
    r=(rules or MarketRules()).for_date(market,date)
    def need(key):
        if key not in r: raise ValueError(f'missing primary rule: {market}/{key}')
        return r[key]
    if market=='MY':
        return min(math.ceil(notional/need('stamp_unit'))*need('stamp_per_unit'),need('stamp_cap'))+min(notional*need('clearing_rate'),need('clearing_cap'))
    if market=='HK':
        return math.ceil(notional*need('stamp_rate'))+notional*(need('trading_rate')+need('sfc_rate')+need('afrc_rate'))+max(need('settlement_min'),min(need('settlement_cap'),notional*need('settlement_rate')))
    if market=='CN':
        # Northbound legs are not yet verified: never invent a complete charge.
        stamp=notional*need('stamp_rate') if side=='SELL' else 0.
        return stamp+notional*need('handling_rate')+notional*need('management_rate')+notional*need('transfer_rate')
    raise ValueError('unsupported market')

@dataclass(frozen=True)
class Commission:
    rate: float
    minimum: float
    source_note: str
    assumption: str = 'REGISTERED_LAB_ASSUMPTION'
    def __post_init__(self):
        if not math.isfinite(self.rate) or not math.isfinite(self.minimum) or min(self.rate,self.minimum)<0 or not self.source_note:
            raise ValueError('invalid registered commission')

class LabCostContract:
    label='LAB_SIMULATED_COST'
    def __init__(self, contract_id='LAB_SIM_COST_v1', *, commissions=None, rules=None):
        if contract_id!='LAB_SIM_COST_v1': raise ValueError('unknown cost contract')
        self.commissions=dict(commissions or {})
        self.rules=rules or MarketRules()
    def cost(self,market,side,notional,date,*,spread,impact):
        c=self.commissions.get(market)
        if type(c) is not Commission: raise ValueError('no registered commission entry')
        if not all(math.isfinite(x) and x >= 0 for x in (spread,impact)): raise ValueError('invalid model costs')
        return explicit_cost(market,side,notional,date,rules=self.rules)+max(c.minimum,notional*c.rate)+notional*(spread+impact)
    def require_stage(self,stage):
        if stage not in {'CANDIDATE','VALIDATED','PAPER'}:
            raise ValueError('LAB_SIMULATED_COST refused by trading gate')
