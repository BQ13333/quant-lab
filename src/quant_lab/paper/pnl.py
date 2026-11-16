"""The intended and actual-fill P&L books have different cost contracts."""
from dataclasses import dataclass
import math
from .shortfall import Shortfall

@dataclass(frozen=True)
class IntendedBook:
    fill_basis: str
    def net(self, gross, shortfall):
        if type(shortfall) is not Shortfall or self.fill_basis != shortfall.ref or not math.isfinite(gross):
            raise ValueError('fill basis mismatch or invalid shortfall')
        return gross-shortfall.total

@dataclass(frozen=True)
class ActualFillBook:
    gross: float
    def net(self, fees):
        """Only fees not yet booked; no shortfall parameter is accepted."""
        if not math.isfinite(self.gross) or not math.isfinite(fees) or fees < 0:
            raise ValueError('invalid P&L/fees')
        return self.gross-fees
