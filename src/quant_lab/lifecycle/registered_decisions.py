"""Lab policy decisions Ben accepted on 16 Nov 2026 (Asia/Kuala_Lumpur).

These are registered facts, not new engines. C17 itself is built in the
private 08-quantitative repo as Holm gate contract v2, report-only.
"""

from __future__ import annotations

from typing import Final

# 1. C17 — Holm gate, report-only, not blocking. Not Benjamini–Yekutieli.
C17_METHOD: Final = "HOLM_STEP_DOWN"
C17_MODE: Final = "report-only"
C17_BLOCKING: Final = False

# 2. Working charter numbers adopted as the working charter until a later review.
CHARTER_FLOORS_MYR: Final = (1_485_000, 756_250, 312_500)
ACTIVE_VOL: Final = 0.0515
INFORMATION_RATIO: Final = 0.29
BURSA_TICKET_COMMISSION_RATE: Final = 0.0042
BURSA_TICKET_COMMISSION_FLOOR_MYR: Final = 12.0

# 3. False-pause target: per sleeve, not a family hard gate.
FALSE_PAUSE_SCOPE: Final = "per_sleeve"
FALSE_PAUSE_MAXIMUM_RATE: Final = 0.20
FALSE_PAUSE_HORIZON_MONTHS: Final = 60
FALSE_PAUSE_FAMILY_RATE_MODE: Final = "report-only"
# If the per-sleeve 20% cap is in force, the observed mean in a 2,000-run
# check must sit at or below this figure (margin 3.45959 points).
FALSE_PAUSE_OBSERVED_MEAN_MAX: Final = 0.1654041

# 4. C10 live-history cost: only PRIMARY / CAPTURED. VENDOR_PIT is later.
LIVE_HISTORY_COST_GRADES: Final = frozenset({"PRIMARY", "CAPTURED"})
VENDOR_PIT_STATUS: Final = "later_option_not_admissible"

# 5. No spend this week (recorded; nothing to encode in a runtime gate).
SPEND_THIS_WEEK: Final = "none"
