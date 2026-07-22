# Validation Playbook

## Non-Negotiables

- Split by time, never random rows.
- Keep train, validation, and test windows chronologically honest.
- Purge overlapping labels around split boundaries.
- Embargo after test windows when labels or features can leak.
- Include transaction costs and turnover from the first test.
- Rank models by out-of-sample behavior, not in-sample fit.

## First-Phase Gates

A model can move from research to paper trading only if it clears these gates:

- Positive out-of-sample Sharpe after costs.
- Max drawdown below the configured limit.
- Turnover below the capacity-adjusted limit.
- Probabilistic Sharpe above the configured confidence threshold.
- Deflated Sharpe above the configured confidence threshold when many trials were tested.
- Stable performance across at least two market regimes.
- No single asset or month explains most of the result.

## Model Tournament Loop

1. Generate candidate signals.
2. Convert scores to normalized target weights.
3. Apply volatility targeting and cost model.
4. Run walk-forward validation.
5. Compute metrics for each split and aggregate.
6. Penalize fragile models.
7. Promote only models that pass gates.
8. Feed promoted models into the regime router.

## Red Flags

- High Sharpe with extreme turnover.
- Strong performance concentrated in one month or one asset.
- Parameter sets clustered at search boundaries.
- Results that vanish after costs.
- Model performance that decays immediately after selection.
- Large gap between validation and final test performance.
