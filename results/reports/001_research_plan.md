# XAUUSD Research Plan 001

## Summary Priority Table

| Rank | Hypothesis | Structural Basis | Primary Risk | Proceed |
|------|------------|------------------|--------------|---------|
| 1 | Asian Range -> London Breakout | Liquidity transition between sessions | False break / spread | Yes, first |
| 2 | London Trend -> NY Reversal | Participant turnover, position unwinding | Regime dependence | Yes, second |
| 3 | PDH/PDL Stop Hunt Rejection | Stop cluster liquidity mechanics | Breakout misclassification | Yes, with 1m data |
| 4 | Post-News Spike Fade | Algorithmic overshoot + human reversal | Small sample, execution | Low confidence |
| 5 | Volatility Compression Expansion | Universal volatility regime cycling | No directional signal | Do not test standalone |

## Universal Kill Conditions

1. Three consecutive refinements show no robustness improvement
2. OOS expectancy < 60% of IS expectancy
3. Edge disappears when spread assumption increases from 30 to 50 pips
4. Parameter sensitivity test: moving any single parameter by 10% causes material degradation
5. Strategy can no longer be explained in two sentences of plain English

## Recommended Execution Order

1. H1 baseline
2. H1 with range-width filter and session-time refinement
3. H2 baseline
4. H2 regime split
5. H3 only after separate data pipeline exists
6. Cross-hypothesis filter combinations only if H1 or H2 survive
7. H4 only if trade count and data quality allow
