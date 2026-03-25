# XAUUSD Research Rules

## Objective
Find robust, executable XAUUSD trading strategies with positive expectancy after realistic costs.

## Core principles
- Prefer simple, explainable market behavior over complex indicator stacking.
- No strategy is accepted from a single backtest window.
- Every strategy must be tested in-sample, out-of-sample, and walk-forward.
- All reviews must include spread, slippage, and execution realism.
- Reject strategies that only work on one narrow parameter cluster.
- Reject strategies with obvious lookahead bias or data leakage.
- Reject strategies whose edge disappears after costs.
- Always show failure modes before proposing improvements.

## Research kill-switch
Terminate the current strategy line immediately if any of the following occur:
1. Three consecutive refinements fail to improve robustness.
2. Out-of-sample expectancy falls below 60% of in-sample expectancy.
3. Nearby parameter values produce materially worse results.
4. The edge disappears after realistic spread/slippage assumptions.
5. Trade count is too low to support confidence.
6. The strategy can no longer be explained clearly in plain English.

## Coding rules
- Separate signal generation, execution, and risk management.
- Keep parameters centralized and typed.
- Add debug logs for entry blocks, entries, exits, and risk decisions.
- Add machine-readable summary output for each test.
- Avoid hidden assumptions and magic constants.

## Research rules
- Start with behavior hypotheses, not indicators.
- Test one hypothesis family at a time.
- Build the simplest baseline first.
- Add one refinement at a time.
- Track what changed and why.
- Attack every result as if it is wrong until proven robust.

## Review standard
Every strategy review must challenge:
1. overfitting risk
2. regime dependence
3. execution fragility
4. parameter instability
5. sample-size weakness
6. session clock mistakes
7. spread/slippage sensitivity
8. long/short asymmetry
