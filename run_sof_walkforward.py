"""
Walk-forward for Short-Only Macro-Filtered Fade.
1-year training, 6-month test, 6-month step. 12 segments, 2019-01 → 2024-12.
"""

import json, math, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from run_backtest import load_ohlcv
from src.strategies.short_only_fade.engine import run_sof_backtest
from src.strategies.short_only_fade.params import SOFParams
from src.strategies.asian_range_breakout.execution import EXIT_SL, EXIT_SL_GAP, EXIT_TIME, EXIT_TP
from src.strategies.asian_range_breakout.results import _approx_sharpe, _max_drawdown_pct, _mean

WF_SEGMENTS = [
    ("2018-01-01","2018-12-31","2019-01-01","2019-06-30"),
    ("2018-07-01","2019-06-30","2019-07-01","2019-12-31"),
    ("2019-01-01","2019-12-31","2020-01-01","2020-06-30"),
    ("2019-07-01","2020-06-30","2020-07-01","2020-12-31"),
    ("2020-01-01","2020-12-31","2021-01-01","2021-06-30"),
    ("2020-07-01","2021-06-30","2021-07-01","2021-12-31"),
    ("2021-01-01","2021-12-31","2022-01-01","2022-06-30"),
    ("2021-07-01","2022-06-30","2022-07-01","2022-12-31"),
    ("2022-01-01","2022-12-31","2023-01-01","2023-06-30"),
    ("2022-07-01","2023-06-30","2023-07-01","2023-12-31"),
    ("2023-01-01","2023-12-31","2024-01-01","2024-06-30"),
    ("2023-07-01","2024-06-30","2024-07-01","2024-12-31"),
]
PARAMS = SOFParams(ma_period=20, slope_lookback=5, spread_price=0.30,
                   slippage_price=0.20, risk_pct=0.01, initial_equity=100_000.0)

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "results/sof_walkforward.json"
    df_is  = load_ohlcv("data/xauusd_15m_2018-01-01_2021-12-31.csv")
    df_oos = load_ohlcv("data/xauusd_15m_2022-01-01_2024-12-31.csv")
    df_all = pd.concat([df_is, df_oos]).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]

    segs, all_r_flat = [], []
    yearly: dict[str, list] = defaultdict(list)

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(WF_SEGMENTS, 1):
        ts = pd.Timestamp(te_s, tz="UTC")
        te = pd.Timestamp(te_e, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
        df_w = df_all[(df_all.index >= pd.Timestamp(tr_s, tz="UTC")) &
                      (df_all.index <= te + pd.Timedelta(hours=26))]
        trades = [t for t in run_sof_backtest(df_w, PARAMS)
                  if ts <= t.setup.entry_timestamp <= te]

        r_vals = [t.realized_r for t in trades]
        wins   = [r for r in r_vals if r > 0]
        losses = [r for r in r_vals if r <= 0]
        pf     = (sum(wins)/abs(sum(losses))) if losses else float("inf")
        exp    = _mean(r_vals) if r_vals else 0.0
        pos    = exp > 0

        for t in trades:
            yr = str(t.setup.entry_timestamp.year)
            yearly[yr].append(t.realized_r)
        all_r_flat.extend(r_vals)

        n = len(trades)
        sign = "+" if pos else "-"
        print(f"  Seg {i:02d} [{te_s} → {te_e}]  {sign}  n={n:3d}  "
              f"exp={exp:+.4f}R  PF={round(pf,3) if not math.isinf(pf) else 'inf'}  "
              f"TP={sum(1 for t in trades if t.exit_reason==EXIT_TP)}  "
              f"SL={sum(1 for t in trades if t.exit_reason in (EXIT_SL,EXIT_SL_GAP))}  "
              f"TIME={sum(1 for t in trades if t.exit_reason==EXIT_TIME)}", flush=True)

        segs.append({"segment":i,"test":f"{te_s}→{te_e}","trades":n,
                     "expectancy_r":round(exp,4),"positive":pos,
                     "profit_factor":round(pf,4) if not math.isinf(pf) else "inf"})

    pos_n = sum(1 for s in segs if s["positive"])
    comb_exp = _mean(all_r_flat) if all_r_flat else 0.0
    all_w = [r for r in all_r_flat if r > 0]
    all_l = [r for r in all_r_flat if r <= 0]
    pf_all = (sum(all_w)/abs(sum(all_l))) if all_l else float("inf")

    print(f"\n{'='*70}")
    print(f"  WF AGGREGATE  (12 segs, 2019-01→2024-12)")
    print(f"  Positive: {pos_n}/12 ({pos_n/12:.0%})  |  Total trades: {len(all_r_flat)}")
    print(f"  Combined exp: {comb_exp:+.4f}R  |  PF: {round(pf_all,4) if not math.isinf(pf_all) else 'inf'}")
    print(f"  Sharpe: {_approx_sharpe(all_r_flat):+.3f}")
    print(f"  Seg values: {[s['expectancy_r'] for s in segs]}")
    print(f"\n  Yearly:")
    for yr in sorted(yearly):
        rs = yearly[yr]; w = [r for r in rs if r > 0]
        print(f"  {yr}: {len(rs)}T  WR={len(w)/len(rs):.1%}  exp={_mean(rs):+.4f}R")

    def _clean(obj):
        if isinstance(obj,float) and math.isinf(obj): return "Infinity"
        if isinstance(obj,dict): return {k:_clean(v) for k,v in obj.items()}
        if isinstance(obj,list): return [_clean(v) for v in obj]
        return obj

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(_clean({
        "strategy":"XAUUSD_Short_Only_Macro_Filtered_Fade","version":"v1.0_walkforward",
        "segments":segs,
        "aggregate":{"positive":pos_n,"negative":12-pos_n,"hit_rate":round(pos_n/12,4),
                     "total_trades":len(all_r_flat),"combined_expectancy_r":round(comb_exp,4),
                     "profit_factor":round(pf_all,4) if not math.isinf(pf_all) else "Infinity",
                     "seg_exp_values":[s["expectancy_r"] for s in segs]},
    }),indent=2))
    print(f"\n  Saved → {out_path}")

if __name__ == "__main__": main()
