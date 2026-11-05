# Take-profit / stop-loss exits as a first-passage problem

Applied Stochastic Modelling project on stock-market trade exits.

An investor buys a stock and places a take-profit order at +a% and a stop-loss at −b%. Which is hit first, and how long does the trade last?
The log-price is modelled as a random walk / Brownian motion with drift, so the question becomes a first-passage (gambler's ruin) problem.
The repository contains the models, the Monte Carlo simulations, the real-data tests and the behavioural / strategy extensions.

## Questions and short answers

| Question | Where | Answer in one line |
|---|---|---|
| Can closed-form first-passage formulas predict exit probability and duration? | `trade_exit.py` | Yes, after shifting the barriers by 0.5826σ because stops are only checked at daily closes. |
| Do they match real trades, including out of sample? | `trade_exit.py`, `robust_model.py` | Success probabilities are well calibrated; the fair-game rule l/(u+l) is rejected. Exit times are the weaker part. |
| What do human exit habits (stop overrides, impulse sales) change? | `human_factors.py` (H1) | Risk, not average growth (Wald's identity). |
| Do fear (VIX) or extrapolated beliefs matter? | `human_factors.py` (H2, H3) | Fear predicts duration; the "overconfidence" Brier penalty is reproduced by an iid simulation, i.e. it is estimation noise. |
| Can any behaviour be a valid strategy? | `strategy_tournament.py` | No: 11 archetypes, none survives a Holm-corrected circular-shift test with costs. |
| Can a behaviour be engineered to be optimal? | `optimal_sizing.py` | Volatility-targeted sizing cuts drawdown but not Sharpe; Kelly (capped) is mostly "stay invested". |
| Does it hold on a wider universe? | `fetch_wide.py`, `robust_model.py`, `*_wide` runs | 36 series (Indian and US stocks, indices, gold, bonds): same conclusions. |

## Models

- **M0** fair-game random walk: P(target first) = l / (u + l), where u = ln(1+a) and l = −ln(1−b).
- **M1** Brownian motion with drift μ and volatility σ: closed-form P(target first) and E[exit time], with the Broadie–Glasserman–Kou continuity correction for daily monitoring.
- **M3** behavioural exits: stop-loss override with probability q, daily impulse-sale hazard c (higher in profit than in loss), capitulation at −25%.

Items marked `[EXTRA]` in the code are outside the core course syllabus (continuity correction, KS test, Kelly criterion, volatility targeting, Holm correction, circular-shift null).

## Repository layout

```
code/
  fetch_data.py            download the six core series         -> data/basket_close.csv
  fetch_human.py           download India VIX and CBOE VIX      -> data/sentiment.csv
  fetch_wide.py            download the 36-series universe      -> data/wide_close.csv
  trade_exit.py            M0/M1 theory vs simulation vs real trades (in-sample and out-of-sample)
  human_factors.py         H1 behavioural exits, H2 fear, H3 extrapolated beliefs (with an iid control)
  strategy_tournament.py   11 behavioural archetypes tested against a circular-shift null
  optimal_sizing.py        buy-and-hold vs Kelly vs volatility targeting
  robust_model.py          the model checks repeated on the 36-series universe
  fig_merged.py            data figure and out-of-sample calibration figure
  style.py                 shared matplotlib style (imported by the scripts that draw figures)
data/                      price and fear-index CSVs (re-created by the fetch scripts)
figures/                   PNG figures written by the scripts
results_*.json             numerical results written by the scripts
```

## Reproducing the results

Python 3.11+.

```bash
pip install -r requirements.txt

# 1. data (optional: the CSVs are already in data/; running these overwrites them with fresh downloads)
python code/fetch_data.py
python code/fetch_human.py
python code/fetch_wide.py

# 2. analysis; run from the repository root, in this order
python code/trade_exit.py            # results_trade.json, figures/t2_theory_sim.png
python code/human_factors.py         # results_human.json, figures/h*.png
python code/strategy_tournament.py   # results_tournament.json, figures/s1..s3
python code/optimal_sizing.py        # results_sizing.json, figures/o1_sizing.png
python code/robust_model.py          # results_wide_model.json, figures/w1_wide_model.png
python code/fig_merged.py            # figures/t1m_data.png, figures/t4m_oos.png

# 3. the wider universe for the tournament and the sizing study
ASM_WIDE=1 python code/strategy_tournament.py   # results_tournament_wide.json (Windows PowerShell: $env:ASM_WIDE=1)
ASM_WIDE=1 python code/optimal_sizing.py        # results_sizing_wide.json
```

All random seeds are fixed, so the numbers reproduce exactly on the same library versions. `strategy_tournament.py` is the slowest script (it uses all CPU cores through joblib).
Downloads come from Yahoo Finance, so a fresh download can differ from the bundled CSVs (prices are adjusted retroactively and the end date moves).

## Method notes

- **Trades.** Sequential and non-overlapping: enter at a close, exit at the first close where the cumulative log-return reaches +u or −l. The unfinished last trade is dropped.
- **Out-of-sample test.** Parameters are fitted on the first half of each series; trades are taken in the second half with frozen parameters. Calibration uses a binomial z-score and the Brier score.
- **Tournament null.** Returns are rolled by a random offset of at least 250 days against the entry signals (1000 repetitions), which keeps drift and volatility clustering but breaks the link between signal and future. P-values are Holm-corrected; costs are 0.1% per side. A strategy counts as valid only if Holm p < 0.05, the net edge is positive in both halves of the sample, and it is positive in at least 4 of the 6 series.
- **Positive control.** The same test is run on simulated iid, trending and mean-reverting worlds to show it can detect an edge when one exists.
- **Sizing.** Weekly rebalancing, 0.1% cost per unit of turnover, no leverage, cash earns 0. Drawdown is reported as a percentage of peak wealth.

## Limitations

- The archetype and behaviour parameters (q, c and so on) are assumptions, not estimates.
- The tournament has limited power against slow trends or reversals.
- The core six series are mostly survivors; the 36-series universe reduces but does not remove survivorship bias, since fully delisted companies are not available.
- Daily closes only: a real standing order would trigger intraday.
- A constant-variance random walk misses volatility clustering and fat tails, which is where exit-time predictions are weakest.
