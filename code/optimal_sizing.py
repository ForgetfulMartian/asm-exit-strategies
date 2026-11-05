"""
Can a behaviour be ENGINEERED to be optimal?  Exit rules cannot raise growth (Wald), so the lever is HOW MUCH to hold.
  B&H   always 100% invested
  KELLY  f = clip(mu_a / sigma^2, 0, 1) from trailing 500 days (log-utility optimum, capped at no leverage)
  VT15 / VT10  volatility targeting: f = min(1, target / (20-day vol, annualised)), target 15% / 10%
  VTF   VT15 with a 25% floor (never fully out)
Weekly rebalance, 0.1% cost per unit of turnover, weight set at close t applies to day t+1.  No leverage, cash earns 0.
Null: roll returns by a random offset against the (original) weights, 1000 times -> p-value for Sharpe / max-drawdown / Calmar improvement.
Run: python code/optimal_sizing.py -> results_sizing.json, figures/o1_sizing.png     [EXTRA] Kelly criterion, volatility targeting.
"""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # shared figure style
WIDE = os.environ.get("ASM_WIDE") == "1"; OUT = "_wide" if WIDE else ""
px = pd.read_csv("data/wide_close.csv" if WIDE else "data/basket_close.csv", index_col=0, parse_dates=True)
NAMES = {"^NSEI": "NIFTY 50", "RELIANCE.NS": "RELIANCE", "TCS.NS": "TCS", "HDFCBANK.NS": "HDFC Bank", "AAPL": "AAPL", "TSLA": "TSLA"}
if WIDE: NAMES = {c: c for c in px.columns}
series = {nm: np.log(px[tk].dropna()).diff().dropna() for tk, nm in NAMES.items()}
COST = 0.001; REB = 5
def weights(r, kind):
    v = r.rolling(20).std()*np.sqrt(252)
    if kind == "KELLY":
        m = r.rolling(500).mean(); s2 = r.rolling(500).var(); w = ((m + s2/2)/s2).clip(0, 1)
    elif kind == "VT15": w = (0.15/v).clip(upper=1)
    elif kind == "VT10": w = (0.10/v).clip(upper=1)
    elif kind == "VTF": w = (0.15/v).clip(lower=0.25, upper=1)
    else: w = pd.Series(1.0, index=r.index)
    w = w.shift(1)                                    # known at the previous close
    ok = w.notna(); first = np.argmax(ok.values); w = w.values.copy()
    for i in range(first, len(w)):                    # weekly rebalance: hold the weight between rebalance days
        if (i - first) % REB: w[i] = w[i-1]
    return w, first
def curve(rv, w):
    cost = COST*np.abs(np.diff(w, prepend=w[0])); return w*rv - cost
def sharpe(a): return a.mean()/a.std()*np.sqrt(252)
def mdd(a): e = np.cumsum(a); return float((e - np.maximum.accumulate(e)).min())
def metr(a): return dict(sharpe=sharpe(a), cagr=a.mean()*252, mdd=mdd(a), mddp=1 - float(np.exp(mdd(a))), calmar=a.mean()*252/abs(mdd(a)))   # mdd: drawdown in cumulative log-return; mddp: same as % of peak wealth
KINDS = ["BH", "KELLY", "VT15", "VT10", "VTF"]
W = {nm: {k: weights(r, k) for k in KINDS} for nm, r in series.items()}
def run(roll_u):
    out = {k: {} for k in KINDS}
    for nm, r in series.items():
        first = max(W[nm][k][1] for k in KINDS); rv = r.values[first:]; n = len(rv)
        if roll_u is not None: rv = np.roll(rv, 250 + int(roll_u*(n - 500)))
        base = metr(curve(rv, np.ones(n)))
        for k in KINDS:
            w = W[nm][k][0][first:]; m = metr(curve(rv, w)); m["expo"] = float(w.mean())
            out[k][nm] = dict(m, dsh=m["sharpe"] - base["sharpe"], dmdd=m["mdd"] - base["mdd"], dcalmar=m["calmar"] - base["calmar"], dcagr=m["cagr"] - base["cagr"])
    return out
obs = run(None); NR = 1000; rng = np.random.default_rng(31)
nul = [run(float(u)) for u in rng.random(NR)]
R = {}
for k in KINDS:
    o = {m: float(np.mean([obs[k][nm][m] for nm in series])) for m in ("sharpe", "cagr", "mdd", "mddp", "calmar", "expo", "dsh", "dmdd", "dcalmar", "dcagr")}
    for m in ("dsh", "dmdd", "dcalmar"):
        nv = np.array([np.mean([x[k][nm][m] for nm in series]) for x in nul]); o["p_" + m] = float((1 + (nv >= o[m]).sum())/(NR + 1)); o["null_" + m] = [float(np.percentile(nv, 2.5)), float(np.percentile(nv, 97.5))]
    o["pos_series_dsh"] = int(sum(obs[k][nm]["dsh"] > 0 for nm in series))
    # halves
    h = {}
    for nm, r in series.items():
        first = max(W[nm][kk][1] for kk in KINDS); rv = r.values[first:]; w = W[nm][k][0][first:]; m_ = len(rv)//2
        h[nm] = [sharpe(curve(rv, w)[sl]) - sharpe(curve(rv, np.ones(len(rv)))[sl]) for sl in (slice(0, m_), slice(m_, None))]
    o["dsh_h1"] = float(np.mean([v[0] for v in h.values()])); o["dsh_h2"] = float(np.mean([v[1] for v in h.values()]))
    o["per_series_dsh"] = {nm: float(obs[k][nm]["dsh"]) for nm in series}; R[k] = o
json.dump(R, open(f"results_sizing{OUT}.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a, (m, t) in zip(ax, (("dsh", "Sharpe minus buy-and-hold"), ("dmdd", "Max drawdown minus buy-and-hold (log scale; positive = shallower)"))):
    for i, k in enumerate(KINDS[1:]):
        a.plot(R[k]["null_" + m], [i, i], color="#999", lw=6, alpha=.5); a.plot(R[k][m], i, "o", color="#1f4e79")
    a.axvline(0, color="k", lw=.8); a.set_yticks(range(4)); a.set_yticklabels(KINDS[1:]); a.set_title(t, fontsize=8)
plt.tight_layout(); plt.savefig(f"figures/o1_sizing{OUT}.png"); plt.close()
for k, o in R.items(): print(k, {m: round(o[m], 3) for m in o if not isinstance(o[m], (list, dict))})
