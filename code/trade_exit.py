"""
Real-world problem: a retail investor buys a stock and sets a take-profit (+a) and a stop-loss (-b).
  * What is P(target is hit before stop)?   * How many days until the trade ends?
Models:  M0 fair-game random walk (gambler's ruin)           P = l/(u+l)
         M1 Brownian motion with drift (i.i.d. normal log-returns), closed-form first passage
Data: daily closes of 6 series (NIFTY 50, RELIANCE, TCS, HDFCBANK, AAPL, TSLA), 2010-2026 (Yahoo Finance).
Run: python code/trade_exit.py  -> results_trade.json, figures/t*.png
[EXTRA] marks items outside the core course syllabus.
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # shared figure style
from scipy import stats                          # normal cdf/pdf, chi-square and KS p-values only

rng = np.random.default_rng(2026)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 9})
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png"); plt.close()

px = pd.read_csv("data/basket_close.csv", index_col=0, parse_dates=True)
NAMES = {"^NSEI": "NIFTY 50", "RELIANCE.NS": "RELIANCE", "TCS.NS": "TCS", "HDFCBANK.NS": "HDFC Bank", "AAPL": "AAPL", "TSLA": "TSLA"}
SCEN = [(0.10, 0.05), (0.15, 0.10), (0.20, 0.10), (0.10, 0.10)]       # (take-profit %, stop-loss %)
HEAD = (0.15, 0.10)
WARM = 20                                        # first trade starts after a 20-day warm-up (same convention as the other scripts)
BGK = 0.5826                                     # [EXTRA] discrete-monitoring barrier shift constant (Broadie-Glasserman-Kou 1997)
R = {"scenarios": SCEN, "headline": HEAD}

# ------------------------------------------------------------------ theory (Brownian motion with drift)
def p_up_cont(mu, sig, u, l):
    """P(hit +u before -l) for X_t = mu t + sig W_t, start 0 (scale-function / gambler's-ruin argument)."""
    if abs(mu) < 1e-10: return l/(u + l)
    k = 2*mu/sig**2
    return (1 - np.exp(k*l)) / (np.exp(-k*u) - np.exp(k*l))
def et_cont(mu, sig, u, l):
    p = p_up_cont(mu, sig, u, l)
    return u*l/sig**2 if abs(mu) < 1e-10 else (u*p - l*(1-p))/mu      # Wald: E[X_T] = mu E[T]
def p_up_disc(mu, sig, u, l):                    # daily closes only: push barriers out by 0.5826 sigma
    return p_up_cont(mu, sig, u + BGK*sig, l + BGK*sig)
def et_disc(mu, sig, u, l):
    return et_cont(mu, sig, u + BGK*sig, l + BGK*sig)

def simulate(mu, sig, u, l, N, Hmax=4000):
    """N exit-times of the daily Gaussian random walk with drift. Returns (outcome[1=target,0=stop,-1=unfinished], days)."""
    x = np.zeros(N); out = -np.ones(N, int); T = np.zeros(N, int); alive = np.arange(N)
    for t in range(1, Hmax + 1):
        if alive.size == 0: break
        x[alive] += mu + sig*rng.standard_normal(alive.size)
        hit_u = x[alive] >= u; hit_l = x[alive] <= -l; done = hit_u | hit_l
        out[alive[hit_u]] = 1; out[alive[hit_l & ~hit_u]] = 0; T[alive[done]] = t; alive = alive[~done]
    return out, T

# ------------------------------------------------------------------ model fitting & trade extraction
def fit(r):
    """r: pandas Series of training log-returns. MLE of the Brownian-motion parameters (per day)."""
    return dict(mu=float(r.mean()), sig=float(r.std(ddof=0)), n=len(r))

def trades(rv, u, l, lo, hi):
    """Sequential non-overlapping trades starting at index lo; each enters at the close of day i, first return seen is r[i+1]; trade ends at
       first day where cumulative log-return >= u (target) or <= -l (stop). The last unfinished trade is dropped."""
    cs = np.cumsum(rv); out = []; i = lo
    while i < hi - 1:
        seg = cs[i+1:hi] - cs[i]; hit = (seg >= u) | (seg <= -l)
        if not hit.any(): break
        k = int(np.argmax(hit)); out.append((i, k + 1, int(seg[k] >= u))); i = i + k + 1
    return pd.DataFrame(out, columns=["start", "days", "target"])

# ------------------------------------------------------------------ main loop
series = {}
for tk, nm in NAMES.items():
    s = px[tk].dropna(); series[nm] = np.log(s).diff().dropna()

# ---- table 1: descriptive & assumption checks
def acf1(x): x = x - x.mean(); return float((x[:-1]*x[1:]).sum()/(x**2).sum())
desc = {}
for nm, r in series.items():
    f = fit(r); v = r.values
    desc[nm] = dict(n=len(r), start=str(r.index[0].date()), mu=f["mu"], sig=f["sig"], sig_ann=f["sig"]*np.sqrt(252), kurt=float(stats.kurtosis(v)),
                    acf_r=acf1(v), acf_r2=acf1(v**2))
R["desc"] = desc

# ---- A. theory vs simulation (full-sample parameters, constant-sigma model M1)
A = {}
for nm, r in series.items():
    f = fit(r); A[nm] = {}
    for (ta, sl) in SCEN:
        u, l = np.log(1+ta), -np.log(1-sl)
        o, T = simulate(f["mu"], f["sig"], u, l, 40000)
        fin = o >= 0
        A[nm][f"{ta}/{sl}"] = dict(p_cont=p_up_cont(f["mu"], f["sig"], u, l), p_disc=p_up_disc(f["mu"], f["sig"], u, l), p_sim=float((o[fin] == 1).mean()),
                                   p_se=float(np.sqrt(.25/fin.sum())), et_cont=et_cont(f["mu"], f["sig"], u, l), et_disc=et_disc(f["mu"], f["sig"], u, l),
                                   et_sim=float(T[fin].mean()), p_fair=l/(u+l), frac_unfinished=float(1-fin.mean()))
R["theory_vs_sim"] = A

# ---- B. in-sample: all data, model vs real sequential trades
B = {}
for nm, r in series.items():
    f = fit(r); rv = r.values; B[nm] = {}
    for (ta, sl) in SCEN:
        u, l = np.log(1+ta), -np.log(1-sl); tr = trades(rv, u, l, WARM, len(rv)); n_ = len(tr)
        B[nm][f"{ta}/{sl}"] = dict(n=n_, emp_p=float(tr.target.mean()), emp_days=float(tr.days.mean()), emp_med=float(tr.days.median()),
            p0=l/(u+l), p1=p_up_disc(f["mu"], f["sig"], u, l), et1=et_disc(f["mu"], f["sig"], u, l),
            se=float(np.sqrt(max(tr.target.mean()*(1-tr.target.mean()), 1e-9)/max(n_, 1))), days_se=float(tr.days.std()/np.sqrt(max(n_, 1))))
        if (ta, sl) == HEAD:                                   # [EXTRA] 2-sample KS between simulated and empirical exit times
            o, T = simulate(f["mu"], f["sig"], u, l, 20000); ks = stats.ks_2samp(T[o >= 0], tr.days.values)
            B[nm][f"{ta}/{sl}"]["ks_M1"] = [float(ks.statistic), float(ks.pvalue)]
R["in_sample"] = B

# ---- C. out-of-sample: fit on first half of each series, trade on second half (parameters frozen)
pool = {f"{a}/{b}": [] for a, b in SCEN}
for nm, r in series.items():
    h = len(r)//2; f = fit(r.iloc[:h]); rv = r.values
    for (ta, sl) in SCEN:
        u, l = np.log(1+ta), -np.log(1-sl); tr = trades(rv, u, l, max(h, WARM), len(rv))
        tr["stock"] = nm; tr["p0"] = l/(u+l); tr["p1"] = p_up_disc(f["mu"], f["sig"], u, l); tr["e1"] = et_disc(f["mu"], f["sig"], u, l); pool[f"{ta}/{sl}"].append(tr)
rows = {}
for k, lst in pool.items():
    t = pd.concat(lst, ignore_index=True); y = t.target.values; res = dict(n=len(t), obs=float(y.mean()), obs_days=float(t.days.mean()))
    for m in ("p0", "p1"):
        p = t[m].values; res[m + "_sum"] = float(p.mean()); res[m + "_brier"] = float(((y - p)**2).mean())
        res[m + "_z"] = float((y.sum() - p.sum())/np.sqrt((p*(1-p)).sum()))               # calibration z-score (sum of Bernoulli variances)
    d01 = (y - t.p0.values)**2 - (y - t.p1.values)**2
    res["dB_M0_M1"] = [float(d01.mean()), float(d01.std()/np.sqrt(len(t)))]; res["days_M1"] = float(t.e1.mean())
    rows[k] = res
R["out_of_sample"] = rows

# ------------------------------------------------------------------ figures (t2_theory_sim; the data figure is made by fig_merged.py)
nm = "TSLA"; r = series[nm]; f = fit(r)
u, l = np.log(1.15), -np.log(0.90)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for i in range(25):
    x = np.cumsum(f["mu"] + f["sig"]*rng.standard_normal(400)); k = np.argmax((x >= u) | (x <= -l)) if ((x >= u) | (x <= -l)).any() else 399
    ax[0].plot(np.arange(k+1), 100*(np.exp(x[:k+1])-1), lw=.8)
ax[0].axhline(15, color="g", ls="--"); ax[0].axhline(-10, color="r", ls="--"); ax[0].set_title("25 simulated TSLA trades (+15% / −10%)"); ax[0].set_xlabel("days"); ax[0].set_ylabel("return %")
o, T = simulate(f["mu"], f["sig"], u, l, 40000); tr = trades(r.values, u, l, WARM, len(r))
bins = np.linspace(0, 150, 31); ax[1].hist(T[o >= 0], bins, density=True, alpha=.5, label="model M1 (sim)"); ax[1].hist(tr.days, bins, density=True, alpha=.5, label=f"real trades (n={len(tr)})"); ax[1].legend(fontsize=7); ax[1].set_title("Days until exit, TSLA"); ax[1].set_xlabel("days")
tas = np.linspace(0.04, 0.30, 14); pc = [p_up_cont(f["mu"], f["sig"], np.log(1+t_), l) for t_ in tas]; pd_ = [p_up_disc(f["mu"], f["sig"], np.log(1+t_), l) for t_ in tas]
ps_ = []
for t_ in tas:
    o_, _ = simulate(f["mu"], f["sig"], np.log(1+t_), l, 15000); ps_.append((o_[o_ >= 0] == 1).mean())
ax[2].plot(100*tas, pc, "b--", label="formula (continuous)"); ax[2].plot(100*tas, pd_, "r-", label="formula (daily-close corrected)"); ax[2].plot(100*tas, ps_, "ko", ms=3, label="simulation")
ax[2].plot(100*tas, [l/(np.log(1+t_)+l) for t_ in tas], "g:", label="fair game l/(u+l)"); ax[2].legend(fontsize=6.5); ax[2].set_xlabel("take-profit % (stop = −10%)"); ax[2].set_title("P(target first) vs take-profit")
save("t2_theory_sim")

json.dump(R, open("results_trade.json", "w"), indent=1, default=float)
