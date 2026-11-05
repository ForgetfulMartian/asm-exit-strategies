"""
Robustness of the core model and of the behavioural exit model on the WIDER universe (data/wide_close.csv, 36 series: the original six plus Indian and US
large/mid caps incl. collapsed names, international indices, gold and bond ETFs).
  A  In-sample: exit-time KS test and success probability per series (+15%/-10%)
  B  Out-of-sample: fit on the first half, trade the second half; pooled calibration z, Brier score, per-series scatter
  C  Behaviour (model M3 replayed on real paths): disciplined vs strong behaviour -> P(target), 5% quantile of trade return, Wald identity
Run: python code/robust_model.py -> results_wide_model.json, figures/w1_wide_model.png        Seed 2028.
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # shared figure style
from scipy import stats

rng = np.random.default_rng(2028)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 9})
px = pd.read_csv("data/wide_close.csv", index_col=0, parse_dates=True)
series = {c: np.log(px[c].dropna()).diff().dropna() for c in px.columns}
BGK = 0.5826; HARD = -np.log(.75); RATIO = 1.5
def p_cont(mu, sig, u, l):
    if abs(mu) < 1e-10: return l/(u + l)
    k = 2*mu/sig**2; a = np.exp(np.clip(k*l, -600, 600)); b = np.exp(np.clip(-k*u, -600, 600)); return (1 - a)/(b - a)
def p_disc(mu, sig, u, l): return p_cont(mu, sig, u + BGK*sig, l + BGK*sig)
def et_cont(mu, sig, u, l):
    p = p_cont(mu, sig, u, l); return u*l/sig**2 if abs(mu) < 1e-10 else (u*p - l*(1-p))/mu
def et_disc(mu, sig, u, l): return et_cont(mu, sig, u + BGK*sig, l + BGK*sig)
def trades(rv, u, l, lo, hi):
    cs = np.cumsum(rv); out = []; i = lo
    while i < hi - 1:
        seg = cs[i+1:hi] - cs[i]; hit = (seg >= u) | (seg <= -l)
        if not hit.any(): break
        k = int(np.argmax(hit)); out.append((i, k + 1, int(seg[k] >= u))); i = i + k + 1
    return pd.DataFrame(out, columns=["start", "days", "target"])
def sim_days(mu, sig, u, l, N=20000, Hmax=6000):
    x = np.zeros(N); T = np.zeros(N, int); alive = np.arange(N)
    for t in range(1, Hmax + 1):
        if alive.size == 0: break
        x[alive] += mu + sig*rng.standard_normal(alive.size); xa = x[alive]; done = (xa >= u) | (xa <= -l); T[alive[done]] = t; alive = alive[~done]
    return T[T > 0]
RULES = {"0.15/0.1": (.15, .10), "0.1/0.05": (.10, .05)}
R = {"series": list(series), "A": {}, "B": {}, "C": {}}
for rk, (a, b) in RULES.items():
    u, l = np.log(1 + a), -np.log(1 - b); A = {}; rows = []
    for nm, r in series.items():
        rv = r.values; n = len(rv); mu, sig = rv.mean(), rv.std()
        tr = trades(rv, u, l, 0, n)                                                # in-sample
        if len(tr) >= 8:
            ks = stats.ks_2samp(tr.days.values, sim_days(mu, sig, u + 0, l + 0))   # simulated daily exit-time distribution under M1
            A[nm] = dict(n=len(tr), obs=float(tr.target.mean()), p1=float(p_disc(mu, sig, u, l)), p0=float(l/(u + l)), days=float(tr.days.mean()), et1=float(et_disc(mu, sig, u, l)), ks_p=float(ks.pvalue))
        h = n//2; m1, s1 = rv[:h].mean(), rv[:h].std(); tr2 = trades(rv, u, l, h, n)       # out-of-sample
        for _, t in tr2.iterrows(): rows.append((nm, int(t.target), p_disc(m1, s1, u, l), l/(u + l), t.days, et_disc(m1, s1, u, l)))
    O = pd.DataFrame(rows, columns=["s", "y", "p1", "p0", "days", "et1"])
    zf = lambda y, p: float((y.sum() - p.sum())/np.sqrt((p*(1-p)).sum()))
    ps = {s: dict(n=int(len(g)), obs=float(g.y.mean()), p1=float(g.p1.mean()), p0=float(g.p0.mean())) for s, g in O.groupby("s")}
    dB = (O.y - O.p0)**2 - (O.y - O.p1)**2
    R["B"][rk] = dict(n=int(len(O)), obs=float(O.y.mean()), p0=float(O.p0.mean()), p1=float(O.p1.mean()), z0=zf(O.y, O.p0), z1=zf(O.y, O.p1),
                      brier0=float(((O.y - O.p0)**2).mean()), brier1=float(((O.y - O.p1)**2).mean()), dB=[float(dB.mean()), float(dB.std()/np.sqrt(len(dB)))],
                      days=float(O.days.mean()), et1=float(O.et1.mean()), per_series=ps)
    ksp = np.array([v["ks_p"] for v in A.values()])
    R["A"][rk] = dict(per_series=A, n_series=len(A), ks_reject_5=int((ksp < .05).sum()), ks_median_p=float(np.median(ksp)),
                      mean_abs_p_err=float(np.mean([abs(v["obs"] - v["p1"]) for v in A.values()])), mean_abs_p_err_fair=float(np.mean([abs(v["obs"] - v["p0"]) for v in A.values()])),
                      dur_ratio_median=float(np.median([v["days"]/v["et1"] for v in A.values()])))
# C: behaviour replayed on real paths (S0 disciplined vs S2 strong: q=.5, c=.01) on the headline rule
u, l = np.log(1.15), -np.log(.90)
def replay(rv, q, c, rg):
    hg, hl = RATIO*c, c; n = len(rv); i = 0; res = []
    while i < n - 1:
        x = 0.0; ov = False; chk = False; j = i
        while True:
            j += 1
            if j >= n: return res
            x += rv[j]
            if x >= u: o = 1; break
            if (not chk) and x <= -l:
                chk = True
                if rg.random() < q: ov = True
                else: o = 0; break
            if ov and x <= -HARD: o = 3; break
            if c > 0 and rg.random() < (hg if x > 0 else hl): o = 2; break
        res.append((j - i, x, o)); i = j
    return res
C = {}
for nm, r in series.items():
    rv = r.values; d0 = np.array(replay(rv, 0, 0, np.random.default_rng(1)), float); rows = []
    for k in range(30): rows += replay(rv, .5, .01, np.random.default_rng(100 + k))
    d2 = np.array(rows, float)
    gs = lambda d: d[:, 1].sum()/d[:, 0].sum()
    C[nm] = dict(p_target0=float((d0[:, 2] == 1).mean()), p_target2=float((d2[:, 2] == 1).mean()), p5_0=float(np.percentile(np.exp(d0[:, 1]) - 1, 5)), p5_2=float(np.percentile(np.exp(d2[:, 1]) - 1, 5)),
                 worst0=float((np.exp(d0[:, 1]) - 1).min()), worst2=float((np.exp(d2[:, 1]) - 1).min()), growth0=float(gs(d0)), growth2=float(gs(d2)), mu=float(rv.mean()))
R["C"] = dict(per_series=C, p5_0=float(np.mean([v["p5_0"] for v in C.values()])), p5_2=float(np.mean([v["p5_2"] for v in C.values()])),
              worse_tail=int(sum(v["p5_2"] < v["p5_0"] for v in C.values())), n=len(C),
              pt0=float(np.mean([v["p_target0"] for v in C.values()])), pt2=float(np.mean([v["p_target2"] for v in C.values()])),
              growth_gap=float(np.mean([abs(v["growth2"] - v["growth0"])/abs(v["mu"]) for v in C.values()])))
json.dump(R, open("results_wide_model.json", "w"), indent=1)
# figure
Bk = R["B"]["0.15/0.1"]["per_series"]; fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, key, ttl in ((ax[0], "p1", "M1 Brownian"), (ax[1], "p0", "M0 fair game")):
    xs = [v[key] for v in Bk.values()]; ys = [v["obs"] for v in Bk.values()]; sz = [4 + v["n"]/1.5 for v in Bk.values()]
    a.scatter(xs, ys, s=sz, alpha=.7); a.plot([.2, .85], [.2, .85], "k--", lw=.8); a.set_xlabel(f"predicted P(target), {ttl}"); a.set_ylabel("observed (out of sample)"); a.set_title(ttl, fontsize=9); a.set_xlim(.2, .85); a.set_ylim(0, 1)
x = np.arange(len(C)); order = sorted(C, key=lambda k: C[k]["p5_2"]); ax[2].plot([C[k]["p5_0"]*100 for k in order], "o-", ms=3, label="disciplined"); ax[2].plot([C[k]["p5_2"]*100 for k in order], "o-", ms=3, label="strong behaviour")
ax[2].set_xlabel("series (sorted)"); ax[2].set_ylabel("5% worst trade return (%)"); ax[2].set_title("5% worst trade per series (sorted by strong-behaviour value)", fontsize=9); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/w1_wide_model.png"); plt.close()
for rk in RULES: print(rk, {k: v for k, v in R["B"][rk].items() if k != "per_series"}, {k: v for k, v in R["A"][rk].items() if k != "per_series"})
print({k: v for k, v in R["C"].items() if k != "per_series"})
