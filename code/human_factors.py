"""
Human factors in the take-profit / stop-loss problem.  Extends trade_exit.py (same data, same trade definition).

H1  Disposition-style exits (model M3): the trader (a) sometimes overrides the stop-loss, (b) sometimes sells on an emotional impulse
    (a daily hazard, the discrete-time version of a Poisson clock, higher when in profit than in loss).
    Solved by simulation, then REPLAYED ON THE REAL PRICE HISTORIES of all six series.            [assumed behaviour, real prices]
H2  Investor fear: India VIX / CBOE VIX at entry vs real trade duration and outcome.             [real sentiment data]
H3  Overconfidence by extrapolation: a trader who believes last 60/250/500 days' drift will continue; calibration of the
    probabilities that belief implies, on real out-of-sample trades.                               [real prices]

Run: python code/human_factors.py  (needs data/basket_close.csv, data/sentiment.csv)  -> results_human.json, figures/h*.png
[EXTRA] items (outside the core syllabus): OLS/logit with robust errors, VIX covariate.  Behaviour parameters in H1 are ASSUMPTIONS (sensitivity analysis), not estimates.
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # shared figure style
import statsmodels.api as sm
from scipy import stats

rng = np.random.default_rng(2027)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 9})
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png"); plt.close()

px = pd.read_csv("data/basket_close.csv", index_col=0, parse_dates=True)
fear = pd.read_csv("data/sentiment.csv", index_col=0, parse_dates=True).ffill()
NAMES = {"^NSEI": "NIFTY 50", "RELIANCE.NS": "RELIANCE", "TCS.NS": "TCS", "HDFCBANK.NS": "HDFC Bank", "AAPL": "AAPL", "TSLA": "TSLA"}
FEAR_OF = {"NIFTY 50": "INDIAVIX", "RELIANCE": "INDIAVIX", "TCS": "INDIAVIX", "HDFC Bank": "INDIAVIX", "AAPL": "VIX", "TSLA": "VIX"}
HEAD = (0.15, 0.10); SHORT = (0.10, 0.05); RULES = [HEAD, SHORT]
W = 20; BGK = 0.5826
HARD = -np.log(0.75)                       # an overriding trader finally capitulates after a 25% price fall
RATIO = 1.5                                # impulse-sale hazard in profit / in loss (disposition effect; Odean 1998 reports roughly 1.5)
SCEN = {"S0 disciplined": (0.0, 0.0), "S1 mild": (0.25, 0.005), "S2 strong": (0.50, 0.010)}   # (override probability q, loss-side hazard c)
R = {"assumptions": dict(hard_price_fall=0.25, hazard_ratio=RATIO, scenarios=SCEN)}
series = {}
for tk, nm in NAMES.items():
    series[nm] = np.log(px[tk].dropna()).diff().dropna()

# ------------------------------------------------------------------ shared theory / fitting (same definitions as trade_exit.py)
def p_up_cont(mu, sig, u, l):
    if abs(mu) < 1e-10: return l/(u + l)
    k = 2*mu/sig**2
    a = np.exp(np.clip(k*l, -600, 600)); b = np.exp(np.clip(-k*u, -600, 600))
    return (1 - a)/(b - a)
def p_up_disc(mu, sig, u, l): return p_up_cont(mu, sig, u + BGK*sig, l + BGK*sig)
def et_cont(mu, sig, u, l):
    p = p_up_cont(mu, sig, u, l); return u*l/sig**2 if abs(mu) < 1e-10 else (u*p - l*(1-p))/mu
def et_disc(mu, sig, u, l): return et_cont(mu, sig, u + BGK*sig, l + BGK*sig)
def fit(r): return dict(mu=float(r.mean()), sig=float(r.std(ddof=0)))
def ul(rule): return np.log(1 + rule[0]), -np.log(1 - rule[1])
def trades(rv, u, l, lo, hi):
    cs = np.cumsum(rv); out = []; i = lo
    while i < hi - 1:
        seg = cs[i+1:hi] - cs[i]; hit = (seg >= u) | (seg <= -l)
        if not hit.any(): break
        k = int(np.argmax(hit)); out.append((i, k + 1, int(seg[k] >= u))); i = i + k + 1
    return pd.DataFrame(out, columns=["start", "days", "target"])

# ================================================================== H1  behavioural exits
def sim_behav(mu, sig, u, l, q, c, N, Hmax=6000):
    """Daily random walk with the human deviations. out: 1 target, 0 stop honoured, 2 impulse sale, 3 capitulation after override, -1 unfinished."""
    hg, hl = RATIO*c, c
    x = np.zeros(N); T = np.zeros(N, int); out = -np.ones(N, int)
    ov = np.zeros(N, bool); chk = np.zeros(N, bool); alive = np.arange(N)
    for t in range(1, Hmax + 1):
        if alive.size == 0: break
        x[alive] += mu + sig*rng.standard_normal(alive.size); xa = x[alive]
        done = xa >= u; out[alive[done]] = 1
        cand = (~done) & (xa <= -l) & (~chk[alive])                       # first touch of the planned stop: decide once
        if cand.any():
            pos = np.where(cand)[0]; idx = alive[pos]; ovr = rng.random(idx.size) < q
            chk[idx] = True; ov[idx[ovr]] = True
            out[idx[~ovr]] = 0; done[pos[~ovr]] = True
        hd = (~done) & ov[alive] & (xa <= -HARD); out[alive[hd]] = 3; done |= hd
        rem = np.where(~done)[0]
        if c > 0 and rem.size:
            em = rng.random(rem.size) < np.where(xa[rem] > 0, hg, hl); out[alive[rem[em]]] = 2; done[rem[em]] = True
        T[alive[done]] = t; alive = alive[~done]
    return out, T, x

def metrics(out, T, x, l, mu):
    fin = out >= 0; o, t, xx = out[fin], T[fin], x[fin]; price = np.exp(xx) - 1; n = len(o)
    growth = xx.sum()/t.sum(); resid = xx - mu*t
    se = float(np.sqrt(resid.var()/n)/t.mean())                            # delta-method s.e. of growth (ratio estimator)
    return dict(n=int(n), p_target=float((o == 1).mean()), p_stop=float((o == 0).mean()), p_impulse=float((o == 2).mean()), p_capit=float((o == 3).mean()),
                days=float(t.mean()), mean_ret=float(xx.mean()), growth=float(growth), growth_se=se, wald_z=float((growth - mu)/se), sd=float(xx.std()),
                p5=float(np.percentile(price, 5)), worst=float(price.min()), p_bigloss=float((price < -(1 - np.exp(-l)) - 0.02).mean()), unfinished=float(1 - fin.mean()))

H1 = {"sim": {}, "grid": {}, "replay": {}}
u, l = ul(HEAD); store = {}
for nm, r in series.items():
    f = fit(r); H1["sim"][nm] = {}
    for sn, (q, c) in SCEN.items():
        out, T, x = sim_behav(f["mu"], f["sig"], u, l, q, c, 20000)
        H1["sim"][nm][sn] = metrics(out, T, x, l, f["mu"])
        if nm in ("TSLA", "NIFTY 50"): store[(nm, sn)] = np.exp(x[out >= 0]) - 1
# sensitivity grid (pooled average over the six series)
QS = [0, .25, .5, .75]; CS = [0, .005, .01, .02]
for q in QS:
    for c in CS:
        rows = []
        for nm, r in series.items():
            f = fit(r); out, T, x = sim_behav(f["mu"], f["sig"], u, l, q, c, 6000); m = metrics(out, T, x, l, f["mu"]); m["mu"] = f["mu"]; rows.append(m)
        H1["grid"][f"{q}/{c}"] = dict(p_target=float(np.mean([m["p_target"] for m in rows])), days=float(np.mean([m["days"] for m in rows])),
            p5=float(np.mean([m["p5"] for m in rows])), growth_ratio=float(np.mean([m["growth"]/m["mu"] for m in rows])),
            p_bigloss=float(np.mean([m["p_bigloss"] for m in rows])))

def replay(rv, q, c, rg):
    """Behavioural trader replayed on a real return path: sequential trades, same entry rule as trades(). Returns list of (days, x, outcome)."""
    hg, hl = RATIO*c, c; n = len(rv); i = 0; res = []
    while i < n - 1:
        x = 0.0; ov = False; chk = False; j = i
        while True:
            j += 1
            if j >= n: return res                                           # unfinished final trade dropped
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

REPS = 200
for nm, r in series.items():
    rv = r.values; H1["replay"][nm] = {}
    for sn, (q, c) in SCEN.items():
        reps = 1 if (q == 0 and c == 0) else REPS; rows = []
        for k in range(reps): rows += replay(rv, q, c, np.random.default_rng(10_000 + k))
        a = np.array(rows, float); d, xx, o = a[:, 0], a[:, 1], a[:, 2]; price = np.exp(xx) - 1
        H1["replay"][nm][sn] = dict(n_per_rep=float(len(a)/reps), p_target=float((o == 1).mean()), p_stop=float((o == 0).mean()), p_impulse=float((o == 2).mean()),
            p_capit=float((o == 3).mean()), days=float(d.mean()), growth=float(xx.sum()/d.sum()), p5=float(np.percentile(price, 5)), worst=float(price.min()),
            p_bigloss=float((price < -(1 - np.exp(-l)) - 0.02).mean()), mu=float(rv.mean()))
R["H1"] = H1

# ================================================================== common: entry covariates
def entry_frame(nm, rule, lo=W):
    r = series[nm]; rv = r.values; uu, ll = ul(rule); tr = trades(rv, uu, ll, lo, len(rv)); idx = r.index
    tr["stock"] = nm; tr["date"] = idx[tr.start.values]
    tr["logvol"] = np.log(r.rolling(W).std().values[tr.start.values])
    fr = fear[FEAR_OF[nm]].reindex(idx, method="ffill").values; tr["logfear"] = np.log(fr[tr.start.values])
    return tr
def zs(df, col): return df.groupby("stock")[col].transform(lambda s: (s - s.mean())/s.std())
def fe(df): return pd.get_dummies(df.stock, drop_first=True).astype(float)
def reg(y, cols, df, hac=None):
    X = pd.concat([df[cols], fe(df)], axis=1); X = sm.add_constant(X)
    m = sm.OLS(df[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac}) if hac else sm.OLS(df[y], X).fit(cov_type="HC1")
    return m
def pack(m, cols): return dict(r2=float(m.rsquared), n=int(m.nobs), coef={c: dict(b=float(m.params[c]), se=float(m.bse[c]), p=float(m.pvalues[c])) for c in cols})

# ================================================================== H2  investor fear
H2 = {}
for rule in RULES:
    df = pd.concat([entry_frame(nm, rule) for nm in series], ignore_index=True)
    df["zvol"] = zs(df, "logvol"); df["zfear"] = zs(df, "logfear"); df["ldays"] = np.log(df.days)
    ma = reg("ldays", ["zvol"], df); mb = reg("ldays", ["zfear"], df); mc = reg("ldays", ["zvol", "zfear"], df)
    lg = sm.Logit(df.target, sm.add_constant(pd.concat([df[["zvol", "zfear"]], fe(df)], axis=1))).fit(disp=0, cov_type="HC1")
    df["fear_tercile"] = df.groupby("stock")["zfear"].transform(lambda s: pd.qcut(s, 3, labels=False))
    ter = {int(k): dict(n=int(len(g)), days=float(g.days.mean()), days_se=float(g.days.std()/np.sqrt(len(g))), win=float(g.target.mean())) for k, g in df.groupby("fear_tercile")}
    m0 = sm.OLS(df.ldays, sm.add_constant(fe(df))).fit()                                  # stock fixed effects only
    H2[f"{rule[0]}/{rule[1]}"] = dict(n=int(len(df)), vol_only=pack(ma, ["zvol"]), fear_only=pack(mb, ["zfear"]), both=pack(mc, ["zvol", "zfear"]),
        r2_fe_only=float(m0.rsquared), dR2_fear_given_vol=float(mc.rsquared - ma.rsquared), dR2_vol_over_fe=float(ma.rsquared - m0.rsquared), dR2_fear_over_fe=float(mb.rsquared - m0.rsquared), dR2_vol_given_fear=float(mc.rsquared - mb.rsquared),
        logit_fear=dict(b=float(lg.params["zfear"]), se=float(lg.bse["zfear"]), p=float(lg.pvalues["zfear"])), terciles=ter)
    if rule == HEAD: head_df = df
# daily relation between trailing volatility and fear
cors = {}
for nm, r in series.items():
    v = np.log(r.rolling(W).std()).dropna(); fr = np.log(fear[FEAR_OF[nm]].reindex(v.index, method="ffill")); cors[nm] = float(v.corr(fr))
H2["corr_vol_fear"] = cors
R["H2"] = H2

# ================================================================== H3  overconfidence by extrapolation
H3 = {}
def logit_(p): p = np.clip(p, .02, .98); return np.log(p/(1 - p))
for rule in RULES:
    uu, ll = ul(rule); rows = []
    for nm, r in series.items():
        h = len(r)//2; rv = r.values; tr = trades(rv, uu, ll, h, len(rv)); tr["stock"] = nm; f = fit(r.iloc[:h])
        tr["p_static"] = p_up_disc(f["mu"], f["sig"], uu, ll); tr["p_fair"] = ll/(uu + ll)
        for w in (60, 250, 500):
            mw = r.rolling(w).mean().values[tr.start.values]; sw = r.rolling(w).std(ddof=0).values[tr.start.values]
            tr[f"p_{w}"] = [p_up_disc(a, b, uu, ll) for a, b in zip(mw, sw)]
        rows.append(tr)
    t = pd.concat(rows, ignore_index=True); y = t.target.values; res = dict(n=int(len(t)), obs=float(y.mean()), models={})
    for k, lab in (("p_fair", "fair game"), ("p_static", "static (first-half fit)"), ("p_500", "belief: last 500 days"), ("p_250", "belief: last 250 days"), ("p_60", "belief: last 60 days")):
        p = t[k].values; z = (y.sum() - p.sum())/np.sqrt((p*(1 - p)).sum())
        try:
            lg = sm.Logit(y, sm.add_constant(logit_(p))).fit(disp=0); sl, sl_se = float(lg.params[1]), float(lg.bse[1])
        except Exception: sl, sl_se = float("nan"), float("nan")
        terc = pd.qcut(p, 3, labels=False, duplicates="drop")
        cal = [dict(pred=float(p[terc == g].mean()), obs=float(y[terc == g].mean()), n=int((terc == g).sum())) for g in np.unique(terc)]
        d_ = (y - p)**2 - (y - t.p_static.values)**2
        res["models"][k] = dict(label=lab, mean_p=float(p.mean()), z=float(z), brier=float(((y - p)**2).mean()), sd_p=float(p.std()), slope=sl, slope_se=sl_se, cal=cal,
                                dB_vs_static=[float(d_.mean()), float(d_.std()/np.sqrt(len(t)))])
    # CONTROL: same belief estimators on simulated iid Gaussian returns (true drift = full-sample mean) -> pure estimation-noise penalty
    crng = np.random.default_rng(5150); ctl = {k: [] for k in ("p_60", "p_250", "p_500")}; ctl_sd = {k: [] for k in ctl}
    for rep_ in range(20):
        rows_ = []
        for nm, r in series.items():
            rs = pd.Series(crng.normal(r.mean(), r.std(ddof=0), len(r)), index=r.index); h = len(rs)//2; rvs = rs.values
            tr_ = trades(rvs, uu, ll, h, len(rvs)); f_ = fit(rs.iloc[:h]); tr_["p_static"] = p_up_disc(f_["mu"], f_["sig"], uu, ll)
            for w in (60, 250, 500):
                mw = rs.rolling(w).mean().values[tr_.start.values]; sw = rs.rolling(w).std(ddof=0).values[tr_.start.values]
                tr_[f"p_{w}"] = [p_up_disc(a, b, uu, ll) for a, b in zip(mw, sw)]
            rows_.append(tr_)
        t_ = pd.concat(rows_, ignore_index=True); y_ = t_.target.values
        for k in ctl: ctl[k].append(float(((y_ - t_[k].values)**2).mean() - ((y_ - t_.p_static.values)**2).mean())); ctl_sd[k].append(float(t_[k].values.std()))
    res["iid_control"] = {k: dict(dBrier_vs_static=float(np.mean(v)), dBrier_sd=float(np.std(v)), sd_p=float(np.mean(ctl_sd[k]))) for k, v in ctl.items()}
    H3[f"{rule[0]}/{rule[1]}"] = res
R["H3"] = H3

json.dump(R, open("results_human.json", "w"), indent=1, default=float)

# ================================================================== figures
cols = {"S0 disciplined": "#1f6f6f", "S1 mild": "#d99a1e", "S2 strong": "#c0412c"}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for k, nm in enumerate(("NIFTY 50", "TSLA")):
    bot = np.zeros(3)
    for key, col, lab in (("p_target", "#2e8b57", "target"), ("p_stop", "#c0412c", "stop honoured"), ("p_capit", "#7a1f12", "capitulation after override"), ("p_impulse", "#d99a1e", "impulse sale")):
        v = np.array([H1["sim"][nm][s][key] for s in SCEN]); ax[0].bar(np.arange(3) + k*3.6, v, bottom=bot, color=col, label=lab if k == 0 else None); bot += v
ax[0].set_xticks(list(range(3)) + [3.6 + i for i in range(3)]); ax[0].set_xticklabels(["disc.", "mild", "strong"]*2, fontsize=7); ax[0].set_title("How trades end (left NIFTY, right TSLA)"); ax[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2)
bins = np.linspace(-0.6, 0.3, 46)
for sn in ("S0 disciplined", "S2 strong"): ax[1].hist(store[("TSLA", sn)], bins, alpha=.55, density=True, color=cols[sn], label=sn)
ax[1].axvline(-.10, color="k", ls=":"); ax[1].set_title("TSLA trade returns: planned stop at −10%"); ax[1].set_xlabel("return per trade"); ax[1].legend(fontsize=7)
nmz = list(series); w_ = .25
for i, sn in enumerate(SCEN): ax[2].bar(np.arange(6) + (i - 1)*w_, [H1["sim"][n][sn]["growth"]/series[n].mean() for n in nmz], w_, color=cols[sn], label=sn)
ax[2].axhline(1, color="k", lw=.8); ax[2].set_xticks(range(6)); ax[2].set_xticklabels(nmz, rotation=30, fontsize=7); ax[2].set_title("Growth per day ÷ drift μ (Wald: = 1)"); ax[2].legend(fontsize=6)
save("h1_behaviour")

hd = H2[f"{HEAD[0]}/{HEAD[1]}"]; sh = H2[f"{SHORT[0]}/{SHORT[1]}"]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for rl, lab, off in ((hd, "+15/−10", -.2), (sh, "+10/−5", .2)):
    ax[0].bar(np.arange(3) + off, [rl["terciles"][str(g) if str(g) in rl["terciles"] else g]["days"] if (str(g) in rl["terciles"] or g in rl["terciles"]) else np.nan for g in range(3)], .4, label=lab)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["low fear", "mid", "high fear"]); ax[0].set_title("Mean days to exit by fear tercile at entry"); ax[0].legend(fontsize=7)
ax[1].scatter(head_df.zfear, head_df.ldays, s=6, alpha=.45); bb = np.polyfit(head_df.zfear, head_df.ldays, 1); xs = np.linspace(head_df.zfear.min(), head_df.zfear.max(), 10); ax[1].plot(xs, np.polyval(bb, xs), "r"); ax[1].set_xlabel("fear index at entry (within-stock z)"); ax[1].set_ylabel("log days"); ax[1].set_title("Duration vs fear, +15%/−10% trades")
for i, (rl, lab) in enumerate(((hd, "+15/−10"), (sh, "+10/−5"))):
    for j, (m, key) in enumerate((("vol_only", "zvol"), ("fear_only", "zfear"))):
        ax[2].bar(i*4 + j, rl[m]["r2"], color=["#1f6f6f", "#d99a1e"][j], label=("trailing vol only" if j == 0 else "fear index only") if i == 0 else None)
    ax[2].bar(i*4 + 2, rl["both"]["r2"], color="#555", label="both" if i == 0 else None)
ax[2].set_xticks([1, 5]); ax[2].set_xticklabels(["+15/−10", "+10/−5"]); ax[2].set_title("R² of log duration (stock effects included)"); ax[2].legend(fontsize=7)
save("h2_fear")

h3 = H3[f"{HEAD[0]}/{HEAD[1]}"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for k, mk in (("p_static", "k-"), ("p_500", "b-"), ("p_250", "g-"), ("p_60", "r-")):
    c_ = h3["models"][k]["cal"]; ax[0].plot([x["pred"] for x in c_], [x["obs"] for x in c_], mk + "o", label=h3["models"][k]["label"], ms=4)
ax[0].plot([.3, .8], [.3, .8], "--", color="gray"); ax[0].set_xlabel("predicted P(target)"); ax[0].set_ylabel("observed"); ax[0].set_title("Reliability by tercile, +15%/−10% (OOS)"); ax[0].legend(fontsize=6.5)
ks = ["p_static", "p_500", "p_250", "p_60"]
ax[1].bar(range(4), [h3["models"][k]["slope"] for k in ks], yerr=[1.96*h3["models"][k]["slope_se"] for k in ks], color=["#555", "#2f5fa5", "#2e8b57", "#c0412c"], capsize=3)
ax[1].axhline(1, color="k", ls=":"); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["static", "500d", "250d", "60d"]); ax[1].set_title("Calibration slope (1 = ideal)")
save("h3_belief")

print("done")
