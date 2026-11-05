"""
Strategy tournament: can any kind of human behaviour be turned into a valid trading strategy?   (extends human_factors.py)

Eleven "trader archetypes" are fixed IN ADVANCE (no tuning), each = an entry habit + an exit habit, all with the same take-profit / stop-loss
backbone (+15% / -10%) unless stated:
  S0 Disciplined        always invested, honours the stop                         (baseline; Wald's identity says no edge)
  S1 Stubborn           overrides the stop 75% of the time, capitulates at -25%
  S2 Nervous            sells on impulse (daily hazard 1.0% in loss, 1.5% in profit)
  S3 Disposition mix    override 50% + impulse 1.0%/1.5%
  S4 Quick-profit       +5% target, -25% stop ("cut winners, let losers run")
  S5 Panic seller       exits when fear (VIX) is in the top 15% of its last year, re-enters only when below the 70th percentile
  S6 Fear contrarian    enters only when fear is in the top third of its last year ("be greedy when others are fearful")
  S7 Complacent         enters only when fear is in the bottom third
  S8 Momentum chaser    enters only when the last-60-day return is in the top third of its last 500 days
  S9 Dip buyer          enters only when the last-20-day return is in the bottom third of its last 500 days
  S10 Low-volatility entry  enters only when 20-day volatility is below its trailing-500-day median

Evidence: (1) real prices of the six series; (2) a NULL distribution made by ROLLING the return series by a random offset (>= 250 days) against the
entry signals, i.e. same drift, same volatility clustering, same signals, but no link between signal and the future  -> Monte-Carlo p-value;
(3) Holm correction over the 11 strategies; (4) first-half / second-half consistency, cost sensitivity (0.1% per side), per-series consistency;
(5) simulated worlds (iid, trending, mean-reverting) to show the test has power when an edge really exists (positive control).

Run: python code/strategy_tournament.py  -> results_tournament.json, figures/s1..s3.png     [EXTRA] circular-shift null, Holm correction.
"""
import json, os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # shared figure style
from joblib import Parallel, delayed

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 9})
WIDE = os.environ.get("ASM_WIDE") == "1"; OUT = "_wide" if WIDE else ""     # ASM_WIDE=1 -> run on the 36-series universe (data/wide_close.csv)
px = pd.read_csv("data/wide_close.csv" if WIDE else "data/basket_close.csv", index_col=0, parse_dates=True)
fear = pd.read_csv("data/sentiment.csv", index_col=0, parse_dates=True).ffill()
NAMES = {"^NSEI": "NIFTY 50", "RELIANCE.NS": "RELIANCE", "TCS.NS": "TCS", "HDFCBANK.NS": "HDFC Bank", "AAPL": "AAPL", "TSLA": "TSLA"}
FEAR_OF = {"NIFTY 50": "INDIAVIX", "RELIANCE": "INDIAVIX", "TCS": "INDIAVIX", "HDFC Bank": "INDIAVIX", "AAPL": "VIX", "TSLA": "VIX"}
if WIDE:
    NAMES = {c: c for c in px.columns}
    FEAR_OF = {c: ("INDIAVIX" if (c.endswith(".NS") or c == "^NSEI") else "VIX") for c in px.columns}
series = {nm: np.log(px[tk].dropna()).diff().dropna() for tk, nm in NAMES.items()}
COST = 0.001                         # per side
HARD = -np.log(0.75); RATIO = 1.5
U15, L10 = np.log(1.15), -np.log(0.90)
U5, L25 = np.log(1.05), -np.log(0.75)

# strategy table: key -> (label, u, l, q, c, kind)
STRATS = {
    "S0":  ("Disciplined",       U15, L10, 0.0,  0.0,  "all"),
    "S1":  ("Stubborn",          U15, L10, 0.75, 0.0,  "all"),
    "S2":  ("Nervous",           U15, L10, 0.0,  0.01, "all"),
    "S3":  ("Disposition mix",   U15, L10, 0.5,  0.01, "all"),
    "S4":  ("Quick-profit",      U5,  L25, 0.0,  0.0,  "all"),
    "S5":  ("Panic seller",      U15, L10, 0.0,  0.0,  "panic"),
    "S6":  ("Fear contrarian",   U15, L10, 0.0,  0.0,  "fear_hi"),
    "S7":  ("Complacent",        U15, L10, 0.0,  0.0,  "fear_lo"),
    "S8":  ("Momentum chaser",   U15, L10, 0.0,  0.0,  "mom"),
    "S9":  ("Dip buyer",         U15, L10, 0.0,  0.0,  "dip"),
    "S10": ("Low-volatility entry",       U15, L10, 0.0,  0.0,  "calm"),
}
RANDOM = {k for k, v in STRATS.items() if v[3] > 0 or v[4] > 0}

# ------------------------------------------------------------------ signals (strictly trailing: threshold uses data up to the day BEFORE)
def trail_q(s, q, win, mp): return s.rolling(win, min_periods=mp).quantile(q).shift(1)
def price_gates(r):
    v20 = r.rolling(20).std(); s60 = r.rolling(60).sum(); s20 = r.rolling(20).sum()
    med = v20.rolling(500, min_periods=250).median().shift(1)
    t60 = trail_q(s60, 2/3, 500, 250); t20 = trail_q(s20, 1/3, 500, 250)
    g = dict(calm=(v20 < med), mom=(s60 > t60), dip=(s20 < t20))
    valid = med.notna() & t60.notna() & t20.notna()
    return {k: v.fillna(False).values for k, v in g.items()}, valid.values
def fear_gates(f):
    q = {p: trail_q(f, p, 250, 150) for p in (.33, .67, .70, .85)}
    valid = q[.85].notna()
    return dict(fear_hi=(f > q[.67]).values, fear_lo=(f < q[.33]).values, panic=(f < q[.70]).values, panic_flag=(f > q[.85]).values), valid.values
def build(nm):
    r = series[nm]; n = len(r); out = dict(all=np.ones(n, bool)); gp, vp = price_gates(r)
    out.update(gp); valid = {"price": vp}
    f = fear[FEAR_OF[nm]].reindex(r.index, method="ffill"); gf, vf = fear_gates(f)
    out.update({k: np.nan_to_num(v.astype(float)).astype(bool) for k, v in gf.items()}); valid["fear"] = vf
    return r, out, valid
def window_start(kind, valid):
    base = np.argmax(valid["price"]);
    if kind in ("fear_hi", "fear_lo", "panic"): return max(base, int(np.argmax(valid["fear"])))
    return base

# ------------------------------------------------------------------ trade engine (day loop; entry at close of day i, one position at a time)
def engine(rv, gate, pflag, u, l, q, c, rng):
    n = len(rv); U = rng.random(n + 1) if (q > 0 or c > 0) else None; V = rng.random(n + 1) if q > 0 else None
    hg, hl = RATIO*c, c; inpos = np.zeros(n, bool); dr = np.zeros(n); ex = np.zeros(n); tr = []
    i = 0
    while i < n - 1:
        if not gate[i]: i += 1; continue
        x = 0.0; ov = False; chk = False; j = i; o = -1
        while True:
            j += 1
            if j >= n: j = n - 1; o = -1; break                 # still open at the end of data: marked to market, not counted as a trade
            x += rv[j]
            if x >= u: o = 1; break
            if pflag is not None and pflag[j]: o = 4; break
            if (not chk) and x <= -l:
                chk = True
                if q > 0 and V[j] < q: ov = True
                else: o = 0; break
            if ov and x <= -HARD: o = 3; break
            if c > 0 and U[j] < (hg if x > 0 else hl): o = 2; break
        inpos[i+1:j+1] = True; dr[i+1:j+1] = rv[i+1:j+1]
        if o >= 0: ex[j] = 2*COST; tr.append((j - i, x, o))
        if o < 0: break
        i = j
    return inpos, dr, ex, tr

def stats(rvw, gate, pflag, cfg, rng, detail=False):
    _, u, l, q, c, kind = cfg
    inpos, dr, ex, tr = engine(rvw, gate, pflag if kind == "panic" else None, u, l, q, c, rng)
    ni = max(int(inpos.sum()), 1); mu = rvw.mean(); drn = dr - ex
    sd = lambda a: a.std() if a.std() > 0 else np.nan
    sh = lambda a: a.mean()/sd(a)*np.sqrt(252) if inpos.any() else 0.0
    bh = rvw.mean()/rvw.std()*np.sqrt(252)
    o = dict(edge_g=(dr.sum()/ni - mu)*252*100, edge_n=(drn.sum()/ni - mu)*252*100, dsh_g=sh(dr) - bh, dsh_n=sh(drn) - bh, expo=inpos.mean(), ntr=len(tr))
    if detail:
        m = len(rvw)//2
        for nm_, sl in (("h1", slice(0, m)), ("h2", slice(m, None))):
            k = max(int(inpos[sl].sum()), 1); o["edge_" + nm_] = (dr[sl].sum()/k - rvw[sl].mean())*252*100
        xs = np.array([t[1] for t in tr]) if tr else np.array([0.0]); pr = np.exp(xs) - 1; ds = np.array([t[0] for t in tr]) if tr else np.array([1])
        eq = np.cumsum(drn); dd = float((eq - np.maximum.accumulate(eq)).min())
        o.update(win=float((np.array([t[2] for t in tr]) == 1).mean()) if tr else 0.0, p5=float(np.percentile(pr, 5)), cvar5=float(pr[pr <= np.percentile(pr, 5)].mean()),
                 days=float(ds.mean()), maxdd=dd, bh_maxdd=float((np.cumsum(rvw) - np.maximum.accumulate(np.cumsum(rvw))).min()), bh_sharpe=float(bh), mu_ann=float(mu*252*100))
    return o

# ------------------------------------------------------------------ one pass over all strategies and series
DATA = {nm: build(nm) for nm in series}
def one_pass(roll_u, seed, detail=False, reps=1):
    """roll_u in [0,1) or None.  Returns {strategy: {series: stats}} (stats averaged over `reps` behavioural draws)."""
    res = {}
    for k, cfg in STRATS.items():
        kind = cfg[5]; res[k] = {}
        for nm, (r, g, valid) in DATA.items():
            ws = window_start(kind, valid); rv = r.values[ws:]; n = len(rv)
            gate = g[kind if kind not in ("panic",) else "panic"][ws:]; gate = g["all"][ws:] if kind == "all" else gate
            pf = g["panic_flag"][ws:] if kind == "panic" else None
            if roll_u is not None: rv = np.roll(rv, 250 + int(roll_u*(n - 500)))
            rr = reps if k in RANDOM else 1; acc = None
            for b in range(rr):
                s = stats(rv, gate, pf, cfg, np.random.default_rng(seed*1000 + b), detail)
                acc = s if acc is None else {kk: acc[kk] + s[kk] for kk in s}
            res[k][nm] = {kk: vv/rr for kk, vv in acc.items()}; res[k][nm]["start"] = str(r.index[ws].date()); res[k][nm]["n"] = n
    return res
def pooled(res, key): return {k: float(np.mean([res[k][nm][key] for nm in res[k]])) for k in res}

# ------------------------------------------------------------------ simulated worlds (positive control)
def sim_world(world, mu, sig, n, rng):
    e = rng.standard_normal(n)
    if world == "iid": return mu + sig*e
    if world == "trending":                              # slowly varying hidden drift (persistence .98, sd ~ 25% a year)
        m = np.zeros(n); sd_m = 0.0010; inn = sd_m*np.sqrt(1 - .98**2)
        for t in range(1, n): m[t] = .98*m[t-1] + inn*rng.standard_normal()
        return mu + m + sig*e
    z = np.zeros(n); inn = 0.12*np.sqrt(1 - .98**2)      # reverting: price = trend + slowly decaying deviation (sd 12%)
    for t in range(1, n): z[t] = .98*z[t-1] + inn*rng.standard_normal()
    return mu + np.diff(z, prepend=0.0) + sig*e
SIMK = ["S0", "S4", "S8", "S9", "S10"]
def sim_one(world, seed):
    rng = np.random.default_rng(seed); out = {k: [] for k in SIMK}
    for nm, r in series.items():
        rv = sim_world(world, r.mean(), r.std(), 3500, rng); s = pd.Series(rv); gp, vp = price_gates(s); ws = int(np.argmax(vp)); rw = rv[ws:]
        for k in SIMK:
            cfg = STRATS[k]; gate = np.ones(len(rw), bool) if cfg[5] == "all" else gp[cfg[5]][ws:]
            st = stats(rw, gate, None, cfg, rng); out[k].append((st["edge_g"], st["dsh_g"]))
    return {k: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))) for k, v in out.items()}

def holm(p):
    idx = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r_, i in enumerate(idx): run = max(run, (m - r_)*p[i]); adj[i] = min(1.0, run)
    return adj

if __name__ == "__main__":
    NULLR, OBSREP, SIMW = 1000, 40, 150
    obs = one_pass(None, 1, detail=True, reps=OBSREP)
    nulls = Parallel(n_jobs=14)(delayed(one_pass)(float(np.random.default_rng(777 + b).random()), 50 + b) for b in range(NULLR))
    keys = list(STRATS); R = {"meta": dict(null_reps=NULLR, obs_reps=OBSREP, cost_per_side=COST, sim_worlds=SIMW), "strategies": {}}
    pe, ps = [], []
    for k in keys:
        o = {m: float(np.mean([obs[k][nm][m] for nm in obs[k]])) for m in ("edge_g", "edge_n", "dsh_g", "dsh_n", "expo", "ntr", "edge_h1", "edge_h2", "win", "p5", "cvar5", "days", "maxdd", "bh_maxdd")}
        ng = {m: np.array([np.mean([nl[k][nm][m] for nm in obs[k]]) for nl in nulls]) for m in ("edge_g", "dsh_g")}
        o["p_edge"] = float((1 + (ng["edge_g"] >= o["edge_g"]).sum())/(NULLR + 1)); o["p_dsh"] = float((1 + (ng["dsh_g"] >= o["dsh_g"]).sum())/(NULLR + 1))
        o["null_edge"] = [float(ng["edge_g"].mean()), float(np.percentile(ng["edge_g"], 2.5)), float(np.percentile(ng["edge_g"], 97.5))]
        o["null_dsh"] = [float(ng["dsh_g"].mean()), float(np.percentile(ng["dsh_g"], 2.5)), float(np.percentile(ng["dsh_g"], 97.5))]
        o["per_series_edge"] = {nm: float(obs[k][nm]["edge_g"]) for nm in obs[k]}; o["per_series_dsh"] = {nm: float(obs[k][nm]["dsh_g"]) for nm in obs[k]}; o["n_series"] = len(obs[k])
        o["n_pos_edge"] = int(sum(v > 0 for v in o["per_series_edge"].values())); o["n_pos_dsh"] = int(sum(v > 0 for v in o["per_series_dsh"].values()))
        o["window_start"] = {nm: obs[k][nm]["start"] for nm in obs[k]}; o["label"] = STRATS[k][0]
        R["strategies"][k] = o; pe.append(o["p_edge"]); ps.append(o["p_dsh"])
    for k, a, b in zip(keys, holm(np.array(pe)), holm(np.array(ps))): R["strategies"][k]["holm_edge"] = float(a); R["strategies"][k]["holm_dsh"] = float(b)
    R["buy_hold"] = {nm: dict(mu_ann=float(series[nm].mean()*252*100), sharpe=float(series[nm].mean()/series[nm].std()*np.sqrt(252))) for nm in series}
    # simulated worlds
    R["sim"] = {}
    for w in (() if WIDE else ("iid", "trending", "reverting")):
        rr = Parallel(n_jobs=14)(delayed(sim_one)(w, 9000 + b) for b in range(SIMW))
        R["sim"][w] = {k: dict(edge_mean=float(np.mean([x[k][0] for x in rr])), edge_sd=float(np.std([x[k][0] for x in rr])),
                               dsh_mean=float(np.mean([x[k][1] for x in rr])), dsh_sd=float(np.std([x[k][1] for x in rr])),
                               edges=[x[k][0] for x in rr]) for k in SIMK}
    for k in (() if WIDE else SIMK):                     # power: share of worlds whose edge exceeds the 97.5th percentile of the iid-world edge
        thr = np.percentile(R["sim"]["iid"][k]["edges"], 97.5)
        for w in ("trending", "reverting"): R["sim"][w][k]["power"] = float((np.array(R["sim"][w][k]["edges"]) > thr).mean())
        thr_lo = np.percentile(R["sim"]["iid"][k]["edges"], 2.5)
        for w in ("trending", "reverting"): R["sim"][w][k]["power_neg"] = float((np.array(R["sim"][w][k]["edges"]) < thr_lo).mean())
    for w in R["sim"]:
        for k in SIMK: R["sim"][w][k].pop("edges")
    json.dump(R, open(f"results_tournament{OUT}.json", "w"), indent=1)

    # ---------------- figures
    lab = [f"{k} {STRATS[k][0]}" for k in keys]; y = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for a, (key, nk, ttl) in zip(ax, (("edge_g", "null_edge", "Return edge per invested day (% a year vs. the stock's own drift)"), ("dsh_g", "null_dsh", "Sharpe ratio minus buy-and-hold Sharpe"))):
        for yi, k in zip(y, keys):
            o = R["strategies"][k]; a.plot(o[nk][1:], [yi, yi], color="#999", lw=6, alpha=.5, solid_capstyle="butt")
            sig = o["holm_edge" if key == "edge_g" else "holm_dsh"] < .05
            a.plot(o[key], yi, "o", color="#c0392b" if sig else "#1f4e79", ms=6)
        a.axvline(0, color="k", lw=.8); a.set_title(ttl, fontsize=8)
    ax[0].set_yticks(y); ax[0].set_yticklabels(lab); ax[0].set_xlabel("grey band = 95% range if the signal were useless"); ax[1].set_xlabel("red dot would mark survival of Holm correction (none do)")
    plt.tight_layout(); plt.savefig(f"figures/s1_tournament{OUT}.png"); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.array([R["strategies"][k]["p5"] for k in keys])*100; g = np.array([R["strategies"][k]["edge_n"] for k in keys])
    ax[0].scatter(x, g, c="#1f4e79"); [ax[0].annotate(k, (x[i], g[i]), fontsize=7, xytext=(3, 3), textcoords="offset points") for i, k in enumerate(keys)]
    ax[0].axhline(0, color="k", lw=.8); ax[0].set_xlabel("5% worst trade (% return)"); ax[0].set_ylabel("net edge (% a year)"); ax[0].set_title("Same average, different tails", fontsize=9)
    ex = np.array([R["strategies"][k]["expo"] for k in keys])*100; sh = np.array([R["strategies"][k]["dsh_n"] for k in keys])
    ax[1].scatter(ex, sh, c="#1f4e79"); [ax[1].annotate(k, (ex[i], sh[i]), fontsize=7, xytext=(3, 3), textcoords="offset points") for i, k in enumerate(keys)]
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xlabel("time invested (%)"); ax[1].set_ylabel("net Sharpe minus buy-and-hold"); ax[1].set_title("Being out of the market vs. Sharpe", fontsize=9)
    plt.tight_layout(); plt.savefig(f"figures/s2_risk{OUT}.png"); plt.close()

    if WIDE: print("done"); sys.exit()
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); wd = .25
    for j, w in enumerate(("iid", "trending", "reverting")):
        ax.bar(np.arange(len(SIMK)) + (j - 1)*wd, [R["sim"][w][k]["edge_mean"] for k in SIMK], wd, yerr=[R["sim"][w][k]["edge_sd"] for k in SIMK], capsize=2, label=w)
    ax.set_xticks(range(len(SIMK))); ax.set_xticklabels([f"{k}\n{STRATS[k][0]}" for k in SIMK], fontsize=7); ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("edge (% a year), mean ± sd over worlds"); ax.legend(title="simulated world", fontsize=7); ax.set_title("Positive control: edges appear only when the world contains them", fontsize=9)
    plt.tight_layout(); plt.savefig(f"figures/s3_worlds{OUT}.png"); plt.close()
    print("done")
