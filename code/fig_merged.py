"""Data figure (t1m_data.png: TSLA price, return histogram, ACF of squared returns) and out-of-sample calibration figure (t4m_oos.png)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style  # shared figure style
from scipy import stats
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 9})
px = pd.read_csv("data/basket_close.csv", index_col=0, parse_dates=True); R = json.load(open("results_trade.json"))
pxs = px["TSLA"].dropna(); r = np.log(pxs).diff().dropna()
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ax[0].plot(pxs.index, pxs.values, lw=.8); ax[0].set_title("TSLA price")
ax[1].hist(r, 80, density=True, alpha=.6); xx = np.linspace(r.min(), r.max(), 300); ax[1].plot(xx, stats.norm.pdf(xx, r.mean(), r.std(ddof=0)), color="#eb6834"); ax[1].set_title("TSLA daily log-returns vs Normal")
a_ = r.values**2 - (r.values**2).mean(); ac = [(a_[:len(a_)-k]*a_[k:]).sum()/(a_**2).sum() for k in range(1, 31)]
ax[2].bar(range(1, 31), ac, width=.6); ax[2].axhline(1.96/np.sqrt(len(r)), color="#eb6834", ls="--"); ax[2].set_title("ACF of r² (TSLA): volatility clusters")
plt.tight_layout(); plt.savefig("figures/t1m_data.png"); plt.close()
O = R["out_of_sample"]; ks = list(O); lab = [f"+{float(k.split('/')[0])*100:.0f}/−{float(k.split('/')[1])*100:.0f}" for k in ks]; x = np.arange(len(ks)); w = .27
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3)); INKC, ORG, BLU = "#0b0b0b", "#eb6834", "#2a78d6"
for xs, key, lab_, col, mk in ((x - .12, "obs", "observed (out of sample)", INKC, "D"), (x, "p0_sum", "M0 fair game", ORG, "o"), (x + .12, "p1_sum", "M1 Brownian", BLU, "o")):
    ax[0].plot(xs, [O[k][key] for k in ks], mk, color=col, ms=7, label=lab_, markeredgecolor="#fcfcfb")
ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set_xlim(-.5, len(ks) - .5); ax[0].set_ylim(.3, .7); ax[0].set_title("P(target before stop), by rule"); ax[0].legend(loc="upper left", ncol=1)
for xs, key, lab_, col in ((x - .08, "p0_brier", "M0 fair game", ORG), (x + .08, "p1_brier", "M1 Brownian", BLU)):
    ax[1].plot(xs, [O[k][key] for k in ks], "o", color=col, ms=7, label=lab_, markeredgecolor="#fcfcfb")
ax[1].axhline(.25, color="#52514e", ls="--", lw=1); ax[1].text(len(ks) - .55, .2503, "always say 50%", ha="right", va="bottom", fontsize=7.5, color="#52514e")
ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set_xlim(-.5, len(ks) - .5); ax[1].set_title("Brier score (lower is better)"); ax[1].legend(loc="upper left")
plt.tight_layout(); plt.savefig("figures/t4m_oos.png"); plt.close(); print("ok")
