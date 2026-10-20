"""
05_figures.py - manuscript figures (300 dpi PNG) from results/*.csv and data/scenarios.pkl
  Fig1_study_design.png        study design (cross-fitting, scenarios)
  Fig2_subgroup_M0.png         M0 calibration-in-the-large, FNR and AUROC by subgroup (95% CI)
  Fig3_province_eb.png         province estimates (raw and empirical-Bayes shrunken)
  Fig4_sources_by_region.png   global vs region-specific vs leave-one-region-out + domain AUROC
  Fig5_mitigation.png          comparison of mitigation strategies (gaps with 95% CI)
  Fig6_reliability_region.png  reliability curves by region, M0 vs M1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import *
from plot_style import apply_style, COLORBLIND_SAFE_PALETTE as PAL

apply_style(9)
T = pd.read_csv(os.path.join(RES_DIR, "T_group_metrics.csv"))
G = pd.read_csv(os.path.join(RES_DIR, "T_gaps.csv"))
EB = pd.read_csv(os.path.join(RES_DIR, "T_province_eb.csv"))
DC = pd.read_csv(os.path.join(RES_DIR, "T_domain_classifier.csv"))
REGIONS = ["Sumatera", "Java", "Bali-Nusa Tenggara", "Kalimantan", "Sulawesi", "Maluku", "Papua"]


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name)); plt.close(fig); print("saved", name)


# ---------------------------------------------------------------- Fig 1
fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 62)
def box(x, y, wd, ht, txt, fc="white", ec="#333333", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), wd, ht, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    ax.text(x + wd / 2, y + ht / 2, txt, ha="center", va="center", fontsize=7.4, fontweight="bold" if bold else "normal")
def arr(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))
box(2, 50, 30, 9, "SKI 2023 under-five module\nn = 84,206; 33,131 PSUs; 970 strata\n121 leakage-free predictors", bold=True)
box(36, 50, 30, 9, "Sensitive attributes (audit only)\nregion (7), 3T regency (Perpres 63/2020),\nresidence, housing quintile, insurance")
box(70, 50, 28, 9, "Survey design kept throughout\nweights in every metric;\nPSU-in-strata bootstrap")
box(2, 33, 46, 11, "5-fold PSU-grouped cross-fitting\ninside each training fold: 80% PSUs fit HGB,\n20% PSUs = calibration set (Platt, t*, patches)\n-> out-of-fold prediction for every child")
box(52, 33, 46, 11, "Sources of disparity\nregion-specific models; leave-one-region-out;\ndomain classifier; province random-effects\n(empirical-Bayes shrinkage)")
box(2, 14, 96, 12, "Scenarios compared\nM0 status quo (national recalibration, single threshold)  |  Unawareness  |  Awareness\n"
    "M1 region recalibration  |  M2 region thresholds (equal FNR)  |  M3 group-balanced reweighting  |  M4 multicalibration")
box(2, 1, 96, 8, "Survey-weighted metrics per group: AUROC, calibration intercept, O/E, ECE, FNR, FPR, PPV, referrals per 1,000\n"
    "Disparity = max - min across groups (95% CI)", fc="#EEF4FA")
arr(17, 50, 17, 44.8); arr(51, 50, 30, 44.8); arr(84, 50, 75, 44.8)
arr(25, 33, 25, 26.8); arr(75, 33, 75, 26.8); arr(50, 14, 50, 9.8)
save(fig, "Fig1_study_design.png")

# ---------------------------------------------------------------- Fig 2
M0 = T[(T.scenario == "M0") & ~T.attribute.isin(["Region x residence", "Intersection"])].copy()
order_attr = ["Overall", "Region", "Underdeveloped district", "Residence", "Housing quintile", "Health insurance", "Age", "Sex"]
M0["ord"] = M0.attribute.map({a: i for i, a in enumerate(order_attr)})
M0 = M0.sort_values(["ord"], kind="stable").reset_index(drop=True)
labels = [f"{g}" if a != "Overall" else "All children" for a, g in zip(M0.attribute, M0.group)]
ypos = []; yv = 0; prev = None; heads = []
for a in M0.attribute:
    if a != prev:
        if prev is not None:
            yv += 0.8
        heads.append((yv, a)); prev = a
    ypos.append(yv); yv += 1
ypos = -np.array(ypos)
fig, axs = plt.subplots(1, 3, figsize=(7.4, 7.6), sharey=True)
for ax, (mt, ttl, ref) in zip(axs, [("cal_intercept", "Calibration intercept\n(>0 = risk underestimated)", 0),
                                     ("fnr", "False-negative rate at t*\n(stunted children missed)", None),
                                     ("auroc", "AUROC", None)]):
    v = M0[mt].values; lo = M0[mt + "_lo"].values; hi = M0[mt + "_hi"].values
    col = np.where(M0.attribute.values == "Overall", PAL[7], PAL[0])
    ax.errorbar(v, ypos, xerr=[v - lo, hi - v], fmt="none", ecolor="#777777", elinewidth=0.8)
    ax.scatter(v, ypos, c=col, s=14, zorder=3)
    allv = M0.loc[M0.attribute == "Overall", mt].values[0]
    ax.axvline(ref if ref is not None else allv, color=PAL[1], lw=0.8, ls="--")
    ax.set_title(ttl, fontsize=8.5)
axs[0].set_yticks(ypos); axs[0].set_yticklabels(labels, fontsize=7.2)
for yh, a in heads:
    if a != "Overall":
        axs[0].text(-0.02, -yh + 0.75, a, transform=axs[0].get_yaxis_transform(), ha="right", fontsize=7.4, fontweight="bold")
fig.suptitle("Status-quo model (M0): survey-weighted performance by subgroup (95% PSU-bootstrap CI)", fontsize=9, fontweight="bold")
save(fig, "Fig2_subgroup_M0.png")

# ---------------------------------------------------------------- Fig 3
RPAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 7.4), sharey=True)
for ax, (mt, ttl, ref) in zip(axs, [("cal_intercept", "Calibration intercept", 0), ("fnr", "False-negative rate at t*", None)]):
    d = EB[EB.metric == mt].sort_values("eb").reset_index(drop=True)
    yy = np.arange(len(d))
    for i, r in enumerate(REGIONS):
        m = d.region == r
        ax.scatter(d.raw[m], yy[m], s=10, facecolors="none", edgecolors=RPAL[i], lw=0.7)
        ax.errorbar(d.eb[m], yy[m], xerr=[d.eb[m] - d.eb_lo[m], d.eb_hi[m] - d.eb[m]], fmt="o", ms=3.2,
                    color=RPAL[i], elinewidth=0.8, label=r if mt == "cal_intercept" else None)
    if ref is not None:
        ax.axvline(ref, color="#555555", lw=0.8, ls="--")
    ax.set_title(ttl, fontsize=9)
    if mt == "cal_intercept":
        ax.set_yticks(yy); ax.set_yticklabels(d.province, fontsize=6.6); order_prov = list(d.province)
    else:
        # keep the same province order as the first panel
        d = d.set_index("province").loc[order_prov].reset_index()
        ax.cla(); ax.set_title(ttl, fontsize=9)
        for i, r in enumerate(REGIONS):
            m = (d.region == r).values
            ax.scatter(d.raw[m], yy[m], s=10, facecolors="none", edgecolors=RPAL[i], lw=0.7)
            ax.errorbar(d.eb[m], yy[m], xerr=[d.eb[m] - d.eb_lo[m], d.eb_hi[m] - d.eb[m]], fmt="o", ms=3.2,
                        color=RPAL[i], elinewidth=0.8)
        ov = T[(T.scenario == "M0") & (T.attribute == "Overall")].fnr.values[0]
        ax.axvline(ov, color="#555555", lw=0.8, ls="--")
axs[0].set_yticks(yy); axs[0].set_yticklabels(order_prov, fontsize=6.6)
axs[0].legend(fontsize=6.5, loc="upper left")
fig.suptitle("Provinces under M0: empirical-Bayes estimates (filled, 95% CI) and raw estimates (open)", fontsize=9, fontweight="bold")
save(fig, "Fig3_province_eb.png")

# ---------------------------------------------------------------- Fig 4
R = T[T.attribute == "Region"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
x = np.arange(len(REGIONS)); wd = 0.26
for j, (sc, lab) in enumerate([("M0", "Global model (cross-fitted)"), ("RS", "Region-specific model"),
                                ("LORO", "Leave-one-region-out")]):
    d = R[R.scenario == sc].set_index("group").loc[REGIONS]
    axs[0].bar(x + (j - 1) * wd, d.auroc, wd, yerr=[d.auroc - d.auroc_lo, d.auroc_hi - d.auroc],
               color=PAL[j], label=lab, error_kw=dict(lw=0.6))
    axs[1].bar(x + (j - 1) * wd, d.cal_intercept, wd,
               yerr=[d.cal_intercept - d.cal_intercept_lo, d.cal_intercept_hi - d.cal_intercept],
               color=PAL[j], error_kw=dict(lw=0.6))
axs[0].set_ylim(0.5, 0.78); axs[0].set_ylabel("AUROC"); axs[0].set_title("Discrimination", fontsize=9)
axs[1].axhline(0, color="#333333", lw=0.7); axs[1].set_ylabel("Calibration intercept"); axs[1].set_title("Calibration-in-the-large", fontsize=9)
for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels([r.replace("Bali-Nusa Tenggara", "Bali-NT") for r in REGIONS], rotation=35, ha="right", fontsize=7.5)
dcv = DC.set_index("region").loc[REGIONS].domain_auroc.values
for i, v in enumerate(dcv):
    axs[0].text(x[i], 0.765, f"{v:.2f}", ha="center", fontsize=6.5, color="#555555")
axs[0].text(-0.6, 0.775, "domain-classifier AUROC:", fontsize=6.5, color="#555555")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.07))
save(fig, "Fig4_sources_by_region.png")

# ---------------------------------------------------------------- Fig 5
SC = ["M0", "U", "AW", "M1", "M2", "M3", "M4"]
LAB = {"M0": "M0 Status quo", "U": "Unawareness", "AW": "Awareness (+region, 3T)", "M1": "M1 Region recalibration",
       "M2": "M2 Region thresholds", "M3": "M3 Balanced reweighting", "M4": "M4 Multicalibration"}
panels = [("Region", "max_abs_intercept", "Region:\nmax |calibration intercept|"),
          ("Region", "fnr_range", "Region:\nFNR range"),
          ("Underdeveloped district", "fnr_range", "3T vs non-3T:\nFNR gap")]
fig, axs = plt.subplots(1, 5, figsize=(7.6, 2.9), sharey=True)
yy = np.arange(len(SC))[::-1]
for ax, (a, mt, ttl) in zip(axs[:3], panels):
    d = G[(G.attribute == a) & (G.metric == mt)].set_index("scenario").loc[SC]
    cols = [PAL[7] if s == "M0" else PAL[0] for s in SC]
    ax.barh(yy, d.value, color=cols, xerr=[d.value - d.lo, d.hi - d.value], error_kw=dict(lw=0.6), height=0.65)
    ax.set_title(ttl, fontsize=7.4)
ov = T[(T.attribute == "Overall") & T.scenario.isin(SC)].set_index("scenario").loc[SC]
axs[3].errorbar(ov.auroc, yy, xerr=[ov.auroc - ov.auroc_lo, ov.auroc_hi - ov.auroc], fmt="o", ms=3, color=PAL[2], elinewidth=0.7)
axs[3].set_xlim(0.64, 0.68); axs[3].set_title("Overall\nAUROC", fontsize=7.4)
axs[4].barh(yy, ov.referral_per_1000, color=PAL[4], height=0.65); axs[4].set_title("Referrals\nper 1,000", fontsize=7.4)
axs[4].set_xlim(400, 500)
axs[0].set_yticks(yy); axs[0].set_yticklabels([LAB[s] for s in SC], fontsize=7)
for ax in axs:
    ax.tick_params(axis="x", labelsize=6.5)
save(fig, "Fig5_mitigation.png")

# ---------------------------------------------------------------- Fig 6
SCN = pd.read_pickle(os.path.join(DATA_DIR, "scenarios.pkl"))["S"]
att = pd.read_pickle(os.path.join(DATA_DIR, "attributes.pkl"))
yv, wv, reg = att.stunted.values, att[W].values, att.region.values
fig, axs = plt.subplots(2, 4, figsize=(7.6, 4.2), sharex=True, sharey=True)
for i, r in enumerate(REGIONS + ["All"]):
    ax = axs.flat[i]; m = reg == r if r != "All" else np.ones(len(yv), bool)
    for sc, c, lab in [("M0", PAL[1], "M0"), ("M1", PAL[0], "M1")]:
        p = SCN[sc][0][m]; yy = yv[m]; ww = wv[m]
        q = np.quantile(p, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
        mp = [np.average(p[b == k], weights=ww[b == k]) for k in range(10) if (b == k).sum() > 0]
        oy = [np.average(yy[b == k], weights=ww[b == k]) for k in range(10) if (b == k).sum() > 0]
        ax.plot(mp, oy, "o-", ms=2.5, lw=0.9, color=c, label=lab)
    ax.plot([0, 0.7], [0, 0.7], color="#999999", lw=0.6, ls="--")
    ax.set_title(r.replace("Bali-Nusa Tenggara", "Bali-NT"), fontsize=8); ax.set_xlim(0, 0.7); ax.set_ylim(0, 0.7)
axs[0, 0].legend(fontsize=7)
fig.supxlabel("Predicted risk (decile mean)", fontsize=8); fig.supylabel("Observed stunting (weighted)", fontsize=8)
save(fig, "Fig6_reliability_region.png")
