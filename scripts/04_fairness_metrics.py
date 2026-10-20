"""
04_fairness_metrics.py
Stage 4 - Survey-weighted subgroup audit and mitigation comparison.

Scenarios (all cross-fitted, every child predicted by a model that never saw its PSU):
  M0  base model + national Platt + single national threshold t* (status quo)
  U   'fairness through unawareness' (no urban / insurance predictors), national Platt
  AW  'awareness' (+ region and 3T as predictors), national Platt
  M1  base model + region-specific Platt recalibration, national t*
  M2  M0 probabilities + region-specific thresholds chosen so that each region's
      false-negative rate on the calibration set equals the national FNR at t*
  M3  group-balanced reweighting during training + national Platt
  M4  M0 probabilities + multicalibration post-processing (calibration set) over
      region, residence, 3T, insurance, housing quintile, age band, region x residence
  RS  region-specific models (each region's own data only) + region Platt
  LORO leave-one-region-out model (transportability; threshold-free metrics only)

Thresholds, recalibration maps and multicalibration patches are learnt on the
calibration PSUs of each outer fold only and applied to that fold's test PSUs.

Inference: stratified PSU bootstrap (PSUs resampled with replacement within each of the
970 design strata). B = 1000 for M0, 500 for the other scenarios.

Outputs: results/T_group_metrics.csv, T_gaps.csv, T_province_M0.csv, T_province_eb.csv,
         T_intersectional_M0.csv, T_thresholds.csv, T_mitigation_summary.csv, data/scenarios.pkl
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import *
from common import _wlogreg

t00 = time.time()
att = pd.read_pickle(os.path.join(DATA_DIR, "attributes.pkl"))
CF = pd.read_pickle(os.path.join(DATA_DIR, "crossfit.pkl"))
y = att.stunted.values.astype(int); w = att[W].values.astype(float)
psu = att[PSU].values; strata = att[STRATA].values
REG = att.region.values; fold = CF["fold"]; role = CF["role"]; RAW = CF["RAW"]
REGIONS = ["Sumatera", "Java", "Bali-Nusa Tenggara", "Kalimantan", "Sulawesi", "Maluku", "Papua"]
N = len(y)


def lgt(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def platt_b(p, yy, ww):
    return _wlogreg(np.column_stack([np.ones(len(p)), lgt(p)]), yy.astype(float), ww)


def youden_t(p, yy, ww):
    ts = np.linspace(0.05, 0.6, 221)
    best, bt = -1, None
    P = ww[yy == 1].sum(); Q = ww[yy == 0].sum()
    for t in ts:
        pos = p >= t
        j = ww[pos & (yy == 1)].sum() / P - ww[pos & (yy == 0)].sum() / Q
        if j > best:
            best, bt = j, t
    return bt


def fnr(p, yy, ww, t):
    m = yy == 1
    return ww[m & (p < t)].sum() / ww[m].sum()


# ------------------------------------------------------------------ group masks
def masks(intersect=False):
    G = {("Overall", "All children"): np.ones(N, bool)}
    for r in REGIONS:
        G[("Region", r)] = REG == r
    G[("Underdeveloped district", "3T regency")] = att.t3.values == 1
    G[("Underdeveloped district", "Non-3T")] = att.t3.values == 0
    G[("Residence", "Urban")] = att.urban.values == 1
    G[("Residence", "Rural")] = att.urban.values == 0
    for k in range(1, 6):
        G[("Housing quintile", f"Q{k}")] = att.housing_q.values == k
    for lvl in ["PBI (subsidised)", "Other insurance", "Uninsured"]:
        G[("Health insurance", lvl)] = att.insurance.values == lvl
    for a in ["0-5 mo", "6-23 mo", "24-59 mo"]:
        G[("Age", a)] = att.age_band.values == a
    for s in ["Boys", "Girls"]:
        G[("Sex", s)] = att.sex.values == s
    if intersect:
        for r in REGIONS:
            for u, nm in [(1, "urban"), (0, "rural")]:
                G[("Region x residence", f"{r}, {nm}")] = (REG == r) & (att.urban.values == u)
        east = att.east.values == 1
        G[("Intersection", "East, rural, housing Q1")] = east & (att.urban.values == 0) & (att.housing_q.values == 1)
        G[("Intersection", "West, urban, housing Q5")] = (~east) & (att.urban.values == 1) & (att.housing_q.values == 5)
        G[("Intersection", "3T, rural, housing Q1")] = (att.t3.values == 1) & (att.urban.values == 0) & (att.housing_q.values == 1)
    return G


# ------------------------------------------------------------------ build scenarios
def calibrated(variant, per_region=False):
    """Returns calibrated test probabilities, per-row threshold (national t* of that fold),
    and the calibrated calibration-set probabilities by fold."""
    p = np.full(N, np.nan); t = np.full(N, np.nan); calp = {}
    for k, (fi, ca, te) in role.items():
        rc = RAW[variant]["cal_by_fold"][k]; rt = RAW[variant]["test"][te]
        if per_region:
            pc = np.zeros(len(ca)); pt = np.zeros(len(te))
            for r in REGIONS:
                mc = REG[ca] == r; mt = REG[te] == r
                b = platt_b(rc[mc], y[ca][mc], w[ca][mc])
                pc[mc] = sig(b[0] + b[1] * lgt(rc[mc])); pt[mt] = sig(b[0] + b[1] * lgt(rt[mt]))
        else:
            b = platt_b(rc, y[ca], w[ca])
            pc = sig(b[0] + b[1] * lgt(rc)); pt = sig(b[0] + b[1] * lgt(rt))
        p[te] = pt; calp[k] = pc
        t[te] = youden_t(pc, y[ca], w[ca])
    return p, t, calp


S = {}
p0, t0, cal0 = calibrated("base")
S["M0"] = (p0, t0)
TSTAR = {k: float(t0[role[k][2]][0]) for k in role}

for v, nm in [("unaware", "U"), ("aware", "AW"), ("reweight", "M3")]:
    p_, t_, _ = calibrated(v)
    S[nm] = (p_, t_)
p1, _, _ = calibrated("base", per_region=True)
S["M1"] = (p1, t0.copy())
prs, _, _ = calibrated("region_specific", per_region=True)
S["RS"] = (prs, t0.copy())

# M2: region thresholds equalising FNR to national calibration FNR
t2 = np.full(N, np.nan); thr_rows = []
for k, (fi, ca, te) in role.items():
    pc, yc, wc = cal0[k], y[ca], w[ca]
    target = fnr(pc, yc, wc, TSTAR[k])
    for r in REGIONS:
        m = REG[ca] == r
        grid = np.linspace(0.02, 0.8, 781)
        f = np.array([fnr(pc[m], yc[m], wc[m], tt) for tt in grid])
        ok = grid[f <= target + 1e-12]
        tr = ok.max() if len(ok) else grid[0]
        t2[te[REG[te] == r]] = tr
        thr_rows.append(dict(fold=k, region=r, national_t=TSTAR[k], national_cal_fnr=target, region_t=tr))
S["M2"] = (p0.copy(), t2)
pd.DataFrame(thr_rows).round(4).to_csv(os.path.join(RES_DIR, "T_thresholds.csv"), index=False)

# M4: multicalibration (iterative logit patches on group x probability-bin cells)
MC_GROUPS = []
for r in REGIONS:
    MC_GROUPS.append(REG == r)
for u in (0, 1):
    MC_GROUPS.append(att.urban.values == u)
for v in (0, 1):
    MC_GROUPS.append(att.t3.values == v)
for lvl in ["PBI (subsidised)", "Other insurance", "Uninsured"]:
    MC_GROUPS.append(att.insurance.values == lvl)
for q in range(1, 6):
    MC_GROUPS.append(att.housing_q.values == q)
for a in ["0-5 mo", "6-23 mo", "24-59 mo"]:
    MC_GROUPS.append(att.age_band.values == a)
for r in REGIONS:
    for u in (0, 1):
        MC_GROUPS.append((REG == r) & (att.urban.values == u))
EDGES = [0, 0.1, 0.2, 0.3, 0.4, 1.01]


def multicalibrate(pc, idx_c, yc, wc, pt, idx_t, alpha=0.01, min_n=200, eta=0.5, iters=200):
    zc, zt = lgt(pc), lgt(pt); n_upd = 0
    for _ in range(iters):
        best, sel = 0, None
        pcur = sig(zc)
        for gi, G in enumerate(MC_GROUPS):
            gc = G[idx_c]
            for b in range(len(EDGES) - 1):
                m = gc & (pcur >= EDGES[b]) & (pcur < EDGES[b + 1])
                if m.sum() < min_n:
                    continue
                ybar_ = np.average(yc[m], weights=wc[m])
                d = ybar_ - np.average(pcur[m], weights=wc[m])
                n_eff = wc[m].sum() ** 2 / np.sum(wc[m] ** 2)
                se = np.sqrt(max(ybar_ * (1 - ybar_), 1e-4) / n_eff)
                # patch only cells whose miscalibration exceeds alpha AND 2 design-effective SEs
                if abs(d) > max(alpha, 2 * se) and abs(d) / se > best:
                    best, sel = abs(d) / se, (gi, b, m)
        if sel is None:
            break
        gi, b, m = sel
        ybar = np.clip(np.average(yc[m], weights=wc[m]), 1e-4, 1 - 1e-4)
        delta = eta * (lgt(np.array([ybar]))[0] - lgt(np.array([np.average(sig(zc[m]), weights=wc[m])]))[0])
        mt = MC_GROUPS[gi][idx_t] & (sig(zt) >= EDGES[b]) & (sig(zt) < EDGES[b + 1])
        zc[m] += delta; zt[mt] += delta; n_upd += 1
    return sig(zt), n_upd


p4 = np.full(N, np.nan); mc_log = []
for k, (fi, ca, te) in role.items():
    p4[te], nu = multicalibrate(cal0[k], ca, y[ca], w[ca], p0[te], te)
    mc_log.append(dict(fold=k, patches=nu))
S["M4"] = (p4, t0.copy())
print("multicalibration patches:", mc_log)

S["LORO"] = (CF["LORO"], np.full(N, np.nan))
pd.to_pickle(dict(S=S, TSTAR=TSTAR), os.path.join(DATA_DIR, "scenarios.pkl"))
print("t* by fold:", {k: round(v, 4) for k, v in TSTAR.items()})


# ------------------------------------------------------------------ metrics
def cal_int(yy, p, ww):
    lp = lgt(p); a = 0.0
    for _ in range(25):
        mu = sig(lp + a); gr = np.sum(ww * (yy - mu)); h = np.sum(ww * mu * (1 - mu))
        step = gr / h; a += step
        if abs(step) < 1e-9:
            break
    return a


def gm(yy, p, ww, t, full=False):
    d = dict(prev=np.average(yy, weights=ww), mean_pred=np.average(p, weights=ww))
    d["oe"] = d["prev"] / d["mean_pred"]
    d["auroc"] = roc_auc_score(yy, p, sample_weight=ww) if 0 < yy.sum() < len(yy) else np.nan
    d["cal_intercept"] = cal_int(yy, p, ww)
    d["ece"] = w_ece(yy, p, ww)
    if not np.all(np.isnan(t)):
        pos = p >= t
        d["fnr"] = ww[(yy == 1) & ~pos].sum() / ww[yy == 1].sum()
        d["fpr"] = ww[(yy == 0) & pos].sum() / ww[yy == 0].sum()
        d["ppv"] = ww[(yy == 1) & pos].sum() / max(ww[pos].sum(), 1e-12)
        d["referral_per_1000"] = 1000 * ww[pos].sum() / ww.sum()
    if full:
        try:
            d["cal_slope"] = cal_slope_intercept(yy, p, ww)[0]
        except Exception:
            d["cal_slope"] = np.nan
        d["brier"] = w_brier(yy, p, ww)
    return d


# stratified PSU bootstrap multiplicities
codes, psu_idx = np.unique(psu, return_inverse=True)
psu_str = pd.Series(strata).groupby(psu_idx).first().values
order = np.argsort(psu_str, kind="stable")
s_sorted = psu_str[order]
_, start, cnt = np.unique(s_sorted, return_index=True, return_counts=True)
str_of_pos = np.repeat(np.arange(len(start)), cnt)
rng = np.random.default_rng(SEED)


def boot_mult():
    u = rng.random(len(order))
    draw = order[start[str_of_pos] + np.floor(u * cnt[str_of_pos]).astype(int)]
    return np.bincount(draw, minlength=len(codes))[psu_idx].astype(float)


GAP_METRICS = ["auroc", "oe", "cal_intercept", "ece", "fnr", "fpr", "ppv", "referral_per_1000"]
GAP_ATTRS = ["Region", "Underdeveloped district", "Residence", "Housing quintile", "Health insurance", "Age", "Sex"]


def gaps_from(table):
    out = {}
    for a in GAP_ATTRS:
        sub = table[table.attribute == a]
        for m_ in GAP_METRICS:
            if m_ not in sub or sub[m_].isna().all():
                continue
            out[(a, m_ + "_range")] = sub[m_].max() - sub[m_].min()
        if "cal_intercept" in sub:
            out[(a, "max_abs_intercept")] = sub["cal_intercept"].abs().max()
    return out


def evaluate(name, B, intersect=False):
    p, t = S[name]
    G = masks(intersect)
    ok = ~np.isnan(p)
    rows = []
    for (a, lvl), m in G.items():
        m = m & ok
        rows.append(dict(scenario=name, attribute=a, group=lvl, n=int(m.sum()),
                         **gm(y[m], p[m], w[m], t[m], full=True)))
    T = pd.DataFrame(rows)
    point_gaps = gaps_from(T)
    boots = {k: [] for k in point_gaps}; bootg = {(a, l): [] for (a, l) in G}
    for b in range(B):
        mult = boot_mult()
        ww = w * mult
        brow = []
        for (a, lvl), m in G.items():
            m = m & ok & (mult > 0)
            try:
                d = gm(y[m], p[m], ww[m], t[m])
            except Exception:
                continue
            brow.append(dict(attribute=a, group=lvl, **d)); bootg[(a, lvl)].append(d)
        bg = gaps_from(pd.DataFrame(brow))
        for k in boots:
            if k in bg:
                boots[k].append(bg[k])
    for mname in ["auroc", "oe", "cal_intercept", "ece", "fnr", "fpr", "ppv", "referral_per_1000"]:
        lo, hi = [], []
        for (a, lvl) in G:
            v = [d.get(mname, np.nan) for d in bootg[(a, lvl)]]
            v = np.array(v, float); v = v[~np.isnan(v)]
            lo.append(np.percentile(v, 2.5) if len(v) else np.nan); hi.append(np.percentile(v, 97.5) if len(v) else np.nan)
        if mname in T:
            T[mname + "_lo"] = lo; T[mname + "_hi"] = hi
    gp = pd.DataFrame([dict(scenario=name, attribute=a, metric=m_, value=v,
                            lo=np.percentile(boots[(a, m_)], 2.5), hi=np.percentile(boots[(a, m_)], 97.5))
                       for (a, m_), v in point_gaps.items()])
    return T, gp, bootg


allT, allG = [], []
for name in ["M0", "U", "AW", "M1", "M2", "M3", "M4", "RS", "LORO"]:
    t = time.time()
    B = int(os.environ.get("BOOT_M0", 1000)) if name == "M0" else int(os.environ.get("BOOT", 500))
    T, gp, bootg = evaluate(name, B, intersect=(name == "M0"))
    allT.append(T); allG.append(gp)
    if name == "M0":
        BOOT_M0 = bootg
    print(f"{name}: done ({time.time()-t:.0f}s)", flush=True)
TT = pd.concat(allT); GG = pd.concat(allG)
TT.round(5).to_csv(os.path.join(RES_DIR, "T_group_metrics.csv"), index=False)
GG.round(5).to_csv(os.path.join(RES_DIR, "T_gaps.csv"), index=False)
TT[(TT.scenario == "M0") & TT.attribute.isin(["Region x residence", "Intersection"])].round(4) \
  .to_csv(os.path.join(RES_DIR, "T_intersectional_M0.csv"), index=False)

# ------------------------------------------------------------------ provinces (M0) + empirical Bayes
p, t = S["M0"]
prov = att.province.values
prow, pb = [], {}
for pr in sorted(set(prov)):
    m = prov == pr
    prow.append(dict(province=pr, region=REG[m][0], n=int(m.sum()), **gm(y[m], p[m], w[m], t[m])))
    pb[pr] = []
for b in range(int(os.environ.get("BOOT", 500))):
    mult = boot_mult(); ww = w * mult
    for pr in pb:
        m = (prov == pr) & (mult > 0)
        pb[pr].append(gm(y[m], p[m], ww[m], t[m]))
P = pd.DataFrame(prow)
for mname in ["oe", "cal_intercept", "fnr", "auroc"]:
    P[mname + "_se"] = [np.nanstd([d[mname] for d in pb[pr]]) for pr in P.province]
P.round(5).to_csv(os.path.join(RES_DIR, "T_province_M0.csv"), index=False)


def dersimonian_laird(est, se):
    v = se ** 2; wi = 1 / v
    mu_f = np.sum(wi * est) / wi.sum()
    Q = np.sum(wi * (est - mu_f) ** 2); df_ = len(est) - 1
    tau2 = max(0.0, (Q - df_) / (wi.sum() - np.sum(wi ** 2) / wi.sum()))
    ws = 1 / (v + tau2); mu = np.sum(ws * est) / ws.sum()
    I2 = max(0.0, (Q - df_) / Q) if Q > 0 else 0.0
    shrink = tau2 / (tau2 + v)
    eb = mu + shrink * (est - mu)
    eb_se = np.sqrt(shrink * v + (1 - shrink) ** 2 * (1 / ws.sum()))
    from scipy.stats import chi2
    return dict(mu=mu, tau=np.sqrt(tau2), I2=I2, Q=Q, Q_p=1 - chi2.cdf(Q, df_)), eb, eb_se


eb_rows, eb_sum = [], []
for mname, transform in [("cal_intercept", None), ("fnr", None), ("auroc", None)]:
    est = P[mname].values; se = P[mname + "_se"].values
    s, eb, ebse = dersimonian_laird(est, se)
    eb_sum.append(dict(metric=mname, **s))
    for i, pr in enumerate(P.province):
        eb_rows.append(dict(province=pr, region=P.region[i], metric=mname, raw=est[i], se=se[i],
                            eb=eb[i], eb_lo=eb[i] - 1.96 * ebse[i], eb_hi=eb[i] + 1.96 * ebse[i]))
pd.DataFrame(eb_rows).round(5).to_csv(os.path.join(RES_DIR, "T_province_eb.csv"), index=False)
pd.DataFrame(eb_sum).round(5).to_csv(os.path.join(RES_DIR, "T_province_heterogeneity.csv"), index=False)
print(pd.DataFrame(eb_sum).round(4))

# ------------------------------------------------------------------ mitigation summary (comparison table)
rows = []
LABEL = {"M0": "M0 Status quo (global model, national recalibration, single threshold)",
         "U": "Unawareness (no residence / insurance predictors)",
         "AW": "Awareness (+ region and 3T predictors)",
         "M1": "M1 Region-specific recalibration",
         "M2": "M2 Region-specific thresholds (equal FNR)",
         "M3": "M3 Group-balanced reweighting",
         "M4": "M4 Multicalibration",
         "RS": "Region-specific models",
         "LORO": "Leave-one-region-out (transportability)"}
for name in LABEL:
    ov = TT[(TT.scenario == name) & (TT.attribute == "Overall")].iloc[0]
    g = GG[(GG.scenario == name) & (GG.attribute == "Region")].set_index("metric")
    g3 = GG[(GG.scenario == name) & (GG.attribute == "Underdeveloped district")].set_index("metric")
    def ci(gx, k):
        return f"{gx.loc[k,'value']:.3f} ({gx.loc[k,'lo']:.3f}-{gx.loc[k,'hi']:.3f})" if k in gx.index else ""
    rows.append(dict(scenario=name, description=LABEL[name], overall_auroc=ov.auroc, overall_brier=ov.brier,
                     overall_ece=ov.ece, overall_fnr=ov.get("fnr", np.nan),
                     referral_per_1000=ov.get("referral_per_1000", np.nan),
                     region_max_abs_intercept=ci(g, "max_abs_intercept"),
                     region_oe_range=ci(g, "oe_range"), region_fnr_range=ci(g, "fnr_range"),
                     region_auroc_range=ci(g, "auroc_range"),
                     t3_fnr_range=ci(g3, "fnr_range"), t3_oe_range=ci(g3, "oe_range")))
MS = pd.DataFrame(rows)
MS.round(4).to_csv(os.path.join(RES_DIR, "T_mitigation_summary.csv"), index=False)
print(MS.round(4).to_string())
print(f"total {(time.time()-t00)/60:.1f} min")
