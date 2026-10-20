"""
03_crossfit_models.py
Stage 3 - Cross-fitted (out-of-fold) risk predictions for EVERY child, so that every
subgroup (incl. small provinces and 3T regencies) is evaluated on data unseen by the model.

Design
  * 5 outer folds, grouped by PSU (no household/cluster appears in train and test)
  * inside each training fold: 80% PSUs = fitting set, 20% PSUs = calibration set
  * learner: histogram gradient boosting (HGB), the best leakage-free model in the
    previous SKI 2023 framework study (same hyper-parameters, unweighted fit)
  * national Platt recalibration + single national threshold t* (weighted Youden J)
    fitted on the calibration set  -> M0 (status quo)

Model variants (same folds):
  base      121 leakage-free predictors (includes urban and health-insurance status)
  unaware   base minus urban and health-insurance status ("fairness through unawareness")
  aware     base plus region (7) and 3T flag
  reweight  base, group-balanced training weights (each region gets equal total weight)

Sources-of-disparity analyses:
  region-specific models  HGB trained only on the region's own fitting data (same folds)
  leave-one-region-out    HGB trained on 6 regions, Platt on their calibration PSUs, tested on the 7th
  domain classifier       HGB (3-fold) separating each region from the rest using the predictors

Outputs: data/crossfit.pkl, results/T_domain_classifier.csv, results/log_03.txt
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from common import *
from common import _wlogreg

t00 = time.time()
A = pd.read_pickle(ANALYTIC)
df, FEAT = A["df"].reset_index(drop=True).copy(), list(A["features"])
att = pd.read_pickle(os.path.join(DATA_DIR, "attributes.pkl"))
assert (att["ID ART"].values == df["ID ART"].values).all()
df["region"] = att.region.values
df["t3"] = att.t3.values
JKN = "11. Kepemilikan Jaminan Kesehatan"
CAT_ALL = [c for c in FEAT + ["region"] if df[c].dtype == object or str(df[c].dtype).startswith("str")]
for c in CAT_ALL:
    df[c] = df[c].astype(object)

FEATS = {
    "base": FEAT,
    "unaware": [f for f in FEAT if f not in ("urban", JKN)],
    "aware": FEAT + ["region", "t3"],
}
y, w, g = df.stunted.values, df[W].values, df[PSU].values
REG = df.region.values
REGIONS = ["Sumatera", "Java", "Bali-Nusa Tenggara", "Kalimantan", "Sulawesi", "Maluku", "Papua"]


def make(feats, max_iter=300):
    cat = [c for c in feats if c in CAT_ALL]; num = [c for c in feats if c not in cat]
    pre = ColumnTransformer([("num", "passthrough", num),
                             ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                                    encoded_missing_value=np.nan), cat)])
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=50, l2_regularization=1.0,
                                         categorical_features=[False] * len(num) + [True] * len(cat),
                                         early_stopping=False, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)]), num + cat


def fit_predict(feats, tr, pred_sets, sw=None, max_iter=300):
    m, cols = make(feats, max_iter)
    m.fit(df.loc[tr, cols], y[tr], clf__sample_weight=sw)
    return [m.predict_proba(df.loc[s, cols])[:, 1] for s in pred_sets]


def lgt(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def platt(p, yy, ww):
    b = _wlogreg(np.column_stack([np.ones(len(p)), lgt(p)]), yy.astype(float), ww)
    return b


def apply_platt(b, p):
    return 1 / (1 + np.exp(-(b[0] + b[1] * lgt(p))))


log = open(os.path.join(RES_DIR, "log_03.txt"), "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


outer = list(GroupKFold(n_splits=N_FOLDS).split(df, y, groups=g))
fold = np.zeros(len(df), int)
role = {}  # fold -> (fit idx, cal idx, test idx)
for k, (tr, te) in enumerate(outer):
    fold[te] = k
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + k)
    a, b = next(gss.split(tr, groups=g[tr]))
    role[k] = (tr[a], tr[b], te)
    say(f"fold {k}: fit {len(tr[a])}, cal {len(tr[b])}, test {len(te)}")

# raw scores on calibration and test parts, per variant
RAW = {v: {"cal": np.full(len(df), np.nan), "test": np.full(len(df), np.nan)} for v in
       ["base", "unaware", "aware", "reweight", "region_specific"]}
CAL_FOLD = np.full(len(df), -1)
for k, (fi, ca, te) in role.items():
    CAL_FOLD[ca] = k
    for v in ["base", "unaware", "aware", "reweight"]:
        t = time.time()
        feats = FEATS["base"] if v == "reweight" else FEATS[v]
        sw = None
        if v == "reweight":
            share = pd.Series(REG[fi]).value_counts(normalize=True)
            sw = (1.0 / pd.Series(REG[fi]).map(share).values) / len(REGIONS)
            sw = sw / sw.mean()
        pc, pt = fit_predict(feats, fi, [ca, te], sw)
        RAW[v].setdefault("cal_by_fold", {})[k] = pc
        RAW[v]["test"][te] = pt
        say(f"  fold {k} {v:9s} test AUROC {roc_auc_score(y[te], pt, sample_weight=w[te]):.4f} ({time.time()-t:.0f}s)")
    # region-specific models
    t = time.time()
    rs_cal = np.full(len(ca), np.nan); rs_te = np.full(len(te), np.nan)
    for r in REGIONS:
        fr, cr, tr_ = fi[REG[fi] == r], np.where(REG[ca] == r)[0], np.where(REG[te] == r)[0]
        pc, pt = fit_predict(FEATS["base"], fr, [ca[cr], te[tr_]])
        rs_cal[cr] = pc; rs_te[tr_] = pt
    RAW["region_specific"].setdefault("cal_by_fold", {})[k] = rs_cal
    RAW["region_specific"]["test"][te] = rs_te
    say(f"  fold {k} region-specific models done ({time.time()-t:.0f}s)")

# leave-one-region-out (transportability)
LORO = np.full(len(df), np.nan); LORO_RAW = np.full(len(df), np.nan)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
for r in REGIONS:
    t = time.time()
    te = np.where(REG == r)[0]; tr = np.where(REG != r)[0]
    a, b = next(gss.split(tr, groups=g[tr]))
    pc, pt = fit_predict(FEATS["base"], tr[a], [tr[b], te])
    bb = platt(pc, y[tr[b]], w[tr[b]])
    LORO[te] = apply_platt(bb, pt); LORO_RAW[te] = pt
    say(f"LORO {r:20s} AUROC {roc_auc_score(y[te], pt, sample_weight=w[te]):.4f} ({time.time()-t:.0f}s)")

# domain classifier: how distinct is each region in predictor space?
dc = []
for r in REGIONS:
    t = time.time()
    lab = (REG == r).astype(int)
    oof = np.zeros(len(df))
    for a, b in GroupKFold(n_splits=3).split(df, lab, groups=g):
        m, cols = make(FEATS["unaware"], max_iter=100)
        m.fit(df.loc[a, cols], lab[a]); oof[b] = m.predict_proba(df.loc[b, cols])[:, 1]
    dc.append(dict(region=r, n=int(lab.sum()), domain_auroc=roc_auc_score(lab, oof)))
    say(f"domain classifier {r:20s} AUROC {dc[-1]['domain_auroc']:.4f} ({time.time()-t:.0f}s)")
pd.DataFrame(dc).round(4).to_csv(os.path.join(RES_DIR, "T_domain_classifier.csv"), index=False)

pd.to_pickle(dict(fold=fold, role=role, RAW=RAW, LORO=LORO, LORO_RAW=LORO_RAW,
                  feats={k: len(v) for k, v in FEATS.items()}), os.path.join(DATA_DIR, "crossfit.pkl"))
say(f"done in {(time.time()-t00)/60:.1f} min")
