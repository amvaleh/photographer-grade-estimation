"""Baseline: predict Kadro's photographer grade from a fixed sample of portfolio photos.

Target: grade 0..6 merged to 5 ordinal classes {0,1,2,3,4+}. Pending (-1) and ungraded are excluded.
One row per photographer, built from the first N_PHOTOS photos of the manifest (already a round-robin
across the photographer's expertises), so photo count cannot leak the label.
Ordinal model = Ridge regression on the grade + cut-points tuned for quadratic weighted kappa on
out-of-fold training predictions. Evaluated (a) temporally: train on signup < 2023, test on 2023+;
(b) 5-fold CV over all labelled photographers.
"""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = lambda *p: os.path.join(ROOT, "data", *p)
N_PHOTOS, SPLIT_DATE, N_CLASSES = 12, "2023-01-01", 5
rng = np.random.default_rng(0)

# ---------- data ----------
ph = pd.read_csv(D("photographers.csv"), parse_dates=["signup_at"])
photos = pd.read_csv(D("photos.csv"))
shards = [np.load(f) for f in sorted(glob.glob(D("embeddings", "shard_*.npz")))]
emb_ids = np.concatenate([s["photo_id"] for s in shards]); emb = np.concatenate([s["emb"] for s in shards]).astype("float32")
row = {int(i): k for k, i in enumerate(emb_ids)}

BOOL = {"t": True, "f": False}
for c in ["approved", "has_studio", "educated", "video_service", "has_instagram", "has_website", "has_online_portfolio"]:
    ph[c] = ph[c].map(BOOL)
photos["has_exif"] = photos["has_exif"].map(BOOL)
ph = ph[ph.grade.isin(range(0, 7))].copy()
ph["y"] = ph.grade.clip(upper=4).astype(int)
photos = photos[photos.photo_id.isin(row) & photos.photographer_id.isin(ph.photographer_id)]
photos = photos.groupby("photographer_id").head(N_PHOTOS)
cnt = photos.groupby("photographer_id").size()
ph = ph[ph.photographer_id.map(cnt).fillna(0) >= N_PHOTOS].reset_index(drop=True)
photos = photos[photos.photographer_id.isin(ph.photographer_id)]
print(f"{len(ph)} labelled photographers with >= {N_PHOTOS} photos;", ph.y.value_counts().sort_index().to_dict())

img, exif = [], []
for pid, g in photos.groupby("photographer_id", sort=False):
    e = emb[[row[int(i)] for i in g.photo_id]]
    c = e.mean(0)
    spread = np.linalg.norm(e - c, axis=1).mean()          # how varied the portfolio is
    img.append(np.concatenate([c, [spread]]))
    exif.append([g.has_exif.mean(), g.camera_model.nunique(), g.software.notna().mean()])
order = list(photos.groupby("photographer_id", sort=False).groups)
img = pd.DataFrame(img, index=order).loc[ph.photographer_id].to_numpy()
exif = pd.DataFrame(exif, index=order).loc[ph.photographer_id].to_numpy()
tab = ph[["has_studio", "educated", "video_service", "has_instagram", "has_website", "has_online_portfolio",
          "n_experiences", "n_expertises"]].astype(float).to_numpy()
meta = np.hstack([tab, exif])
FEATURES = {"metadata only": meta, "images only": img, "images + metadata": np.hstack([img, meta])}

# ---------- ordinal model ----------
def qwk(a, b): return cohen_kappa_score(a, b, weights="quadratic")
def to_class(p, cuts): return np.searchsorted(cuts, p)
def fit_cuts(p, y):
    cuts = np.arange(N_CLASSES - 1) + 0.5
    for _ in range(4):                                      # coordinate search over each cut-point
        for k in range(len(cuts)):
            lo = cuts[k - 1] + 0.01 if k else -1; hi = cuts[k + 1] - 0.01 if k < len(cuts) - 1 else N_CLASSES
            grid = np.linspace(lo, hi, 80)
            cuts[k] = grid[np.argmax([qwk(y, to_class(p, np.r_[cuts[:k], t, cuts[k + 1:]])) for t in grid])]
    return cuts
def make_model(): return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 9)))
def fit_predict(Xtr, ytr, Xte):
    oof = np.zeros(len(ytr))
    for a, b in KFold(5, shuffle=True, random_state=0).split(Xtr):
        oof[b] = make_model().fit(Xtr[a], ytr[a]).predict(Xtr[b])
    cuts = fit_cuts(oof, ytr)
    return to_class(make_model().fit(Xtr, ytr).predict(Xte), cuts)
def report(y, p):
    boot = [qwk(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
    return dict(n=int(len(y)), qwk=round(qwk(y, p), 3), qwk_ci=[round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
                mae_steps=round(float(np.abs(y - p).mean()), 3), acc=round(float((y == p).mean()), 3),
                within1=round(float((np.abs(y - p) <= 1).mean()), 3))

y = ph.y.to_numpy(); train = (ph.signup_at < SPLIT_DATE).to_numpy()
out = {"temporal": {}, "cv": {}}
print(f"\ntemporal split: train {train.sum()} (<{SPLIT_DATE}), test {(~train).sum()};",
      "test classes", np.bincount(y[~train], minlength=N_CLASSES).tolist())
maj = np.full((~train).sum(), np.bincount(y[train]).argmax())
out["temporal"]["majority class"] = report(y[~train], maj)
for name, X in FEATURES.items():
    p = fit_predict(X[train], y[train], X[~train])
    out["temporal"][name] = report(y[~train], p)
    if name == "images + metadata": print("confusion (rows = true, cols = predicted):\n", confusion_matrix(y[~train], p, labels=range(N_CLASSES)))
    pcv = np.zeros(len(y), int)
    for a, b in KFold(5, shuffle=True, random_state=1).split(X): pcv[b] = fit_predict(X[a], y[a], X[b])
    out["cv"][name] = report(y, pcv)
for split in out:
    print(f"\n== {split} ==")
    print(pd.DataFrame(out[split]).T.to_string())
json.dump(out, open(D("results_baseline.json"), "w"), indent=1)
