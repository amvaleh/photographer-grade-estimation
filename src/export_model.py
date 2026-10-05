"""Train the production grader and write the artifacts the service loads.

Same pipeline that was evaluated (count-free: mean photo score + profile, each photographer weighs equally in
stage 1, one stage-2 model per photo-count bucket) but built from the shared src/grader_core.py so that
training features and serving features come from the same code.

  python src/export_model.py --out models/final                                   # final model, all graded photographers
  python src/export_model.py --out models/holdout --train-before 2023-01-01       # for the end-to-end validation

Stage 1: 5-fold cross-fitting over photographers gives out-of-fold photo scores (what stage 2 trains on), then one
         final MLP on everything is saved.
Stage 2: per bucket K in grader_core.KS: random K photos per photographer (3 draws, folds grouped by photographer),
         ridge on [mean score | profile], ordinal cut-points tuned for quadratic kappa on out-of-fold predictions,
         and the out-of-fold confusion matrix kept as the probability calibration.
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grader_core as gc

D = lambda *p: os.path.join(ROOT, "data", *p)
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(ROOT, "models", "final"))
ap.add_argument("--train-before", help="only photographers who joined before this date (hold-out validation model)")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--draws", type=int, default=3)
ap.add_argument("--emb", default="embeddings_clip,embeddings_clip_rest")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0); torch.set_num_threads(8)
N_CLASSES = gc.N_CLASSES

# ------------------------------------------------------------ data (same filters as train_photo_level.py)
ph = pd.read_csv(D("photographers.csv"), parse_dates=["signup_at"]); photos = pd.read_csv(D("photos_all.csv"))
shards = [np.load(f) for d in a.emb.split(",") for f in sorted(glob.glob(D(d, "shard_*.npz")))]
emb_ids = np.concatenate([s["photo_id"] for s in shards]); emb = np.concatenate([s["emb"] for s in shards]).astype("float32")
row = {int(i): k for k, i in enumerate(emb_ids)}
ph = ph[ph.grade.isin(range(0, 7))].copy(); ph["y"] = ph.grade.clip(upper=4).astype(int)
if a.train_before: ph = ph[ph.signup_at < a.train_before]
photos = photos[photos.photo_id.isin(row) & photos.photographer_id.isin(ph.photographer_id)]
ph = ph[ph.photographer_id.isin(photos.photographer_id)].reset_index(drop=True)
photos = photos[photos.photographer_id.isin(ph.photographer_id)].reset_index(drop=True)
pidx = {p: i for i, p in enumerate(ph.photographer_id)}
photos["pi"] = photos.photographer_id.map(pidx)
X_all = emb[photos.photo_id.map(row).to_numpy()]; pi = photos.pi.to_numpy(); y = ph.y.to_numpy()
rows_of = {i: np.asarray(ix) for i, ix in photos.groupby("pi").indices.items()}
print(f"training on {len(ph)} photographers, {len(photos)} photos; classes {np.bincount(y).tolist()}", flush=True)

# ------------------------------------------------------------ profile preprocessing (fitted on the training photographers only)
prof = pd.read_csv(D("profile.csv")).set_index("photographer_id").reindex(ph.photographer_id)
gear_df = pd.read_csv(D("gear.csv")); gear_df = gear_df[gear_df.photographer_id.isin(ph.photographer_id)].copy()
gear_df["name"] = (gear_df.brand.fillna("") + " " + gear_df.model.fillna("")).str.strip()
gear = gc.GearEmbedder(D("gear_bert.npz"))
gear.embed_names(gear_df.name.unique()); gear.save(os.path.join(a.out, "gear_bert.npz"))
items = {k: gear_df[gear_df.kind == k].groupby("photographer_id").name.apply(list).to_dict() for k in ("camera", "lens")}

pre = {"median_years": float(np.nanmedian(np.log1p(prof.years_been_photographer.to_numpy(float)))),
       "median_projects": float(np.nanmedian(np.log1p(prof.projects_payed_count.to_numpy(float))))}
for kind in ("camera", "lens"):
    agg = np.stack([gc.gear_aggregate(items[kind].get(pid, []), gear)[0] for pid in ph.photographer_id])
    pca = PCA(16, random_state=0).fit(agg)
    pre[f"pca_{kind}_mean"] = pca.mean_.astype(np.float32); pre[f"pca_{kind}_components"] = pca.components_.astype(np.float32)
np.savez(os.path.join(a.out, "preprocess.npz"), **pre)
meta = np.stack([gc.profile_features(pre, prof.years_been_photographer.iloc[i], prof.projects_payed_count.iloc[i],
                                     items["camera"].get(pid, []), items["lens"].get(pid, []), gear)
                 for i, pid in enumerate(ph.photographer_id)])
print(f"profile features: {meta.shape[1]}", flush=True)

# ------------------------------------------------------------ stage 1
def train_mlp(tr_rows):
    X = X_all[tr_rows]; mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd); yt = torch.tensor(y[pi[tr_rows]], dtype=torch.float32)
    cw = 1.0 / np.sqrt(np.bincount(y[pi[tr_rows]], minlength=N_CLASSES)); wt = torch.tensor(cw[y[pi[tr_rows]]], dtype=torch.float32)
    e = 1.0 / np.bincount(pi)[pi[tr_rows]]; wt = wt * torch.tensor(e / e.mean(), dtype=torch.float32)   # every photographer weighs equally
    net = gc.make_net(X.shape[1]); n = len(Xt)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * ((n + 511) // 512))
    for _ in range(a.epochs):
        net.train()
        for b in torch.randperm(n).split(512):
            loss = (wt[b] * (net(Xt[b]).squeeze(1) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net, mu, sd

def score(net, mu, sd, rows):
    with torch.no_grad(): return net(torch.tensor((X_all[rows] - mu) / sd)).squeeze(1).numpy()

t0 = time.time(); oof = np.zeros(len(photos), np.float32)
for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(len(ph)))):
    net, mu, sd = train_mlp(np.where(np.isin(pi, tr))[0]); r = np.where(np.isin(pi, te))[0]; oof[r] = score(net, mu, sd, r)
    print(f"  stage-1 fold {f + 1}/5 done ({time.time() - t0:.0f}s)", flush=True)
net, mu, sd = train_mlp(np.arange(len(photos)))
torch.save({"state": net.state_dict(), "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}, os.path.join(a.out, "stage1.pt"))

# ------------------------------------------------------------ stage 2, one model per photo-count bucket
def qwk(t, p): return cohen_kappa_score(t, p, weights="quadratic")
def fit_cuts(p, t):
    cuts = np.arange(N_CLASSES - 1) + 0.5
    for _ in range(4):
        for k in range(len(cuts)):
            lo = cuts[k - 1] + 0.01 if k else -1; hi = cuts[k + 1] - 0.01 if k < len(cuts) - 1 else N_CLASSES
            g = np.linspace(lo, hi, 80); cuts[k] = g[np.argmax([qwk(t, np.searchsorted(np.r_[cuts[:k], v, cuts[k + 1:]], p)) for v in g])]
    return cuts
def ridge(): return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9)))

stage2, summary = {}, {}
for K in gc.KS:
    X, Y, G = [], [], []
    for d in range(1 if K >= 40 else a.draws):
        rng = np.random.default_rng(1000 * K + d)
        means = np.array([oof[rng.permutation(rows_of[i])[:K]].mean() for i in range(len(ph))])
        X.append(np.column_stack([means, meta])); Y.append(y); G.append(np.arange(len(ph)))
    X, Y, G = np.vstack(X), np.concatenate(Y), np.concatenate(G)
    o2 = np.zeros(len(Y))
    for tr, te in KFold(5, shuffle=True, random_state=2).split(np.arange(len(ph))):
        m_tr, m_te = np.isin(G, tr), np.isin(G, te)
        o2[m_te] = ridge().fit(X[m_tr], Y[m_tr]).predict(X[m_te])
    cuts = fit_cuts(o2, Y); pred = np.searchsorted(cuts, o2)
    C = np.ones((N_CLASSES, N_CLASSES))                                  # Laplace-smoothed P(true grade | predicted grade)
    for p_, t_ in zip(pred, Y): C[p_, t_] += 1
    C /= C.sum(1, keepdims=True)
    m = ridge().fit(X, Y)
    stage2.update({f"b{K}_mean": m[0].mean_, f"b{K}_scale": m[0].scale_, f"b{K}_coef": m[1].coef_, f"b{K}_intercept": np.array(m[1].intercept_),
                   f"b{K}_cuts": cuts, f"b{K}_calibration": C})
    summary[K] = {"oof_qwk": round(qwk(Y, pred), 3), "oof_acc": round(float((pred == Y).mean()), 3),
                  "oof_within1": round(float((np.abs(pred - Y) <= 1).mean()), 3), "alpha": float(m[1].alpha_)}
    print(f"  bucket K={K}: {summary[K]}", flush=True)
np.savez(os.path.join(a.out, "stage2.npz"), **stage2)

cfg = {"version": f"clip-b16-mean+profile-{time.strftime('%Y%m%d')}", "trained_at": time.strftime("%Y-%m-%d %H:%M"),
       "train_before": a.train_before, "n_photographers": int(len(ph)), "n_photos": int(len(photos)),
       "class_counts": np.bincount(y).tolist(), "ks": gc.KS, "oof_by_bucket": summary,
       "note": "oof_* are out-of-fold same-period estimates (optimistic); the temporal hold-out is the realistic number."}
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
print("saved to", a.out, flush=True)
