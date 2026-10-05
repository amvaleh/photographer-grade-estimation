"""Photo-level model (multiple-instance style): train an MLP on every photo with its photographer's grade as a
weak label, then aggregate photo scores per photographer. Same split/eval protocol as train_baseline.py.

Stage 1: MLP regression on frozen DINOv2 embeddings, trained on ALL available photos (<=24) of training photographers.
Stage 2: per-photographer stats of the photo scores (mean/std/max/min/top-half mean) [+ metadata] -> Ridge,
         cut-points tuned for QWK on cross-fitted (out-of-fold) stage-1 scores. Test = first 12 photos only.
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = lambda *p: os.path.join(ROOT, "data", *p)
N_PHOTOS, SPLIT_DATE, N_CLASSES, EPOCHS = int(os.environ.get("MIN_PHOTOS", 12)), "2023-01-01", 5, int(os.environ.get("EPOCHS", 8))   # N_PHOTOS = minimum portfolio size to be included
torch.set_num_threads(8); SEED = int(os.environ.get("SEED", 0)); torch.manual_seed(SEED); rng = np.random.default_rng(0)

ph = pd.read_csv(D("photographers.csv"), parse_dates=["signup_at"]); photos = pd.read_csv(D(os.environ.get("PHOTOS", "photos.csv")))   # PHOTOS=photos_all.csv -> the whole portfolios
BOOL = {"t": True, "f": False}
for c in ["has_studio", "educated", "video_service", "has_instagram", "has_website", "has_online_portfolio"]: ph[c] = ph[c].map(BOOL)
shards = [np.load(f) for d in os.environ.get("EMB", "embeddings").split(",") for f in sorted(glob.glob(D(d, "shard_*.npz")))]   # EMB may list several folders
emb_ids = np.concatenate([s["photo_id"] for s in shards]); emb = np.concatenate([s["emb"] for s in shards]).astype("float32")
row = {int(i): k for k, i in enumerate(emb_ids)}
ph = ph[ph.grade.isin(range(0, 7))].copy(); ph["y"] = ph.grade.clip(upper=4).astype(int)
photos = photos[photos.photo_id.isin(row) & photos.photographer_id.isin(ph.photographer_id)].copy()
if os.environ.get("RANDOM_RANK"):   # random (not first-by-id) subset when only N_EVAL photos are scored
    photos = photos.sample(frac=1, random_state=int(os.environ.get("SEED", 0))).reset_index(drop=True)
photos["rank"] = photos.groupby("photographer_id").cumcount()
cnt = photos.groupby("photographer_id").size(); ph = ph[ph.photographer_id.map(cnt).fillna(0) >= N_PHOTOS].reset_index(drop=True)
photos = photos[photos.photographer_id.isin(ph.photographer_id)].reset_index(drop=True)
EXP = None
pidx = {p: i for i, p in enumerate(ph.photographer_id)}
photos["pi"] = photos.photographer_id.map(pidx); photos["er"] = photos.photo_id.map(row)
X_all = emb[photos.er.to_numpy()]; exp_id = photos.expertise_id.to_numpy(); y_ph = ph.y.to_numpy(); pi = photos.pi.to_numpy(); N_EVAL = int(os.environ.get("N_EVAL", N_PHOTOS)); in_eval = (photos["rank"] < N_EVAL).to_numpy()  # photographers stay the same (>=12 photos); only how many photos are scored changes
prof = pd.read_csv(D("profile.csv")).set_index("photographer_id").reindex(ph.photographer_id)
gear = pd.read_csv(D("gear.csv")); gear = gear[gear.photographer_id.isin(ph.photographer_id)].copy()
gear["name"] = (gear.brand.fillna("") + " " + gear.model.fillna("")).str.strip()

def num(col):
    """Numeric profile column: log-scaled, median-imputed (no labels involved) + a missing indicator."""
    v = np.log1p(prof[col].to_numpy(dtype=float)); miss = np.isnan(v)
    return np.column_stack([np.where(miss, np.nanmedian(v), v), miss.astype(float)])

def bert_embed(names, cache=D("gear_bert.npz")):
    """Exact camera/lens name -> 768-d BERT vector (mean-pooled). Cached; unseen models later embed the same way."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); have = dict(zip(z["names"], z["vecs"]))
    else: have = {}
    todo = [n for n in names if n not in have]
    if todo:
        from transformers import AutoModel, AutoTokenizer
        tok, bert = AutoTokenizer.from_pretrained("bert-base-uncased"), AutoModel.from_pretrained("bert-base-uncased").eval()
        for i in range(0, len(todo), 64):
            enc = tok(todo[i:i + 64], padding=True, return_tensors="pt")
            with torch.no_grad(): h = bert(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1); v = ((h * m).sum(1) / m.sum(1)).numpy()
            have.update(zip(todo[i:i + 64], v))
        np.savez(cache, names=np.array(list(have)), vecs=np.stack(list(have.values())))
    return have

def gear_features(kind, n_components=16):
    """Mean BERT vector of the photographer's declared cameras (or lenses), PCA-reduced (unsupervised) + missing flag."""
    g = gear[gear.kind == kind]; vec = bert_embed(sorted(g.name.unique()))
    agg = np.zeros((len(ph), 768), dtype=np.float32); has = np.zeros(len(ph))
    for pid, names in g.groupby("photographer_id").name:
        agg[pidx[pid]] = np.mean([vec[n] for n in names], axis=0); has[pidx[pid]] = 1
    from sklearn.decomposition import PCA
    z = PCA(n_components, random_state=0).fit_transform(agg)
    return np.column_stack([z, 1 - has])

meta_ext = np.hstack([num("years_been_photographer"), num("projects_payed_count"),
                      gear_features("camera"), gear_features("lens")])      # all self-reportable by a non-Kadro photographer
print(f"profile features: {meta_ext.shape[1]} (years, paid projects, BERT camera + lens PCA)", flush=True)
print(f"{len(ph)} photographers, {len(photos)} photos", flush=True)

def qwk(a, b): return cohen_kappa_score(a, b, weights="quadratic")
def to_class(p, c): return np.searchsorted(c, p)
def fit_cuts(p, y):
    cuts = np.arange(N_CLASSES - 1) + 0.5
    for _ in range(4):
        for k in range(len(cuts)):
            lo = cuts[k - 1] + 0.01 if k else -1; hi = cuts[k + 1] - 0.01 if k < len(cuts) - 1 else N_CLASSES
            g = np.linspace(lo, hi, 80); cuts[k] = g[np.argmax([qwk(y, to_class(p, np.r_[cuts[:k], t, cuts[k + 1:]])) for t in g])]
    return cuts

def train_mlp(tr_photo):
    """Fit stage-1 MLP on the given photo rows; returns a function photo-rows -> scores."""
    X = X_all[tr_photo]; mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd); yt = torch.tensor(y_ph[pi[tr_photo]], dtype=torch.float32)
    w = 1.0 / np.sqrt(np.bincount(y_ph[pi[tr_photo]], minlength=N_CLASSES)); wt = torch.tensor(w[y_ph[pi[tr_photo]]], dtype=torch.float32)
    if os.environ.get("EQUAL_PH"):   # every photographer counts equally, however many photos they have
        e = 1.0 / np.bincount(pi)[pi[tr_photo]]; wt = wt * torch.tensor(e / e.mean(), dtype=torch.float32)
    net = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05); n = len(Xt)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * ((n + 511) // 512))
    for _ in range(EPOCHS):
        net.train()
        for b in torch.randperm(n).split(512):
            loss = (wt[b] * (net(Xt[b]).squeeze(1) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    def score(rows):
        with torch.no_grad(): return net(torch.tensor((X_all[rows] - mu) / sd)).squeeze(1).numpy()
    return score

COLS = ["mean", "std", "max", "min", "tophalf", "exp_max", "exp_min", "exp_std", "exp_best2"]
def agg(scores, rows):
    """Per-photographer features from photo scores (only the first N_EVAL photos of each).
    exp_*: statistics of the mean score per expertise (shoot type), so 'strong in one, weak in another' is visible."""
    keep = in_eval[rows]
    df = pd.DataFrame({"pi": pi[rows], "e": exp_id[rows], "s": scores})[keep]
    g = df.groupby("pi").s
    top = g.apply(lambda s: np.sort(s.to_numpy())[len(s) // 2:].mean())
    em = df.groupby(["pi", "e"]).s.mean().groupby("pi")
    best2 = em.apply(lambda s: np.sort(s.to_numpy())[-2:].mean())
    return pd.DataFrame({"mean": g.mean(), "std": g.std().fillna(0), "max": g.max(), "min": g.min(), "tophalf": top,
                         "exp_max": em.max(), "exp_min": em.min(), "exp_std": em.std().fillna(0), "exp_best2": best2})

def stage1_scores(train_ph, test_ph, inner=5):
    """Cross-fitted scores for train photographers + full-model scores for test photographers."""
    tr_rows = np.where(np.isin(pi, train_ph))[0]; te_rows = np.where(np.isin(pi, test_ph))[0]
    oof = pd.DataFrame(index=train_ph, columns=COLS, dtype=float)
    for a, b in KFold(inner, shuffle=True, random_state=0).split(train_ph):
        f = train_mlp(np.where(np.isin(pi, train_ph[a]))[0]); r = np.where(np.isin(pi, train_ph[b]))[0]
        oof.loc[train_ph[b]] = agg(f(r), r).loc[train_ph[b]].to_numpy()
    f = train_mlp(tr_rows)
    return oof, agg(f(te_rows), te_rows).loc[test_ph]

def ridge(): return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9)))
def evaluate(train_ph, test_ph):
    t0 = time.time(); oof, te = stage1_scores(train_ph, test_ph); print(f"  stage-1 done in {time.time() - t0:.0f}s", flush=True)
    ytr, res = y_ph[train_ph], {}
    S, SE = COLS[:5], COLS
    for name, cols, M in [("profile only (no photos)", [], meta_ext),
                          ("photo MLP, mean score", ["mean"], None),
                          ("photo MLP, score stats", S, None),
                          ("mean score + profile (count-free)", ["mean"], meta_ext),
                          ("photos + profile", S, meta_ext),
                          ("photos(expertise-aware)", SE, None),
                          ("photos(expertise-aware) + profile", SE, meta_ext)]:
        Xtr, Xte = oof[cols].to_numpy(), te[cols].to_numpy()
        if M is not None: Xtr, Xte = np.hstack([Xtr, M[train_ph]]), np.hstack([Xte, M[test_ph]])
        o2 = np.zeros(len(ytr))
        for a, b in KFold(5, shuffle=True, random_state=2).split(Xtr): o2[b] = ridge().fit(Xtr[a], ytr[a]).predict(Xtr[b])
        res[name] = to_class(ridge().fit(Xtr, ytr).predict(Xte), fit_cuts(o2, ytr))
    return res

def report(y, p):
    boot = [qwk(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
    return dict(n=len(y), qwk=round(qwk(y, p), 3), qwk_ci=[round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
                mae_steps=round(float(np.abs(y - p).mean()), 3), acc=round(float((y == p).mean()), 3), within1=round(float((np.abs(y - p) <= 1).mean()), 3))

allp = np.arange(len(ph)); train = (ph.signup_at < SPLIT_DATE).to_numpy(); out = {}
print("temporal:", flush=True)
res = evaluate(allp[train], allp[~train]); yte = y_ph[~train]
out["temporal"] = {k: report(yte, v) for k, v in res.items()}
print("confusion (photos + profile):\n", confusion_matrix(yte, res["photos + profile"], labels=range(N_CLASSES)))
if "--cv" in sys.argv:
    print("cv:", flush=True); pred = {}
    for a, b in KFold(5, shuffle=True, random_state=1).split(allp):
        for k, v in evaluate(allp[a], allp[b]).items(): pred.setdefault(k, np.zeros(len(allp), int))[b] = v
    out["cv"] = {k: report(y_ph, v) for k, v in pred.items()}
for s in out: print(f"\n== {s} ==\n" + pd.DataFrame(out[s]).T.to_string())
json.dump(out, open(D(f"results_photo_level_{os.environ.get('EMB', 'embeddings').replace(',', '+')}_n{N_EVAL}{os.environ.get('TAG', '')}_seed{SEED}.json"), "w"), indent=1)
