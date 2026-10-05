"""End-to-end validation of the serving path against the offline evaluation.

Loads the HOLD-OUT artifacts (trained only on photographers who joined before 2023-01-01), pushes the real photos
of photographers who joined later through the exact code path the API uses (preprocess -> CLIP -> MLP -> stage 2),
and scores the result against the admins' grades. If this lands near the offline temporal numbers (QWK ~0.50-0.52 at
10 photos) the service is faithful to the research pipeline.

  python src/validate_serving_path.py --artifacts models/holdout
"""
import argparse, glob, os, sys, time
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import grader_core as gc

D = lambda *p: os.path.join(ROOT, "data", *p)
ap = argparse.ArgumentParser()
ap.add_argument("--artifacts", default=os.path.join(ROOT, "models", "holdout"))
ap.add_argument("--after", default="2023-01-01"); ap.add_argument("--pool", type=int, default=10)
ap.add_argument("--max-photographers", type=int, default=0)
a = ap.parse_args()

g = gc.Grader(a.artifacts)
ph = pd.read_csv(D("photographers.csv"), parse_dates=["signup_at"])
ph = ph[ph.grade.isin(range(7)) & (ph.signup_at >= a.after)].copy(); ph["y"] = ph.grade.clip(upper=4).astype(int)
photos = pd.read_csv(D("photos_all.csv")); photos = photos[photos.photographer_id.isin(ph.photographer_id)]
prof = pd.read_csv(D("profile.csv")).set_index("photographer_id")
gear = pd.read_csv(D("gear.csv")); gear["name"] = (gear.brand.fillna("") + " " + gear.model.fillna("")).str.strip()

def file_of(pid_):
    hits = glob.glob(D("images", str(pid_), "medium_*"))
    return hits[0] if hits else None

rng = np.random.default_rng(0)
pool = {}
for pid, grp in photos.groupby("photographer_id"):
    ids = rng.permutation(grp.photo_id.to_numpy())
    files = [f for f in (file_of(i) for i in ids[: a.pool * 2]) if f][: a.pool]
    if files: pool[pid] = files
ph = ph[ph.photographer_id.isin(pool)]
if a.max_photographers: ph = ph.head(a.max_photographers)
print(f"{len(ph)} held-out photographers (joined >= {a.after}), up to {a.pool} photos each", flush=True)

# ---- 1) embedding fidelity: service path vs the stored offline embeddings
shards = [np.load(f) for d in ("embeddings_clip", "embeddings_clip_rest") for f in sorted(glob.glob(D(d, "shard_*.npz")))]
stored = dict(zip(np.concatenate([s["photo_id"] for s in shards]).tolist(), np.concatenate([s["emb"] for s in shards])))
sample = [f for fs in list(pool.values())[:60] for f in fs[:3]][:150]
pid_of = lambda f: int(os.path.basename(os.path.dirname(f)))
imgs = [Image.open(f).convert("RGB") for f in sample]
cos = lambda u, v: float((u * v).sum(1).mean())
nz = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
ref = nz(np.stack([stored[pid_of(f)] for f in sample]).astype(np.float32))
print(f"embedding cosine vs stored: raw-as-is {cos(nz(g.clip.embed(imgs, normalize=False)), ref):.4f}   "
      f"with kadro_normalize {cos(nz(g.clip.embed(imgs, normalize=True)), ref):.4f}", flush=True)

# ---- 2) full-path predictions
t0 = time.time(); scores = {}
for n, pid in enumerate(ph.photographer_id):
    scores[pid] = g.score_embeddings(g.clip.embed([Image.open(f) for f in pool[pid]], normalize=True))
    if (n + 1) % 100 == 0: print(f"  embedded {n + 1}/{len(ph)} ({time.time() - t0:.0f}s)", flush=True)

def names(kind, pid): return gear[(gear.photographer_id == pid) & (gear.kind == kind)].name.tolist()
rows = []
for K in (1, 3, 5, 10):
    pred = []
    for pid in ph.photographer_id:
        s = scores[pid]; sub = np.random.default_rng(K * 7 + int(pid)).permutation(len(s))[:K]
        p = prof.loc[pid] if pid in prof.index else None
        out = g.from_scores(s[sub], None if p is None else p.years_been_photographer, None if p is None else p.projects_payed_count,
                            names("camera", pid), names("lens", pid))
        pred.append(out["estimated_grade"])
    pred, y = np.array(pred), ph.y.to_numpy()
    rows.append({"photos": K, "qwk": round(cohen_kappa_score(y, pred, weights="quadratic"), 3), "exact_acc": round(float((pred == y).mean()), 3),
                 "within_1": round(float((np.abs(pred - y) <= 1).mean()), 3), "avg_miss": round(float(np.abs(pred - y).mean()), 3), "n": len(y)})
print(pd.DataFrame(rows).to_string(index=False))
print("offline reference (temporal, clean, 3 seeds): QWK 1->0.44, 3->0.47, 5->0.50, 10->0.50; exact acc 39-45%")
