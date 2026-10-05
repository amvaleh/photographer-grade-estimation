"""Shared inference logic for the photographer grader.

Used by BOTH src/export_model.py (to build the training features) and service/app.py (to serve), so the two
cannot drift apart. Pipeline for one request:

  photos -> kadro_normalize (re-create Kadro's upload compression) -> CLIP ViT-B/16 -> 1536-d
         -> standardise -> stage-1 MLP -> one quality score per photo -> MEAN over the photos
  profile -> years, paid projects, BERT(camera names), BERT(lens names) -> 38 numbers
  [mean score | profile] -> stage-2 ridge (one per photo-count bucket) -> ordinal cut-points -> grade 0..4+
  grade -> empirical probabilities (out-of-fold confusion matrix) -> likely range

Only the MEAN photo score is used, never best/worst/spread, and the photo count only selects which calibrated
stage-2 model to use (fewer photos = a noisier mean, and the stage-2 model for that bucket knows it).
"""
import io, json, os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms as T

CLIP_ID = "openai/clip-vit-base-patch16"
BERT_ID = "bert-base-uncased"
N_CLASSES = 5
KS = [1, 3, 5, 10, 20, 40]          # photo-count buckets the stage-2 models are calibrated for
MAX_PHOTOS = 40
YEARS_RANGE, PROJECTS_RANGE = (0, 60), (0, 5000)   # same validity windows the training SQL applied
GRADE_LABELS = ["نوآموز", "⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐ و بالاتر"]
GRADE_LABELS_EN = ["beginner", "1 star", "2 stars", "3 stars", "4+ stars"]


def pick_device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def make_net(dim):
    """Stage-1 photo scorer. Architecture must match what export_model.py trains."""
    return nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.3),
                         nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))


# ---------------------------------------------------------------- photos
def kadro_normalize(img: Image.Image) -> Image.Image:
    """Re-create what Kadro's uploader did to every training photo, so external photos reach CLIP with the same
    compression artefacts: fit inside 1360x970 + JPEG q65 (large), then fit inside 800x800 as WebP (medium_webp)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((1360, 970), Image.LANCZOS)                      # downscale only
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=65, optimize=True); buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.thumbnail((800, 800), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "WEBP", quality=75); buf.seek(0)
    return Image.open(buf).convert("RGB")


_CLIP_TF = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))])


class ClipEmbedder:
    """[post-layernorm pooled CLS | mean of patch tokens] = 1536 numbers per photo, float16-rounded like the training shards."""
    def __init__(self, device=None):
        from transformers import CLIPVisionModel
        self.device = device or pick_device()
        self.model = CLIPVisionModel.from_pretrained(CLIP_ID).to(self.device).eval()

    def embed(self, images, normalize=True, batch=16):
        out = []
        for i in range(0, len(images), batch):
            chunk = [kadro_normalize(im) if normalize else im.convert("RGB") for im in images[i:i + batch]]
            x = torch.stack([_CLIP_TF(im) for im in chunk]).to(self.device)
            with torch.no_grad():
                o = self.model(pixel_values=x)
                f = torch.cat([o.pooler_output, o.last_hidden_state[:, 1:].mean(1)], dim=1)
            out.append(f.float().cpu().numpy().astype(np.float16).astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 1536), np.float32)


# ---------------------------------------------------------------- profile
def clean_name(s):
    return " ".join(str(s).split())


class GearEmbedder:
    """Exact camera/lens name -> 768-d BERT vector (mean-pooled). Cached; names never seen in training are
    embedded on the fly, which is the point of using BERT instead of a one-hot list."""
    def __init__(self, cache_path, device="cpu"):
        self.cache_path, self.device, self._bert = cache_path, device, None
        self.vecs = {}
        if cache_path and os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=True)
            self.vecs = dict(zip(map(str, z["names"]), z["vecs"]))

    def _load(self):
        if self._bert is None:
            from transformers import AutoModel, AutoTokenizer
            self._bert = (AutoTokenizer.from_pretrained(BERT_ID), AutoModel.from_pretrained(BERT_ID).to(self.device).eval())
        return self._bert

    def embed_names(self, names):
        todo = sorted({clean_name(n) for n in names if clean_name(n) and clean_name(n) not in self.vecs})
        if todo:
            tok, bert = self._load()
            for i in range(0, len(todo), 64):
                enc = tok(todo[i:i + 64], padding=True, truncation=True, max_length=32, return_tensors="pt").to(self.device)
                with torch.no_grad(): h = bert(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1); v = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
                self.vecs.update(zip(todo[i:i + 64], v))
        return {clean_name(n): self.vecs[clean_name(n)] for n in names if clean_name(n)}

    def save(self, path):
        np.savez(path, names=np.array(list(self.vecs)), vecs=np.stack(list(self.vecs.values())))


def gear_aggregate(names, gear: GearEmbedder):
    """Mean BERT vector of a photographer's declared items (zeros when none declared, as in training)."""
    names = [clean_name(n) for n in names if clean_name(n)]
    if not names:
        return np.zeros(768, np.float32), 0.0
    v = gear.embed_names(names)
    return np.mean([v[n] for n in names], axis=0).astype(np.float32), 1.0


def _num(value, lo, hi, median):
    """log1p + missing indicator; out-of-range or absent -> median-imputed + flag (matches the training SQL cleaning)."""
    ok = value is not None and not (isinstance(value, float) and np.isnan(value)) and lo <= float(value) <= hi
    return [float(np.log1p(float(value))) if ok else float(median), 0.0 if ok else 1.0]


def profile_features(pre, years, projects, cameras, lenses, gear: GearEmbedder):
    """38 numbers: years(2) + paid projects(2) + camera PCA16 + missing + lens PCA16 + missing.
    `pre` holds the fitted preprocessing: medians and the two PCAs."""
    f = _num(years, *YEARS_RANGE, pre["median_years"]) + _num(projects, *PROJECTS_RANGE, pre["median_projects"])
    for kind, names in (("camera", cameras), ("lens", lenses)):
        agg, has = gear_aggregate(names, gear)
        z = (agg - pre[f"pca_{kind}_mean"]) @ pre[f"pca_{kind}_components"].T
        f += list(z) + [1.0 - has]
    return np.array(f, np.float32)


# ---------------------------------------------------------------- calibrated grader
def likely_range(probs, mass=0.8):
    """Smallest run of neighbouring grades, grown from the most likely one, holding >= `mass` of the probability."""
    lo = hi = int(np.argmax(probs)); total = probs[lo]
    while total < mass and (lo > 0 or hi < len(probs) - 1):
        left = probs[lo - 1] if lo > 0 else -1; right = probs[hi + 1] if hi < len(probs) - 1 else -1
        if left >= right: lo -= 1; total += left
        else: hi += 1; total += right
    return lo, hi


def reliability(n_photos):
    return "low" if n_photos < 5 else ("medium" if n_photos < 10 else "good")


class Grader:
    def __init__(self, artifacts_dir, device=None, load_clip=True):
        self.dir = artifacts_dir
        self.device = device or pick_device()
        self.cfg = json.load(open(os.path.join(artifacts_dir, "config.json")))
        s1 = torch.load(os.path.join(artifacts_dir, "stage1.pt"), map_location="cpu", weights_only=False)
        self.mu, self.sd = s1["mu"], s1["sd"]
        self.net = make_net(len(self.mu)); self.net.load_state_dict(s1["state"]); self.net.eval()
        z = np.load(os.path.join(artifacts_dir, "stage2.npz")); self.s2 = {k: z[k] for k in z.files}
        pre = np.load(os.path.join(artifacts_dir, "preprocess.npz"))
        self.pre = {k: pre[k] for k in pre.files}
        self.pre["median_years"] = float(self.pre["median_years"]); self.pre["median_projects"] = float(self.pre["median_projects"])
        self.gear = GearEmbedder(os.path.join(artifacts_dir, "gear_bert.npz"), device="cpu")
        self.clip = ClipEmbedder(self.device) if load_clip else None

    def score_embeddings(self, emb):
        with torch.no_grad():
            return self.net(torch.tensor((emb - self.mu) / self.sd, dtype=torch.float32)).squeeze(1).numpy()

    def from_scores(self, scores, years=None, projects=None, cameras=(), lenses=()):
        n = len(scores)
        if n < 1: raise ValueError("at least one photo is required")
        K = max(k for k in KS if k <= n)
        x = np.concatenate([[float(np.mean(scores))],
                            profile_features(self.pre, years, projects, cameras, lenses, self.gear)])
        z = (x - self.s2[f"b{K}_mean"]) / self.s2[f"b{K}_scale"]
        raw = float(z @ self.s2[f"b{K}_coef"] + self.s2[f"b{K}_intercept"])
        cls = int(np.searchsorted(self.s2[f"b{K}_cuts"], raw))
        probs = self.s2[f"b{K}_calibration"][cls]
        lo, hi = likely_range(probs)
        return {"estimated_grade": cls, "grade_label": GRADE_LABELS[cls], "grade_label_en": GRADE_LABELS_EN[cls],
                "probabilities": {GRADE_LABELS_EN[i]: round(float(p), 3) for i, p in enumerate(probs)},
                "likely_range": {"from": lo, "to": hi, "from_label": GRADE_LABELS[lo], "to_label": GRADE_LABELS[hi]},
                "photos_used": n, "reliability": reliability(n), "raw_score": round(raw, 3),
                "mean_photo_score": round(float(np.mean(scores)), 3), "model_version": self.cfg["version"]}

    def estimate(self, images, years=None, projects=None, cameras=(), lenses=()):
        images = images[:MAX_PHOTOS]
        return self.from_scores(self.score_embeddings(self.clip.embed(images)), years, projects, cameras, lenses)
