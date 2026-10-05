"""Embed every fetched photo with a frozen DINOv2 backbone.

Per photo we store [CLS | mean of patch tokens] (2 x hidden_dim) as float16.
Resumable: output goes to data/embeddings/shard_XXXXX.npz every SHARD photos; finished shards are skipped.

Usage: python src/embed.py [--manifest data/photos.csv] [--limit N] [--model facebook/dinov2-base]
"""
import argparse, glob, os, time
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import AutoModel

SHARD = 4096
ROOT = os.path.join(os.path.dirname(__file__), "..")
IMG_DIR = os.path.join(ROOT, "data", "images")

# Standard resize(256)+center-crop(224). Crops the edges of wide photos: noted as a limitation in the write-up.
STATS = {"dino": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), "clip": ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))}
def make_tf(kind):
    return T.Compose([T.Resize(224 if kind == "clip" else 256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                      T.ToTensor(), T.Normalize(*STATS[kind])])


def find_file(photo_id):
    hits = glob.glob(os.path.join(IMG_DIR, str(photo_id), "medium_*"))
    return hits[0] if hits else None


class Photos(Dataset):
    def __init__(self, ids, tf): self.ids, self.tf = ids, tf
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        path = find_file(self.ids[i])
        try:
            return self.tf(Image.open(path).convert("RGB")), True
        except Exception:
            return torch.zeros(3, 224, 224), False  # unreadable / missing: flagged and dropped later


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "photos.csv"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="embeddings", help="subfolder of data/")
    ap.add_argument("--skip", help="subfolder of data/ whose already-embedded photo_ids are skipped")
    ap.add_argument("--graded-only", action="store_true", help="only photos of photographers that have a grade 0..6")
    a = ap.parse_args()

    man = pd.read_csv(a.manifest, usecols=["photo_id", "photographer_id"])
    if a.graded_only:
        g = pd.read_csv(os.path.join(ROOT, "data", "photographers.csv"), usecols=["photographer_id", "grade"])
        man = man[man.photographer_id.isin(g[g.grade.isin(range(0, 7))].photographer_id)]
    ids = man.photo_id.tolist()
    if a.skip:
        done_ids = set(np.concatenate([np.load(f)["photo_id"] for f in glob.glob(os.path.join(ROOT, "data", a.skip, "shard_*.npz"))]).tolist())
        ids = [i for i in ids if i not in done_ids]
    ids = [i for i in ids if os.path.isdir(os.path.join(IMG_DIR, str(i)))]  # only what was downloaded
    if a.limit: ids = ids[: a.limit]
    OUT_DIR = os.path.join(ROOT, "data", a.out); os.makedirs(OUT_DIR, exist_ok=True)
    kind = "clip" if "clip" in a.model.lower() else "dino"; tf = make_tf(kind)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if kind == "clip":
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained(a.model).to(dev).eval()
    else:
        model = AutoModel.from_pretrained(a.model).to(dev).eval()
    print(f"{len(ids)} photos, device={dev}, model={a.model}", flush=True)

    t0, done = time.time(), 0
    for s in range(0, len(ids), SHARD):
        out = os.path.join(OUT_DIR, f"shard_{s // SHARD:05d}.npz")
        chunk = ids[s : s + SHARD]
        if os.path.exists(out):
            done += len(chunk); continue
        feats, ok = [], []
        for x, good in DataLoader(Photos(chunk, tf), batch_size=a.batch, num_workers=4):
            with torch.no_grad():
                o = model(pixel_values=x.to(dev))
                h = o.last_hidden_state
                # DINO: [CLS | mean patch]. CLIP: [post-layernorm pooled CLS | mean patch] (same 2 x hidden layout).
                f = torch.cat([o.pooler_output if kind == "clip" else h[:, 0], h[:, 1:].mean(1)], dim=1)
            feats.append(f.float().cpu().numpy().astype(np.float16)); ok.append(good.numpy())
        ok = np.concatenate(ok)
        np.savez(out, photo_id=np.array(chunk)[ok], emb=np.concatenate(feats)[ok])
        done += len(chunk)
        print(f"{done}/{len(ids)}  {done / (time.time() - t0):.1f} img/s", flush=True)


if __name__ == "__main__":
    main()
