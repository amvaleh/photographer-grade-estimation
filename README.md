# Estimating expert photographer grades from portfolio photos

Can a model reproduce the grade that human experts assign to a photographer, from nothing but a few portfolio photos and a few
self-reportable facts? This repository is the research behind a model that was **deployed in production** at
[Kadro](https://kadro.co), an Iranian marketplace for booking photographers, where administrators grade every photographer
(a 5-level ordinal scale that also sets the photographer's price tier).

> **No data and no model are released here.** The training data (photographers' portfolios and profiles) belongs to a
> commercial platform and its users, and the trained model is a business asset. This repository contains the code,
> queries, evaluation protocol and aggregate results so the method can be inspected and reproduced on similar data.

## Headline results (honest, count-free, temporal hold-out)

Train on photographers who joined before 2023, test on those who joined from 2023 on (n = 385, never seen in training).
QWK = quadratic weighted kappa (0 = chance, 1 = perfect agreement, penalises distant errors more). Mean of 3 seeds.

| photos scored | photos only | photos + profile | exact grade | within 1 grade | average miss |
|---|---|---|---|---|---|
| 1 | 0.32 | 0.44 | 39 % | 83 % | 0.83 |
| 3 | 0.44 | 0.47 | 45 % | 86 % | 0.73 |
| 5 | 0.48 | 0.50 | 42 % | 86 % | 0.76 |
| 10 | 0.52 | 0.50 | 44 % | 88 % | 0.71 |
| 20 | 0.53 | 0.52 | 44 % | 87 % | 0.71 |
| all | 0.50 | 0.52 | 45 % | 85 % | 0.72 |

A profile-only model (no photos) scores 0.38. Under 5-fold cross-validation over all 2,977 graded photographers (same period, so easier)
the numbers are about 0.05 higher: QWK 0.50 with one photo, 0.57 with ten, 0.58 with all photos; exact grade 45 % to 55 %.
Raw numbers: [`results/`](results/).

**Reading it plainly:** the model clearly captures signal (QWK around 0.5, ~45 % exact, ~87 % within one grade) but it is a
moderately accurate advisor, not a replacement for the experts. One photo gives a rough guess; the estimate stabilises at roughly 10 to 20 photos.

## Method

```
photos --> re-create the platform's upload compression --> frozen CLIP ViT-B/16 (1536-d) --> small MLP: one quality score per photo
       --> MEAN over the photographer's photos
profile (years as photographer, paid projects, camera + lens names) --> BERT(exact name) --> PCA16
[mean photo score | profile] --> ridge --> ordinal cut-points tuned for QWK --> grade 0 / 1 / 2 / 3 / 4+
grade --> empirical probabilities (out-of-fold confusion matrix, one calibrated stage-2 per photo-count bucket)
```

- **Labels:** the grade administrators assigned (merged to 5 ordinal classes; the top three are rare). Customer ratings were rejected as a target:
  they sit at 9.3 to 10 out of 10 and, if anything, fall as the grade rises.
- **Photo-level learning:** each photo is trained against its photographer's grade (weak labels, a multiple-instance setup) and scores are then averaged.
  This beat averaging embeddings first (+0.02 QWK).
- **Photographers weigh equally** in training, so portfolios of hundreds of photos do not dominate.
- **Gear as text:** camera and lens names are embedded with BERT so a model never seen in training still lands near similar ones.
- **Training/serving parity:** `src/grader_core.py` builds the features for both training and the serving API, and
  `src/validate_serving_path.py` checks that the deployed path reproduces the offline numbers on held-out photographers (QWK 0.45 / 0.46 / 0.53 / 0.48
  at 1 / 3 / 5 / 10 photos vs 0.44 / 0.47 / 0.50 / 0.50 offline).

## What went wrong on the way (and was fixed)

These are kept visible on purpose.

1. **Portfolio size leaked the label.** Higher grades have much bigger portfolios (median photos per grade rises from about 30 to over 600), so a model that sees every photo can read the grade from the count,
   mechanically, through best/worst/spread statistics. Photo count alone, with no image information, scores QWK 0.33. A first "full portfolio" run reported 0.57 on the temporal test;
   capping the number of photos scored dropped it to 0.53 and the count-free model gives 0.52. The final design uses only the *mean* photo score and a random k photos, so a portfolio of one photo is judged on equal terms.
   An intermediate table of mine mislabelled the leaky variant as mean-only; it was caught, rerun (`..._clean_...` files) and corrected.
2. **Backbone:** CLIP beat DINOv2 on QWK (+0.02 to +0.06) but not on exact accuracy.
3. **Expertise-aware statistics** (per shoot-type score means) helped only while portfolios were thin; they added nothing with full portfolios.
4. **Customer-feedback score** added almost nothing: only 16 % of graded photographers have any feedback, and it does not exist for outside photographers anyway, so it was dropped.
5. **Dirty fields** (impossible birthdays, negative years, 10^9 project counts) are nulled in SQL and handled with missing-value indicators.

## Limitations

- **The ceiling is unknown.** Inter-admin agreement has not been measured, so it is unclear how close ~0.5 QWK is to what is achievable.
- **No external validation.** All photos came through one platform's uploader and one market; accuracy on photographers outside it is untested.
- **The temporal test set is small** (n = 385, bootstrap interval about +/-0.09 QWK); differences below ~0.03 should not be trusted.
- **Labels are one organisation's judgement** (and a grade also encodes price tier), not a ground truth of photographic quality.
- Photos judge portfolios, not service quality, punctuality or reliability, and the tool is advisory only.

## Ethics and data statement

- Photographers' images, profiles and identifiers are not included and must never be committed (`.gitignore` blocks `data/`, `models/`, CSV/NPZ/PT files).
- Only `photographer_id` (an internal integer) was used as a key during experiments; no names, contact details or national ids were exported. Gender and city were exported to the local workspace but are not used by the final model.
- A system that scores people can cause harm if misused. The deployed service is advisory and does not rank named individuals. Since October 2026 it stores the photos and results that users submit, behind a visible notice on the page, for review by the operator's staff; they are not published, are not part of this repository, and can be deleted on request.

## Reproducing on your own data

You need a labelled collection of portfolios. The queries in `sql/` define the dataset for this platform's schema (adapt table names to yours).

```bash
pip install -r requirements-research.txt
SSH_HOST=<alias> DB_NAME=<db> scripts/export_manifest.sh          # manifests -> data/
SSH_HOST=<alias> UPLOADS_DIR=<dir> scripts/fetch_images.sh data/all_missing.tsv
python src/embed.py --model openai/clip-vit-base-patch16 --manifest data/photos_all.csv --out embeddings_clip
MIN_PHOTOS=1 RANDOM_RANK=1 EQUAL_PH=1 EMB=embeddings_clip PHOTOS=photos_all.csv N_EVAL=10 SEED=0 python src/train_photo_level.py --cv
python src/export_model.py --out models/final                     # production model (not released)
python src/export_model.py --out models/holdout --train-before 2023-01-01 && python src/validate_serving_path.py
```

## Layout

| path | what |
|---|---|
| `sql/` | read-only queries that define the dataset (manifests, profile fields, gear) |
| `scripts/` | export and image-fetch scripts (hosts and paths come from environment variables) |
| `src/embed.py` | frozen DINOv2 / CLIP embeddings, resumable in shards |
| `src/train_baseline.py`, `src/train_photo_level.py` | experiments and the evaluation protocol (temporal split, CV, bootstrap, seeds) |
| `src/grader_core.py`, `src/export_model.py` | the shared feature/inference code and the production model export |
| `src/validate_serving_path.py` | end-to-end check of the serving path against the offline results |
| `results/` | aggregate metrics (see its README) |

The commercial serving layer (API, web page, Docker, deployment) is in a separate private repository.

## License

The code in this repository is released under the [MIT License](LICENSE). The license covers the code only: the training data and the trained model
are not part of this repository and are not licensed or released.
