# results/

Aggregate evaluation metrics only (quadratic weighted kappa, exact-grade accuracy, within-one-grade accuracy, mean error in grade
steps, bootstrap interval, n). No per-photographer or per-photo information.

| files | protocol | status |
|---|---|---|
| `results_photo_level_embeddings_clip+embeddings_clip_rest_n{1,3,5,10,20,100000}_clean_seed{0,1,2}.json` | **final, count-free**: every graded photographer with >=1 photo, k random photos scored (`100000` = all), temporal split, 3 seeds. Variants: profile only, photos only (`mean score`), `mean score + profile (count-free)`, plus the older `photos + profile` variant that **leaks portfolio size** (kept for contrast) | authoritative |
| `..._n{1,10,100000}_cv_seed0.json` | same, 5-fold cross-validation over all 2,977 graded photographers, seed 0 | authoritative (CV) |
| `..._n100000_exp_seed0.json`, `..._n40_exp_seed0.json` | leakage control: full portfolios vs portfolios capped at 40 photos, with best/worst/spread statistics | explains why stats were dropped |
| `results_{dino,clip}_balanced24_seed{0,1,2}.json` | backbone comparison, earlier protocol (balanced sample of <=24 photos, photographers with >=12 photos) | historical |
| `results_baseline.json` | first baseline: mean DINOv2 embedding + ridge, 12 photos | historical |
