# Data Loading Pipeline — File Organization

## Where this goes in your project

This whole `ml_training/` folder is **separate from your Django app** — it is
offline training code, not something the live web server imports. Put it
next to your Django project root, like this:

```
agrosmart_project/              <- your existing Django project root
├── manage.py
├── disease_detection/          <- your existing Django app
│   ├── views.py
│   ├── models.py
│   └── ml/
│       └── disease_loader.py   <- existing inference loader (unchanged for now)
│
└── ml_training/                <- NEW — everything from this phase goes here
    ├── dataset/                 <- your finalized 7-crop, 29-class image folders
    ├── config.py
    ├── data_loader.py
    ├── label_encoder.py
    ├── preprocessing.py
    ├── dataset_builder.py
    ├── batch_generator.py
    └── build_dataset.py
```

Why separate: training code touches thousands of image files, runs for a
long time, and has no business being imported by `views.py` or running
inside a request/response cycle. Only once a model is trained (later
phases) does a *lightweight* prediction module get placed inside
`disease_detection/ml/` for the Django app to actually use — that's Phase 9,
not this one.

## What each file does

| File | Requirement it satisfies | Purpose |
|---|---|---|
| `config.py` | — | Central place for paths, image size, split ratios, seed |
| `data_loader.py` | 1. Scan dataset folders | Walks `dataset/<Crop>/<Class>/` and lists every image path |
| `label_encoder.py` | 2. Assign numeric labels | Maps each unique (crop, class) pair to an integer 0–28 |
| `preprocessing.py` | 3. Load images<br>4. Resize<br>5. Normalize | Opens one image, resizes it, scales pixels to [0, 1] |
| `dataset_builder.py` | 6. Encode labels<br>7. Shuffle<br>8. Split | Loads all images into arrays, one-hot encodes, shuffles, splits into train/val/test |
| `batch_generator.py` | 9. Mini-batches | Yields shuffled `(X_batch, y_batch)` pairs for training loops |
| `build_dataset.py` | — | Orchestrator: runs everything above once and saves the result |

## How to run it

1. Place your verified `dataset/` folder inside `ml_training/` (or edit
   `DATASET_ROOT` in `config.py` to point elsewhere).
2. Install Pillow if you haven't already: `pip install Pillow`
3. Run:
   ```bash
   cd ml_training
   python build_dataset.py
   ```
4. This produces two files you'll use in the next phase (CNN training):
   - `prepared_dataset.npz` — the preprocessed, split, shuffled arrays
   - `label_map.json` — maps each integer label back to its "Crop__Class" name

Loading ~14,800 images at 128×128 will take some time and hold a fair
amount of RAM (roughly 14,800 × 128 × 128 × 3 × 4 bytes ≈ 2.9 GB as
float32). If your machine struggles with that, tell me and I'll adapt
`dataset_builder.py` to process and save the data in chunks instead of
holding it all in memory at once.

## A note for your defense

Pillow is used in `preprocessing.py` purely to decode image files from disk
(open a JPEG/PNG and get pixel values) — there is no way to do that in
plain NumPy, since NumPy has no image codec. It is not used for anything
resembling a machine learning computation. Everything from that point on —
normalization, label encoding, shuffling, splitting, batching, and (in the
next phase) the CNN itself — is implemented with NumPy only. This is the
same distinction as using a CSV reader to load tabular data before running
your own from-scratch algorithm on it.
