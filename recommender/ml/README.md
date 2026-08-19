# Machine Learning Training Pipeline — `recommender/ml/`

This folder contains the trained models and the scripts that build them for the
**AgroSmart** crop-recommendation and disease-detection system.

---

## 1. Crop Recommendation Model (`Crop_recommendation_RF.pkl`)

**Algorithm:** Random Forest Classifier, implemented **from scratch** in
`random_forest.py` / `decision_tree.py` (no scikit-learn at inference time).
The pickle exposes sklearn-compatible `.predict()`, `.predict_proba()`, and
`.classes_` so `loader.py` works unchanged.
**Inputs (7 features):** `N, P, K, temperature, humidity, ph, rainfall`
**Output:** one of 13 Nepal crops (barley, buckwheat, cardamom, ginger, lentil,
maize, millet, mustard, potato, rice, sugarcane, tea, wheat) — as present in
`agrosmart_crop_dataset.csv`.

### Dataset
The live model is trained on **`agrosmart_crop_dataset.csv`**: **1950 rows**
across **13 Nepal crops** (barley, buckwheat, cardamom, ginger, lentil, maize,
millet, mustard, potato, rice, sugarcane, tea, wheat), with 7 features each
(`N, P, K, temperature, humidity, ph, rainfall`). Crop N/P/K/temp/humidity/ph/
rainfall ranges are anchored to real, cited Nepal-specific (and regional) agronomy
studies in `crop_ranges.py` (see `references.md` for sources).

> Note: an earlier version merged a real Kaggle set with a synthetic zoned set
> (15 crops, 1860 rows). The current `agrosmart_crop_dataset.csv` supersedes
> that — use the file present in this folder.

### Algorithm comparison (why Random Forest)
We compared 5 classifiers on the merged dataset with the same train/test
split (`compare_algorithms.py`):

| Algorithm | Test Accuracy |
|---|---:|
| Random Forest | 0.9328 |
| Decision Tree | 0.9113 |
| SVM (RBF) | 0.9409 |
| K-Nearest Neighbors | 0.9409 |
| Logistic Regression | 0.9113 |

All five scored 91–94%, so accuracy was comparable. We chose **Random
Forest** because it provides a **native probability output** (powers the
confidence % and top-3 suggestions via `model.predict_proba` in `loader.py`),
needs **no feature scaling** (unlike SVM/KNN), and as an ensemble is **more
robust than a single Decision Tree**.

### How to retrain
```bash
# from recommender/ml/
.\.venv\Scripts\python.exe train_and_save.py
```
This reads `agrosmart_crop_dataset.csv`, trains the from-scratch forest
(15 trees, max_depth 10, min_samples_split 5), and overwrites
`Crop_recommendation_RF.pkl` in the `{"model":..., "feature_cols":[...]}`
format that `loader.py` expects.

### Files
- `train_and_save.py` — **OFFICIAL** crop-training script (use this).
- `agrosmart_crop_dataset.csv` — merged training data (15 Nepal crops).
- `random_forest.py` — from-scratch Random Forest (sklearn-compatible API).
- `decision_tree.py` — from-scratch decision tree used by the forest.
- `crop_ranges.py` — agronomic N/P/K/temp/humidity/ph/rainfall ranges per crop
  (anchored to NARC/CIMMYT studies), used for synthetic data generation.
- `compare_algorithms.py` — the 5-classifier accuracy comparison above.
- `loader.py` — loads the pickle bundle and runs inference for `views.py`.

### Real Nepal soil data (collected, not yet used for labels)
`collect_nepal_data.py` queries `https://soil.narc.gov.np/soil/api/?lat={lat}&lon={lon}`
(NARC + CIMMYT, Dec 2024) and returns real measured soil N/P/K/pH across Nepal.
Future work: collect real field labels for all 15 crops and retrain end-to-end
on 100% Nepal data.

---

## 2. Disease Detection Model (`disease_cnn_model.npz`)

**Algorithm:** Convolutional Neural Network written **entirely from scratch in
NumPy** (no TensorFlow/Keras). This is the model the live app actually uses —
the earlier Keras `.h5` model described in older docs was replaced by this
from-scratch implementation.
**Input:** 128×128 RGB leaf image.
**Output:** 29 disease/health classes across 7 crops (Maize, Potato, Rice,
Mango, Sugarcane, Wheat, Banana).

### Architecture
```
Conv2D(3→16, 3×3, pad=1) → ReLU → MaxPool(2)   : 128 → 64
Conv2D(16→32, 3×3, pad=1) → ReLU → MaxPool(2)  : 64  → 32
Conv2D(32→64, 3×3, pad=1) → ReLU → MaxPool(2)  : 32  → 16
Flatten (16*16*64 = 16384) → Dense(16384→128) → ReLU → Dense(128→29)
```
Trained with SGD + momentum (lr 0.01, momentum 0.9), cross-entropy loss, 20
epochs, batch size 32, 70/15/15 train/val/test split, seed 42, with on-the-fly
augmentation.

### Dataset (real images)
The `label_map.json` maps each integer 0–28 to a `Crop__Class` name (e.g.
`Maize__Common_Rust`). Class counts are listed in `dataset_summary.csv`
(14,855 images total). Sources (see citations below):
- Maize + Potato — downsampled plant-disease dataset
- Rice — RiceDiseases-DataSet (GitHub)
- Mango — Mango-Leaf-Disease-Detection (GitHub)
- Sugarcane — Sugarcane-Leaf-Disease-Detection (GitHub)
- Wheat — Kaggle `olyadgetch/wheat-leaf-dataset`
- Banana — Mendeley `BananaLSD` dataset (9tb7k297ff)

### Live inference
- `disease_loader.py` — image preprocessing + inference (`predict_disease`).
  It loads `disease_cnn_model.npz` via `cnn.model_io.load_model`, resizes the
  uploaded image to 128×128 with Pillow, normalizes with NumPy, and returns
  `{"crop", "disease", "is_healthy", "confidence"}`.

### How to retrain
The training code lives **outside** the web app, in `ml_training/` (offline):
```bash
cd ml_training
python build_dataset.py     # builds prepared_dataset.npz + label_map.json
python cnn/train.py         # trains and saves disease_cnn_model.npz
```
Then copy `ml_training/disease_cnn_model.npz` and `ml_training/label_map.json`
into `recommender/ml/` so the live app can load them.

### Files
- `disease_cnn_model.npz` — trained from-scratch CNN weights (loaded by the app).
- `label_map.json` — 29 class names (`Crop__Class`).
- `disease_loader.py` — image preprocessing + inference (`predict_disease`).
- `ml_training/cnn/` — the from-scratch NumPy CNN (model, layers, optimizer,
  loss, training loop).
- `ml_training/build_dataset.py` — offline dataset prep pipeline.

---

## Dataset Citations (IEEE style — for the project report)

[1] A. Ingle, "Crop Recommendation Dataset," Kaggle. [Online]. Available:
    https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

[2] attaullah, "Downsampled Plant Disease Dataset (plant64.npz)," GitHub.
    [Online]. Available:
    https://github.com/attaullah/downsampled-plant-disease-dataset

[3] aldrin233, "RiceDiseases-DataSet," GitHub. [Online]. Available:
    https://github.com/aldrin233/RiceDiseases-DataSet

[4] Anas436, "Mango-Leaf-Disease-Detection," GitHub. [Online]. Available:
    https://github.com/Anas436/Mango-Leaf-Disease-Detection

[5] RoshitaB, "Sugarcane-Leaf-Disease-Detection," GitHub. [Online]. Available:
    https://github.com/RoshitaB/Sugarcane-Leaf-Disease-Detection

[6] olyadgetch, "Wheat Leaf Dataset," Kaggle. [Online]. Available:
    https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset

[7] P. Gonzalez-De-La-Cruz et al., "Banana Leaf Spot Diseases (BananaLSD)
    Dataset," Mendeley Data, 2022. [Online]. Available:
    https://data.mendeley.com/datasets/9tb7k297ff/1

[8] Nepal Agricultural Research Council (NARC) and CIMMYT, "Digital Soil Map
    of Nepal," 2024. [Online]. Available: https://soil.narc.gov.np/data
