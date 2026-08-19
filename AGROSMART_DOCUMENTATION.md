# AgroSmart: Smart Crop Recommendation and Plant Disease Detection System for Nepal

> **Source of truth** for the AgroSmart project. This document describes the system as it actually exists in the deployed codebase.

---

## 1. Project Overview

AgroSmart is a web-based agricultural decision-support system built with **Django**. It helps farmers and agricultural officers in **Nepal** make two primary decisions using machine learning:

1. **Which crop to grow** — given soil nutrient and climate inputs, the system recommends the most suitable crop along with a confidence score, alternative options, and a suitability checklist.
2. **What disease a plant has** — given a photo of a plant leaf, the system identifies the crop and the specific disease (or "Healthy") and returns treatment advice.

To improve academic understanding, the machine learning algorithms were implemented from scratch using Python and NumPy. The implementation exposes all major stages of the algorithms including tree construction, feature selection, convolution, pooling, backpropagation and optimization.

The application also includes user accounts, an admin dashboard, prediction history, favorites, notifications, fertilizer advice, a keyword chatbot, a crop calendar, crop comparison, live weather integration, and data export (CSV/PDF).

---

## 2. Project Objectives

- Provide farmers with a **data-driven crop recommendation** based on soil N-P-K, temperature, humidity, pH, and rainfall.
- Enable **early, low-cost disease detection** from a simple leaf photograph, covering 7 major crops and 29 disease/health classes.
- Offer **fertilizer guidance** derived from soil nutrient levels.
- Present a **multilingual, accessible UI** (English + Nepali preference) suitable for field use.
- Demonstrate **from-scratch ML implementations** (Random Forest, CNN) with reproducible training pipelines.
- Give administrators **analytics and oversight** over users and predictions.

---

## 3. Problem Statement

Smallholder farmers in Nepal often lack access to agronomic expertise. Choosing the wrong crop for local soil and climate leads to low yields and financial loss. Likewise, leaf diseases are frequently identified too late or incorrectly, causing preventable crop damage.

AgroSmart addresses both gaps:

- A **crop-recommendation engine** that reasons over measurable environmental features.
- A **visual disease classifier** that works on ordinary leaf photos.
- A **transparent, from-scratch implementation** so every computation can be inspected and defended.

---

## 4. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Web framework | Django 6 |
| Database | SQLite 3 (`db.sqlite3`) |
| Numerical computing | NumPy (CNN + dataset pipeline) |
| Image decoding | Pillow (reads/resizes images; no ML) |
| Templates | Django Template Language (29 HTML templates) |
| Styling/UI | HTML/CSS + static assets |
| Charts | Chart.js (admin dashboard, analytics, history) |
| Weather API | Open-Meteo (Geocoding + Forecast), free, no API key |
| ML models (live) | Custom implementations developed in Python and NumPy (Random Forest + CNN) |

The deployed models use custom implementations developed in Python and NumPy.

---

## 5. System Architecture

The system has three logical tiers:

```
                ┌─────────────────────────────────────────────┐
   Browser ───► │  Django (recommender app)                    │
                │  URLs → Views → Models (SQLite) → Templates  │
                └───────────┬───────────────┬─────────────────┘
                            │               │
              ┌─────────────▼────┐   ┌───────▼──────────────────┐
              │ Crop Rec ML      │   │ Disease Detection ML       │
              │ recommender/ml/  │   │ recommender/ml/            │
              │  loader.py       │   │  disease_loader.py         │
              │  → RF pickle     │   │  → cnn.model_io.load_model │
              └──────────────────┘   └────────────────────────────┘
                            │               │
                    agrosmart_crop_   disease_cnn_model.npz
                    dataset.csv        + label_map.json (29 classes)
                            │
              ┌─────────────▼──────────────────────────────────┐
              │ OFFLINE TRAINING (not imported by web app)       │
              │  ml_training/  (dataset pipeline + cnn/)         │
              │  produces prepared_dataset.npz, .npz weights     │
              └──────────────────────────────────────────────────┘
```

**Request flows**
- **Crop recommendation:** `predict` view → optional Open-Meteo weather fill → `loader.predict_with_confidence` → custom Random Forest `predict_proba` → save `Prediction` → suitability checklist + top-3 → render.
- **Disease detection:** `disease_detection` view → `disease_loader.predict_disease` → load custom CNN → Pillow resize + NumPy normalize → `predict_proba` → `Crop__Class` label split → `get_disease_info` → save `DiseaseDetection` + `Notification` → render.

---

## 6. Project Folder Structure

```
cropsite/               Django project package (settings, urls, wsgi, asgi)
manage.py               Django CLI entrypoint
db.sqlite3              SQLite database
media/                  Uploaded media (leaf images)
recommender/            The Django application
    models.py
    urls.py
    views.py
    templates/
    static/
    ml/                  ML models + inference (live)
ml_training/            Offline training code (not web-imported)
```

---

## 7. Database Design

Engine: **SQLite**. ORM models live in `recommender/models.py`. Django's `User` (auth) is extended via a one-to-one `UserProfile`.

### Tables

| Model | Fields | Notes |
|---|---|---|
| **UserProfile** | `user` (1:1 User), `phone`, `bio` (null), `profile_image`, `preferred_language` (default `en`) | Extended user profile |
| **Prediction** | `user` (FK), `N,P,K,temperature,humidity,ph,rainfall` (Float), `predicted_label`, `created_at` | Crop records; ordered by `-created_at` |
| **FavoriteCrop** | `user` (FK), `crop_name`, `created_at` | Saved crops |
| **Feedback** | `name`, `email`, `message`, `created_at` | User feedback |
| **Notification** | `user` (FK), `message`, `is_read` (default False), `created_at` | System alerts |
| **DiseaseDetection** | `user` (FK), `image`, `predicted_label` (null), `confidence` (null), `created_at` | Detection records |
| **FertilizerRecommendation** | `user` (FK), `soil_nitrogen,soil_phosphorus,soil_potassium`, `crop_type`, `recommendation`, `created_at` | Fertilizer advice |

All models registered in Django admin. Relationships: `User` 1—1 `UserProfile`; `User` 1—N `Prediction`, `FavoriteCrop`, `Notification`, `DiseaseDetection`, `FertilizerRecommendation`.

---

## 8. Complete Feature List

### User Management
- Registration (name, phone, email, password)
- Login / Logout
- Profile edit (name, phone, bio, language, image)
- Password management (change, forgot, reset)

### Crop Recommendation Module
- Crop prediction (N,P,K,temp,humidity,ph,rainfall → crop + confidence + top-3)
- Live weather integration (Open-Meteo by city)
- Suitability checklist (input vs. crop's acceptable ranges)

### Disease Detection Module
- Leaf image upload
- CNN-based disease prediction (crop + disease + confidence)
- Treatment recommendation (symptoms, causes, prevention, treatment)

### Fertilizer Recommendation Module
- Soil N/P/K + crop → rule-based fertilizer advice

### History Module
- Prediction history (with monthly chart)
- Delete prediction
- Favorites (save / remove)
- CSV export (user's own predictions)

### Admin Module
- Admin login (staff gate)
- Dashboard (totals, recent, top-crops donut)
- Analytics (totals, top crops, monthly chart)
- User management (list / delete users)
- Feedback review
- Predictions review
- Export predictions (CSV / PDF)

### Additional Features
- Chatbot (keyword-based Q&A)
- Crop calendar
- Compare crops (2 crops side-by-side)
- Map location view
- Farmer connect page
- Feedback form
- About page
- Home / landing page (site stats)

---

## 9. User Module

Accessible to authenticated (and, for some, anonymous) users.

| Feature | View | Route |
|---|---|---|
| Home | `home` | `/` |
| Sign up | `signup_view` | `/signup/` |
| Login | `login_view` | `/login/` |
| Logout | `logout_view` | `/logout/` |
| Predict (crop) | `predict_view` | `/predict/` |
| History | `user_history_view` | `/history/` |
| Delete prediction | `user_delete_prediction` | `/history_delete/<id>/` |
| Profile | `profile_view` | `/profile/` |
| Change password | `change_password_view` | `/change_password/` |
| Forgot password | `forgot_password_view` | `/forgot_password/` |
| Reset password | `reset_password_view` | `/reset_password/` |
| Favorites | `favorites_view` | `/favorites/` |
| Remove favorite | `remove_favorite_view` | `/favorites/remove/<id>/` |
| Feedback | `feedback_view` | `/feedback/` |
| Notifications | `notifications_view` | `/notifications/` |
| Mark notification read | `mark_notification_read` | `/notifications/read/<id>/` |
| Delete notification | `delete_notification` | `/notifications/delete/<id>/` |
| Disease detection | `disease_detection_view` | `/disease_detection/` |
| Fertilizer | `fertilizer_recommendation_view` | `/fertilizer/` |
| Crop calendar | `crop_calendar_view` | `/crop_calendar/` |
| Compare crops | `compare_crops_view` | `/compare_crops/` |
| Chatbot | `chatbot_view` | `/chatbot/` |
| Clear chat | `clear_chat` | `/chatbot/clear/` |
| Map location | `map_location_view` | `/map_location/` |
| Farmer connect | `farmer_connect_view` | `/farmer-connect/` |
| About | `about_view` | `/about/` |
| Weather JSON | `weather_json_view` | `/api/weather/` |
| Export my CSV | `export_user_predictions_csv` | `/export_user_predictions_csv/` |

User accounts use Django's built-in `User` (username = email). Passwords require ≥6 characters. Profiles store phone, bio, language preference, and an optional image.

---

## 10. Admin Module

Guarded by `@user_passes_test(is_staff)` with `login_url='admin_login'`.

| Feature | View | Route |
|---|---|---|
| Admin login | `admin_login_view` | `/admin_login/` |
| Dashboard | `admin_dashboard_view` | `/admin_dashboard/` |
| Predictions | `admin_predictions_view` | `/admin_predictions/` |
| Users | `admin_users_view` | `/admin_users/` |
| Delete user | `delete_user_view` | `/delete_user/<id>/` |
| Feedback | `admin_feedback_view` | `/admin_feedback/` |
| Export CSV | `export_predictions_csv` | `/export_predictions_csv/` |
| Export PDF | `export_predictions_pdf` | `/export_predictions_pdf/` |
| Analytics | `analytics_view` | `/analytics/` |

The dashboard shows total users, total predictions, recent predictions, and a Chart.js donut of top-5 predicted crops. The PDF export is generated without any library (`build_simple_pdf` writes raw PDF objects).

---

## 11. Crop Recommendation Module

**Inputs (7 features):** `N, P, K, temperature, humidity, ph, rainfall`.

**Model:** A custom **Random Forest** (`random_forest.py` + `decision_tree.py`). The pickle `Crop_recommendation_RF.pkl` holds `{"model": RandomForest, "feature_cols": [...]}`. It exposes `predict()`, `predict_proba()`, `classes_` so `loader.py` is library-agnostic.

**Pipeline (`predict_view`):**
1. Read 7 numeric inputs (validated, must be non-empty).
2. If a city is given, fetch live weather and override temperature/humidity/rainfall.
3. `predict_with_confidence(data)` → label, confidence (real `predict_proba`), top-3.
4. Save a `Prediction`.
5. `get_crop_info(label)` returns the crop's agronomy; `build_suitability_checklist` compares inputs to the crop's acceptable ranges (temp/humidity/ph/rainfall, with 10% tolerance) and flags each as met/unmet.
6. Render `predict.html` with result, top-3 alternatives, and checklist.

**Crop coverage:** 13 crops in `agrosmart_crop_dataset.csv` — barley, buckwheat, cardamom, ginger, lentil, maize, millet, mustard, potato, rice, sugarcane, tea, wheat. A separate `get_crop_info()` dictionary (in `views.py`) holds agronomy for **30 crops** for display/comparison.

---

## 12. Disease Detection Module

**Input:** Uploaded leaf image (RGB).
**Output:** `{"crop", "disease", "is_healthy", "confidence"}` over **29 classes across 7 crops**.

**Live inference (`disease_loader.py`):**
1. `get_model()` (cached) loads `disease_cnn_model.npz` via `cnn.model_io.load_model`, rebuilding the exact architecture (29 classes, 128×128×3).
2. `preprocess_uploaded_image` → Pillow `convert("RGB")` + `resize(128,128)` + NumPy `/255.0` → shape `(1,128,128,3)`.
3. `model.predict_proba(x)` → `argmax` gives predicted index; `label_map.json` maps index → `"Crop__Class"`.
4. Split on `"__"` → crop + disease (underscores → spaces). `is_healthy` if disease == `"Healthy"`.
5. `get_disease_info(crop, disease)` (`disease_info.py`) returns symptoms/causes/prevention/treatment for all 29 classes.
6. Save `DiseaseDetection`; create a `Notification`.

**Classes (29):** Banana (Cordana, Healthy, Pestalotiopsis, Sigatoka), Maize (Common_Rust, Gray_Leaf_Spot, Healthy, Northern_Leaf_Blight), Mango (Anthracnose, Bacterial_Canker, Cutting_Weevil, Die_Back, Gall_Midge, Healthy, Powdery_Mildew, Sooty_Mould), Potato (Early_Blight, Healthy, Late_Blight), Rice (Bacterial_Leaf_Blight, Blast, Brown_Spot, Healthy), Sugarcane (Healthy, Red_Rot, Red_Rust), Wheat (Healthy, Septoria, Stripe_Rust).

---

## 13. Fertilizer Recommendation Module

**Input:** soil nitrogen, phosphorus, potassium (numeric) + crop type.
**Logic (`fertilizer_recommendation_view`):** rule-based thresholds:
- Nitrogen: <50 → apply urea/compost; 50–80 → half-dose urea; ≥80 → avoid.
- Phosphorus: <30 → DAP/bone meal; 30–60 → 50% DAP; ≥60 → none.
- Potassium: <200 → potassium sulfate; 200–400 → MOP; ≥400 → sufficient.

Result saved as a `FertilizerRecommendation`. Crop options: Rice, Maize, Wheat, Potato, Lentil, Chickpea, Mango, Banana, Ginger.

---

## 14. Weather Integration

`views.py` → `get_weather_data(city)`:
1. Geocode city via `https://geocoding-api.open-meteo.com/v1/search?name=<city>&count=1`.
2. Fetch current weather via `https://api.open-meteo.com/v1/forecast?latitude=..&longitude=..&current=temperature_2m,relative_humidity_2m,precipitation`.
3. Returns `{temperature, humidity, rainfall, description}`.

Used in `predict_view` to auto-fill climate fields, and exposed as JSON at `/api/weather/`. Free, no API key, 8s timeout; falls back to user-entered values on failure.

---

## 15. Dataset Description

### 15.1 Crop recommendation dataset
The crop recommendation dataset was prepared using agronomic crop ranges suitable for Nepal. The dataset contains **1,950 records** representing **13 major crops** with seven environmental features (N, P, K, temperature, humidity, pH, rainfall). Crop ranges are anchored to Nepal-specific (and regional) agronomic studies, and values are sampled to reflect realistic field conditions with correlated humidity and rainfall.

### 15.2 Disease detection dataset
- Folder: `ml_training/dataset/<Crop>/<Class>/`.
- **14,855 images total** across 7 crops / 29 classes (see `dataset_summary.csv`):

| Crop | Images |
|---|---|
| Maize | 4188 |
| Mango | 4000 |
| Potato | 2152 |
| Rice | 2569 |
| Banana | 937 |
| Sugarcane | 603 |
| Wheat | 406 |

- Sources: downsampled plant-disease dataset (Maize/Potato), RiceDiseases-DataSet, Mango-Leaf-Disease-Detection, Sugarcane-Leaf-Disease-Detection, Kaggle wheat-leaf-dataset, Mendeley BananaLSD.
- **Class imbalance:** Wheat/Sugarcane are much smaller than Maize/Mango — a known limitation.
- **Data hygiene:** `verify_dataset.py` checks image counts, empty folders, corrupted images (PIL `verify`), mixed formats, duplicate class names (e.g. `Healthy` vs `healthy`), and structure vs. the 7-crop/29-class plan.

---

## 16. Random Forest Algorithm (from scratch)

Implemented in `recommender/ml/random_forest.py` + `decision_tree.py`.

**DecisionTree**
- Recursively splits rows on the feature + threshold that maximize **information gain** (entropy-based).
- `max_depth`, `min_samples_split` limit growth; leaves store the majority label.
- `predict(row)` walks the tree to a leaf.

**RandomForest**
- `n_trees` (15), `max_depth=10`, `min_samples_split=5`, `n_features_to_try="sqrt"`.
- Each tree trained on a **bootstrap sample** of rows, considering a random subset of features per split.
- Prediction = **majority vote** across trees.
- `predict_proba` averages per-tree class probabilities → powers confidence % and top-3.
- `classes_` collects all leaf labels for compatibility with `loader.py`.

---

## 17. CNN Architecture (from scratch)

Implemented in `ml_training/cnn/` in Python and NumPy. Input 128×128×3, 29 outputs.

```
Conv2D(3 → 16, 3×3, pad=1) → ReLU → MaxPool(2×2)    128→64
Conv2D(16 → 32, 3×3, pad=1) → ReLU → MaxPool(2×2)    64→32
Conv2D(32 → 64, 3×3, pad=1) → ReLU → MaxPool(2×2)    32→16
Flatten (16·16·64 = 16384)
Dense(16384 → 128) → ReLU
Dense(128 → 29)                                    raw logits
```

| Component | Responsibility |
|---|---|
| `conv2d` | Convolution via `im2col`/`col2im` vectorization; He initialization |
| `dense` | Fully-connected layer (He init) |
| `pooling` | MaxPool2D |
| `flatten` | Flatten spatial → vector |
| `activations` | ReLU |
| `softmax` | Softmax |
| `losses` | Fused softmax + cross-entropy (grad = `probs − y_true`) |
| `optimizer` | SGD with momentum (lr 0.01, momentum 0.9), in-place updates |
| `model` | `CNNModel` chaining layers; `forward`, `backward`, `get_params_and_grads`, `predict_proba` |
| `model_io` | Save/load all layer W/b into one `.npz` by rebuilding the architecture |

Backpropagation runs gradients backward through each layer. The optimizer applies momentum updates to every `(param, grad)` pair.

---

## 18. Training Pipeline

**Disease CNN (`ml_training/`)**
1. `config.py` — `IMG_SIZE=(128,128)`, 70/15/15 split, seed 42, batch 32.
2. `data_loader.py` — walks `dataset/<Crop>/<Class>/`.
3. `label_encoder.py` — maps each `(crop,class)` to 0–28.
4. `preprocessing.py` — Pillow decode + resize + NumPy normalize to [0,1].
5. `dataset_builder.py` — load all images, one-hot encode, shuffle, split → `prepared_dataset.npz`.
6. `batch_generator.py` / `augmented_batch_generator.py` — mini-batches, with augmentation.
7. `cnn/train.py` — 20 epochs, SGD+momentum, tracks train/val loss & accuracy; saves `disease_cnn_model.npz`.

To deploy: copy `ml_training/disease_cnn_model.npz` + `label_map.json` into `recommender/ml/`.

**Crop Random Forest (`recommender/ml/`)**
- The crop dataset is prepared using Nepal agronomic ranges, then `train_and_save.py` trains the custom RF and pickles `Crop_recommendation_RF.pkl` in `{"model","feature_cols"}` form.

---

## 19. Testing Strategy

- **Dataset verification:** `verify_dataset.py` validates the image dataset (counts, corruption, format consistency, duplicate class names, structure vs. plan) before training.
- **ML evaluation:** `cnn/evaluate.py` and `train.py` report per-epoch train/val loss & accuracy.
- **Django:** `recommender/tests.py` is the app test module (run with `python manage.py test`).
- **Manual/integration:** views are exercised through the browser; admin guards verified via `@user_passes_test`.

---

## 20. Deployment

**Development**
```bash
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver
```
**Production checklist (not yet configured)**
- Set `DEBUG = False`; generate a new `SECRET_KEY`; populate `ALLOWED_HOSTS`.
- Switch from SQLite to PostgreSQL for concurrency.
- Serve `media/` and `static/` via a real web server (nginx) or CDN.
- Run behind WSGI/ASGI (gunicorn/uvicorn) + reverse proxy.
- The custom ML runs in-process (NumPy); no external model server needed.

---

## 21. System Requirements

### Software Requirements
- Python 3.13
- Django 6
- NumPy
- Pillow
- SQLite
- VS Code (or any IDE)
- Git

### Hardware Requirements
- Intel i5 (or equivalent) or better
- 8 GB RAM (minimum)
- 20 GB free storage
- Internet connection (for weather API and dataset download)

---

## 22. Functional Requirements

- **FR1 — User Registration:** Users can create an account with name, phone, email, and password.
- **FR2 — User Login:** Registered users can authenticate and access protected features.
- **FR3 — Crop Recommendation:** System predicts the most suitable crop from 7 environmental inputs with confidence and alternatives.
- **FR4 — Disease Detection:** System identifies crop and disease from an uploaded leaf image with confidence and treatment info.
- **FR5 — Weather Retrieval:** System fetches live temperature, humidity, and rainfall for a given city.
- **FR6 — Fertilizer Recommendation:** System advises fertilizer type based on soil N/P/K and crop.
- **FR7 — Prediction History:** Users can view and delete their past predictions and export them as CSV.
- **FR8 — Admin Dashboard:** Admins can view analytics, manage users, review predictions/feedback, and export data (CSV/PDF).

---

## 23. Non-functional Requirements

- **Performance:** Crop prediction and disease inference run in-process with low latency suitable for interactive use; training is performed offline.
- **Usability:** Bilingual (English/Nepali) preference, simple forms, and clear result pages for non-technical farmers.
- **Security:** Password-hashed auth via Django, CSRF protection, login-required and staff-required guards on sensitive views.
- **Availability:** Runs as a standard Django web app; dev server suffices for demonstration, reverse-proxy for production.
- **Reliability:** Input validation, graceful weather-API fallback, and dataset verification scripts ensure stable operation.
- **Scalability:** Stateless request handling; SQLite is adequate for demo, PostgreSQL recommended for production load.

---

## 24. Scope

### In Scope
- Crop recommendation
- Plant disease detection
- Fertilizer suggestion
- Weather integration
- Prediction history and favorites
- Admin analytics and management

### Out of Scope
- IoT soil/moisture sensors
- Drone imagery
- Satellite remote sensing
- Real-time pest monitoring

---

## 25. Project Limitations

- **SECRET_KEY is hardcoded** in `settings.py` — must change for production.
- **SQLite** does not scale to concurrent production load.
- **Class imbalance** in the disease dataset (Wheat 406 vs Maize 4188) biases the CNN toward majority classes.
- **Chatbot and fertilizer advice are rule-based heuristics**, not learned models.
- **Debug mode on**; no `ALLOWED_HOSTS`, no HTTPS/CSRF hardening for production.
- **Three separate crop lists** (13 recommendation crops, 29 disease classes, 30 in `get_crop_info`) are not unified.
- **No automated test coverage** for ML math or full view set.
- Inconsistent class naming (`Wheat/septoria` lowercase) in the dataset.

---

## 26. Future Enhancements

- Retrain disease CNN on **balanced / augmented** data to reduce majority-class bias.
- Collect **real Nepal field labels** (NARC soil API) and retrain crop model end-to-end.
- Replace the keyword chatbot with a real NLP/LLM assistant.
- Add **model versioning + evaluation dashboard** (per-class precision/recall, confusion matrix).
- Unify crop vocabularies across modules.
- Add **automated tests** (pytest) for ML layers and views.
- Production hardening: env-based secrets, PostgreSQL, CI/CD, containerization.
- Add **mobile-responsive PWA** and offline mode for low-connectivity fields.
- Multi-language UI beyond EN/NE.

---

## 27. References

[1] Django Software Foundation, "Django documentation." https://docs.djangoproject.com/

[2] NumPy, "NumPy documentation." https://numpy.org/doc/

[3] Open-Meteo, "Weather API (Geocoding + Forecast)." https://open-meteo.com/

[4] A. Ingle, "Crop Recommendation Dataset," Kaggle. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

[5] attaullah, "Downsampled Plant Disease Dataset (plant64.npz)," GitHub. https://github.com/attaullah/downsampled-plant-disease-dataset

[6] aldrin233, "RiceDiseases-DataSet," GitHub. https://github.com/aldrin233/RiceDiseases-DataSet

[7] Anas436, "Mango-Leaf-Disease-Detection," GitHub. https://github.com/Anas436/Mango-Leaf-Disease-Detection

[8] RoshitaB, "Sugarcane-Leaf-Disease-Detection," GitHub. https://github.com/RoshitaB/Sugarcane-Leaf-Disease-Detection

[9] olyadgetch, "Wheat Leaf Dataset," Kaggle. https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset

[10] P. Gonzalez-De-La-Cruz et al., "Banana Leaf Spot Diseases (BananaLSD) Dataset," Mendeley Data, 2022. https://data.mendeley.com/datasets/9tb7k297ff/1

[11] Nepal Agricultural Research Council (NARC) and CIMMYT, "Digital Soil Map of Nepal," 2024. https://soil.narc.gov.np/data

[12] Agronomic crop ranges (N/P/K/temp/humidity/ph/rainfall) anchored to NARC / Krishipatrika and regional studies, cited per-crop in `recommender/ml/crop_ranges.py`.

---

*End of document.*
