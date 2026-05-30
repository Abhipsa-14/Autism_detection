# Autism Detection Web Application

🚀 A fast, modern, and production-grade Web Application for early Autism Spectrum Disorder (ASD) detection. 

It powers a machine learning model based on the clinically validated **AQ-10 (Autism Spectrum Quotient)** screening questionnaire combined with user demographics, accurately predicting an individual's autism risk level and providing animated visual feedback.

## 🌟 Key Features

* **Machine Learning Intelligence:** XGBoost classifier (selected over RandomForest by cross-validated ROC-AUC), trained on the **merged UCI Adult + Adolescent + Child** screening cohorts (1,100 records) and balanced with SMOTE.
* **Calibrated, honest probabilities:** Probabilities are calibrated with `CalibratedClassifierCV` (isotonic) on a held-out set, so a reported "78%" actually behaves like 78%.
* **Explainable results (SHAP):** Every prediction returns the top factors that pushed the score up or down, surfaced directly in the UI.
* **Asynchronous High-Performance API:** Built with **FastAPI** and `asyncpg`. CPU-bound ML inference runs in an isolated thread pool to keep the event loop free.
* **Modern "Glassmorphism" UI:** Lightweight vanilla JS frontend styled with **Tailwind CSS**.
* **Stateless Security:** JWT authentication with `bcrypt` password hashing.
* **Database Persistence:** Every account and prediction is logged to **PostgreSQL**, enabling future retraining.
* **Model versioning:** Each training run writes `model_metadata.json` (version, training date, dataset SHA-256, full metrics), and timestamped metric reports are kept under `models/reports/`.

---

## ⚠️ Scientific Honesty: What This Tool Is (and Isn't)

This is an **AQ-10 screening assistant**, not an autism "detector" and **not a clinical diagnosis**. Two important caveats a reviewer should know:

1. **Label derivation / leakage.** In the UCI dataset, the `Class/ASD` label is itself derived from the AQ-10 score (a sum-threshold rule), not from an independent clinical diagnosis. Our EDA measured this directly: the simple rule *"AQ-10 sum ≥ 6"* agrees with the stored label **88.5%** of the time, and **every** record with a sum below 6 is labeled negative. We drop the leaky `result` column during training, but the model still partly re-learns the questionnaire's own scoring rule. High accuracy here is therefore partly an artifact — it does **not** prove the model out-performs the questionnaire.
2. **Purpose.** The model's real job is to flag risk, **explain why** (SHAP), and route the user to a qualified professional for a formal evaluation.

### Held-out test metrics (v1, 220 unseen samples)

| Metric | Value |
|---|---|
| ROC-AUC | 0.989 |
| PR-AUC | 0.976 |
| Sensitivity / Recall | 0.962 |
| Specificity | 0.993 |
| Precision | 0.987 |
| F1 | 0.974 |
| Brier score (calibrated) | 0.018 |

Full, reproducible metrics are written to `backend/models/model_metadata.json` and `backend/models/reports/` on every training run.

---

## 🛠 Tech Stack

* **Backend Framework:** FastAPI (Python 3)
* **Databases:** PostgreSQL (via `asyncpg` and SQLAlchemy 2.0)
* **Machine Learning:** `scikit-learn`, `xgboost`, `imblearn` (SMOTE), `pandas`
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (ES6+), Tailwind CSS (CDN)
* **Security:** `python-jose` (JWT), `passlib[bcrypt]`

---

## 📂 Project Structure

The repository is modularly structured, enforcing a strict separation of concerns between API routing, database models, ML code, and static frontend assets.

```
Autism_detection_Project/
├── backend/
│   ├── app/
│   │   ├── api/         # FastAPI endpoints (Auth & Predictions)
│   │   ├── core/        # Security, JWT tokens, and hashing
│   │   ├── db/          # Database connection and SQLAlchemy models
│   │   ├── schemas/     # Pydantic validation schemas
│   │   └── services/    # ML inference singleton (thread pool + SHAP)
│   ├── data/
│   │   ├── download_datasets.py  # fetch UCI Adult/Adolescent/Child (ARFF)
│   │   ├── prepare_dataset.py    # merge -> autism_merged.csv (+ age_group)
│   │   ├── analyze_dataset.py    # EDA / leakage check
│   │   ├── autism_merged.csv     # merged training data (1,100 rows)
│   │   └── raw/                  # downloaded .arff source files
│   ├── models/
│   │   ├── train_model.py        # training + calibration + SHAP + versioning
│   │   ├── trained_model.pkl     # calibrated model (used by the API)
│   │   ├── base_model.pkl        # tree model (used for SHAP)
│   │   ├── scaler.pkl            # StandardScaler
│   │   ├── model_metadata.json   # version, dataset hash, full metrics
│   │   └── reports/              # metrics JSON + shap_summary.png
│   ├── main.py          # FastAPI server entrypoint
│   └── requirements.txt # Python dependencies
│
├── frontend/
│   ├── index.html       # Landing page / authentication portal
│   └── dashboard.html   # Main dashboard (questionnaire + explainable results)
│
├── .env                 # Environment variables (DB URL, secrets)
└── README.md            # You are here
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
* **Python 3.9+** installed.
* **PostgreSQL** installed and running.

### 2. Configure Environment Variables
Inside the project root directory, edit the `.env` file to contain your Database URL and a secure JWT Secret:

```ini
# .env 

# Frontend and Backend URLs
BACKEND_URL=http://127.0.0.1:8000
FRONTEND_URL=http://127.0.0.1:8000

# PostgreSQL Connection String
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/autism_db

# Security
JWT_SECRET=super_secret_secure_key_12345
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440
```

*Note: You must explicitly create the database `autism_db` inside PostgreSQL first.*

### 3. Install Dependencies
Navigate into the `backend/` directory and install the required Python packages:

```bash
cd backend
pip install -r requirements.txt
```

### 4. Optional: Rebuild Data & Retrain the Model

The training pipeline uses a **merged** dataset (Adult + Adolescent + Child). To rebuild it from scratch and retrain:

```bash
cd backend
python data/download_datasets.py    # fetch the 3 UCI cohorts (ARFF)
python data/prepare_dataset.py      # merge -> data/autism_merged.csv (+ age_group)
python data/analyze_dataset.py      # optional EDA (prints leakage check)
python models/train_model.py        # train, calibrate, SHAP, version, save
```

This overwrites `trained_model.pkl` (calibrated), `base_model.pkl` (for SHAP), `scaler.pkl`, and writes `model_metadata.json` plus a timestamped metrics report under `models/reports/`.

### 5. Start the Server
Start the Uvicorn ASGI server from the `backend/` directory:

```bash
cd backend
uvicorn main:app --reload
```

---

## 🖥 Usage

1. Open a browser and navigate to **`http://127.0.0.1:8000`**.
2. **Register** a new account.
3. **Log in** to your account.
4. Fill out the **AQ-10 Questionnaire**.
5. Submit the form to instantly view your evaluated Risk Level and Confidence Percentage. The result will permanently sync to your History feed on the left side of the screen.

---

## 👩‍💻 Author Information

This project was carefully architected and crafted to modernize autism screening accessibility.

**Made by:** Abhipsa Padhi  
**Email:** [abhidisha14@gmail.com](mailto:abhidisha14@gmail.com)
