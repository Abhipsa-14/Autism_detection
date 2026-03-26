# Autism Detection Web Application

🚀 A fast, modern, and production-grade Web Application for early Autism Spectrum Disorder (ASD) detection. 

It powers a machine learning model based on the clinically validated **AQ-10 (Autism Spectrum Quotient)** screening questionnaire combined with user demographics, accurately predicting an individual's autism risk level and providing animated visual feedback.

## 🌟 Key Features

* **Machine Learning Intelligence:** Powered by an XGBoost model tuned and balanced via SMOTE, achieving ~89% AUC.
* **Asynchronous High-Performance API:** Built with **FastAPI** and `asyncpg`. The API handles database calls asynchronously and executes CPU-bound ML predictions in an isolated thread pool to prevent blocking.
* **Modern "Glassmorphism" UI:** A stunning, lightweight vanilla JavaScript frontend styled dynamically using **Tailwind CSS**.
* **Stateless Security:** Robust JWT (JSON Web Token) authentication with `bcrypt` password hashing.
* **Database Persistence:** Every user account and prediction result is safely logged to a **PostgreSQL** database, paving the way for future continuous model retraining.
* **Historical Tracking:** Users can view a complete sidebar feed of all their past screenings.

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
│   │   ├── api/         # FastAPI execution endpoints (Auth & Predictions)
│   │   ├── core/        # Security, JWT tokens, and hashing
│   │   ├── db/          # Database connection strings and SQLAlchemy models
│   │   ├── schemas/     # Pydantic validation schemas
│   │   └── services/    # ML Inference Singleton Layer (Thread Pool)
│   ├── models/          # Jupyter scripts, training pipeline, and `.pkl` artifacts
│   ├── data/            # Local datasets (autism.csv)
│   ├── main.py          # FastAPI Server Entrypoint
│   └── requirements.txt # Python Dependencies
│
├── frontend/
│   ├── index.html       # Landing Page / Authentication Portal
│   ├── dashboard.html   # Main Dashboard (Questionnaire & Results UI)
│   └── static/          # Custom CSS stylesheets
│
├── .env                 # Environment variables (DB URL, Secrets)
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

### 4. Optional: Retrain the Model
If you've updated the dataset in `backend/data/autism.csv`, you can retrain the model locally to overwrite the existing `.pkl` files:

```bash
cd backend
python models/train_model.py
```

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
