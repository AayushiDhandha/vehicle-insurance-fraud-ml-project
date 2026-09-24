# 🛡️ Vehicle Insurance Fraud Detection & Intelligence Platform

A comprehensive Machine Learning project developed for **Semester 5 Machine Learning Lab**. This repository implements the complete end-to-end data science lifecycle from exploratory analysis to cloud deployment with real-time inference (Tasks 1 to 6).

---

## 📌 Project Overview & Tasks Summary

| Task | Component | Description | File / Artifact |
| :--- | :--- | :--- | :--- |
| **Task 1** | **Exploratory Data Analysis (EDA)** | Univariate, bivariate analysis, correlation matrices, and fraud risk distributions across 12,002 claims | [`Task1.ipynb`](./Task1.ipynb) |
| **Task 2** | **Data Cleaning & Preprocessing** | Handling missing entries (`?`), outlier filtering (3σ), categorical one-hot encoding, and feature scaling | [`Task2.ipynb`](./Task2.ipynb) |
| **Task 3** | **Linear Regression & Gradient Descent** | Regression analysis predicting financial loss and claim payout with cost optimization | [`Task3.ipynb`](./Task3.ipynb) |
| **Task 4 & 5** | **Classification Modeling & Tuning** | Evaluated 5 classifiers (Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting) with 5-fold cross-validation | [`Task5_Classification.ipynb`](./Task5_Classification.ipynb) |
| **Task 6** | **Cloud Deployment** | Streamlit Community Cloud + Full-Stack React/Vite Frontend on Vercel with FastAPI Backend on Render | [`streamlit_app.py`](./streamlit_app.py), [`backend/`](./backend), [`Front-end/`](./Front-end) |

---

## 🏆 Best Classification Model (Task 5)
- **Winning Algorithm:** Gradient Boosting Classifier (`Scikit-Learn`)
- **Trained Dataset:** `cleaned_data.csv` (12,002 rows, 42 features)
- **Model Accuracy:** 77.8% (Cross-Validated F1 Score: ~0.21)
- **Serialized Model File:** [`backend/model.joblib`](./backend/model.joblib)

---

## 🚀 Live Cloud Deployment Links (Task 6)

- **Streamlit Live Dashboard:** [Open Streamlit App](https://share.streamlit.io)
- **Full-Stack Vercel Frontend:** [Open Vercel App](https://vercel.com)
- **FastAPI Render Backend:** `https://your-service-name.onrender.com`

---

## 🛠️ How to Run Locally

### 1. Run Streamlit Dashboard (Task 1 to 6 in One App)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 2. Run Full-Stack Mode (FastAPI Backend + React Frontend)

**Start FastAPI Backend:**
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- Interactive Swagger API Docs: `http://localhost:8000/docs`

**Start Frontend Dashboard:**
```bash
cd Front-end
npm install
npm run dev
```
- Open in browser: `http://localhost:3000`
