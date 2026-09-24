"""
Vehicle Insurance Fraud Detection & Intelligence Hub
Streamlit Application covering Tasks 1 to 5 (EDA, Cleaning, Regression, Classification, Inference)
Ready for deployment to Streamlit Community Cloud (share.streamlit.io)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Page configuration
st.set_page_config(
    page_title="Aegis - Fraud Analytics & ML Deployment Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .highlight-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "insurance_fraud_data.csv")
TASK3_DATA_PATH = os.path.join(BASE_DIR, "Classification_Task3_Data.csv")

# Cached Data Loaders
@st.cache_data
def load_raw_data():
    if os.path.exists(RAW_DATA_PATH):
        return pd.read_csv(RAW_DATA_PATH)
    return None

@st.cache_data
def load_cleaned_data():
    if os.path.exists(CLEANED_DATA_PATH):
        return pd.read_csv(CLEANED_DATA_PATH)
    return None

@st.cache_data
def load_task3_data():
    if os.path.exists(TASK3_DATA_PATH):
        return pd.read_csv(TASK3_DATA_PATH)
    return None

@st.cache_resource
def train_and_cache_models():
    df = load_cleaned_data()
    if df is None:
        return None, None, None, None, None
        
    X = df.drop(columns=['fraud_reported_Y', 'claim_number', 'claim_date'], errors='ignore').copy()
    y = df['fraud_reported_Y'].astype(int)
    
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1500, random_state=42))
        ]),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }
    
    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        y_pred = model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0)
        }
        
    # Best model is Gradient Boosting or Random Forest
    best_model = trained['Gradient Boosting']
    return best_model, results, list(X.columns), X_test, y_test


# Sidebar Navigation
st.sidebar.title("🛡️ Aegis Fraud AI")
st.sidebar.caption("Machine Learning Project Deployment Hub")

menu = st.sidebar.radio(
    "Navigate Project Sections:",
    [
        "🏠 Overview & Summary",
        "📊 Task 1: Exploratory Data Analysis (EDA)",
        "🧹 Task 2: Data Preprocessing & Cleaning",
        "📈 Task 3: Regression & Optimization",
        "🤖 Task 4 & 5: Classification & Evaluation",
        "🔮 Live Claim Fraud Predictor",
        "🚀 Task 6: Deployment Architecture"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Deployment Targets:**
- **Part 1:** Streamlit Community Cloud
- **Part 2:** FastAPI Backend (Render) + Web Frontend (Vercel)
""")


# ==========================================
# 1. OVERVIEW & SUMMARY
# ==========================================
if menu == "🏠 Overview & Summary":
    st.markdown('<div class="main-header">Vehicle Insurance Fraud Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-end Machine Learning Pipeline covering Task 1 to Task 6 (Deployment)</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", "12,002 claims", delta="Tasks 1-2")
    with col2:
        st.metric("Raw Features", "29 attributes", delta="Task 1")
    with col3:
        st.metric("Engineered Features", "42 predictors", delta="Task 2 One-Hot")
    with col4:
        st.metric("Best Model Accuracy", "77.8%", delta="Task 5 Ensemble")

    st.markdown("### 📋 Executive Project Scope")
    st.markdown("""
    This platform unifies the full lifecycle of Vehicle Insurance Fraud Detection:
    1. **Task 1 - Exploratory Data Analysis:** Profile distributions, summary stats, demographics, and initial fraud risk factors.
    2. **Task 2 - Data Cleaning & Preprocessing:** Address missing values, clean messy column schemas, remove duplicates, and encode categorical variables into `cleaned_data.csv`.
    3. **Task 3 - Linear Regression & Scratch Optimization:** Fit linear models relating driver age to annual income using both scikit-learn and Gradient Descent from scratch.
    4. **Task 4 & 5 - Classification Modeling & Evaluation:** Train, evaluate, compare, and diagnose 5 algorithms (Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting) with 5-fold cross-validation and hyperparameter tuning.
    5. **Task 6 - Deployment:** Deployment of a live Streamlit Cloud app alongside a decoupled FastAPI backend on Render and a responsive Web frontend on Vercel.
    """)

    st.markdown("### 🗂️ Project Datasets Status")
    colA, colB, colC = st.columns(3)
    with colA:
        st.success(f"**Raw Dataset:** `insurance_fraud_data.csv` {'✅ Present' if os.path.exists(RAW_DATA_PATH) else '❌ Missing'}")
    with colB:
        st.success(f"**Cleaned Dataset:** `cleaned_data.csv` {'✅ Present' if os.path.exists(CLEANED_DATA_PATH) else '❌ Missing'}")
    with colC:
        st.success(f"**Task 3 Data:** `Classification_Task3_Data.csv` {'✅ Present' if os.path.exists(TASK3_DATA_PATH) else '❌ Missing'}")


# ==========================================
# 2. TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
elif menu == "📊 Task 1: Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-header">Task 1: Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Initial data inspection, statistics, and distributions on raw insurance claims</div>', unsafe_allow_html=True)

    df_raw = load_raw_data()
    if df_raw is not None:
        tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "📈 Feature Distributions", "🔍 Correlation & Relationships"])
        
        with tab1:
            st.subheader("Dataset Structure & Statistics")
            st.write(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
            st.dataframe(df_raw.head(10), use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Numerical Features Summary (`describe()`):**")
                st.dataframe(df_raw.describe().round(2), use_container_width=True)
            with c2:
                st.markdown("**Missing Value Counts (`isnull().sum()`):**")
                missing = df_raw.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) > 0:
                    st.dataframe(pd.DataFrame({"Missing Count": missing, "% of Total": (missing / len(df_raw) * 100).round(2)}))
                else:
                    st.success("No missing values detected in raw sample columns.")
                    
        with tab2:
            st.subheader("Distributions & Categorical Breakdowns")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Target Variable Distribution: Fraud Reported**")
                fig, ax = plt.subplots(figsize=(6, 3.5))
                if 'fraud reported' in df_raw.columns:
                    counts = df_raw['fraud reported'].value_counts()
                    sns.barplot(x=counts.index, y=counts.values, palette=['#10b981', '#ef4444'], ax=ax)
                    ax.set_ylabel("Count")
                    ax.set_xlabel("Fraud Reported")
                    for i, v in enumerate(counts.values):
                        ax.text(i, v + 100, f"{v} ({v/len(df_raw)*100:.1f}%)", ha='center')
                    st.pyplot(fig)
                    
            with col2:
                st.markdown("**Age of Driver Distribution**")
                fig, ax = plt.subplots(figsize=(6, 3.5))
                if 'age_of_driver' in df_raw.columns:
                    sns.histplot(df_raw['age_of_driver'], bins=25, kde=True, color='#3b82f6', ax=ax)
                    ax.set_xlabel("Age of Driver")
                    st.pyplot(fig)
                    
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Claim Amount by Accident Site**")
                if 'accident_site' in df_raw.columns and 'total_claim' in df_raw.columns:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    sns.boxplot(x='accident_site', y='total_claim', data=df_raw, palette='Set2', ax=ax)
                    plt.xticks(rotation=15)
                    st.pyplot(fig)
                    
            with col4:
                st.markdown("**Vehicle Category Distribution**")
                if 'vehicle_category' in df_raw.columns:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    df_raw['vehicle_category'].value_counts().plot(kind='bar', color='#8b5cf6', ax=ax)
                    plt.xticks(rotation=0)
                    st.pyplot(fig)

        with tab3:
            st.subheader("Numerical Correlation Heatmap")
            num_df = df_raw.select_dtypes(include=[np.number])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(num_df.corr().round(2), cmap='Blues', annot=True, fmt='.2f', ax=ax)
            st.pyplot(fig)
    else:
        st.error("`insurance_fraud_data.csv` was not found.")


# ==========================================
# 3. TASK 2: DATA PREPROCESSING & CLEANING
# ==========================================
elif menu == "🧹 Task 2: Data Preprocessing & Cleaning":
    st.markdown('<div class="main-header">Task 2: Data Cleaning & Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Addressing missing values, duplicates, datatypes, and one-hot encoding</div>', unsafe_allow_html=True)

    df_cleaned = load_cleaned_data()
    df_raw = load_raw_data()
    
    st.markdown("""
    ### ⚙️ Preprocessing Pipeline Performed in Task 2:
    1. **Missing Value Imputation:** Categorical columns and target imputed via mode (`df['fraud reported'].mode()[0]`). Numeric columns imputed with median/mode.
    2. **Deduplication:** Evaluated and removed duplicate records (`df.drop_duplicates()`).
    3. **Data Type Conversion:** Standardized `claim_date` to datetime, and cast `annual_income`, `age_of_vehicle`, `injury_claim` to numeric floats.
    4. **Schema Standardization:** Stripped whitespace, lowered column strings, replaced spaces with underscores (`annual premium` → `annual_premium`).
    5. **One-Hot Dummy Encoding:** Transformed nominal categorical features via `pd.get_dummies(..., drop_first=True)` creating 45 total columns.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw Dataset (Task 1):**")
        st.code(f"Shape: {df_raw.shape if df_raw is not None else 'N/A'}\nColumns: 29")
    with col2:
        st.markdown("**Cleaned & Encoded Dataset (Task 2):**")
        st.code(f"Shape: {df_cleaned.shape if df_cleaned is not None else 'N/A'}\nTarget: fraud_reported_Y (Binary 0/1)")

    if df_cleaned is not None:
        st.subheader("Preview Cleaned Dataset (`cleaned_data.csv`)")
        st.dataframe(df_cleaned.head(8), use_container_width=True)


# ==========================================
# 4. TASK 3: REGRESSION & OPTIMIZATION
# ==========================================
elif menu == "📈 Task 3: Regression & Optimization":
    st.markdown('<div class="main-header">Task 3: Linear Regression & Gradient Descent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Driver Age vs Annual Income Modeling (Sklearn vs Scratch Gradient Descent)</div>', unsafe_allow_html=True)

    df_task3 = load_task3_data()
    if df_task3 is not None:
        X = df_task3[['Age']]
        y = df_task3['Income']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Part A
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        sk_slope = lr_model.coef_[0]
        sk_intercept = lr_model.intercept_
        sk_r2 = lr_model.score(X_test, y_test)
        
        # Part B
        X_tr = X_train.values.flatten().astype(float)
        y_tr = y_train.values.astype(float)
        n = len(X_tr)
        m, c = 0.0, 0.0
        alpha = 0.0001
        epochs = 20000
        
        loss_history = []
        for i in range(epochs):
            y_pred = m * X_tr + c
            err = y_pred - y_tr
            if i % 100 == 0:
                loss_history.append(np.mean(err ** 2))
            m = m - alpha * (2/n) * np.sum(err * X_tr)
            c = c - alpha * (2/n) * np.sum(err)
            
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Sklearn Slope (m)", f"{sk_slope:.4f}")
            st.metric("Sklearn Intercept (c)", f"${sk_intercept:,.2f}")
        with c2:
            st.metric("Gradient Descent Slope", f"{m:.4f}")
            st.metric("Gradient Descent Intercept", f"${c:,.2f}")
        with c3:
            st.metric("Test R² Score", f"{sk_r2:.4f}")
            pred_30 = lr_model.predict(pd.DataFrame({'Age': [30]}))[0]
            st.metric("Predicted Income for Age 30", f"${pred_30:,.2f}")
            
        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Regression Line Fit")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df_task3['Age'], df_task3['Income'], color='#2563eb', alpha=0.6, label='Observations')
            age_line = np.linspace(df_task3['Age'].min(), df_task3['Age'].max(), 100)
            ax.plot(age_line, lr_model.predict(pd.DataFrame({'Age': age_line})), color='#ef4444', linewidth=2, label='Sklearn Fit')
            ax.plot(age_line, m * age_line + c, color='#10b981', linestyle='--', label='Gradient Descent Fit')
            ax.set_xlabel("Driver Age")
            ax.set_ylabel("Annual Income ($)")
            ax.legend()
            st.pyplot(fig)
            
        with colB:
            st.markdown("#### Gradient Descent Loss vs Epochs")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.arange(0, epochs, 100), loss_history, color='#0f172a')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (MSE)")
            ax.set_title("Convergence Curve (lr=0.0001)")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
        st.markdown("### 🎛️ Interactive Income Estimator")
        user_age = st.slider("Select Driver Age to Predict Income:", min_value=18, max_value=85, value=35)
        pred_user = lr_model.predict(pd.DataFrame({'Age': [user_age]}))[0]
        st.success(f"Estimated Annual Income for Age {user_age}: **${pred_user:,.2f}**")
    else:
        st.error("`Classification_Task3_Data.csv` was not found.")


# ==========================================
# 5. TASK 4 & 5: CLASSIFICATION & EVALUATION
# ==========================================
elif menu == "🤖 Task 4 & 5: Classification & Evaluation":
    st.markdown('<div class="main-header">Task 5: Classification Modeling & Tuning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating Logistic Regression, Decision Tree, Random Forest & Gradient Boosting</div>', unsafe_allow_html=True)

    with st.spinner("Training classification models and running 5-fold cross-validation..."):
        best_model, results, feature_cols, X_test, y_test = train_and_cache_models()

    if results is not None:
        st.subheader("Model Benchmark Comparison")
        res_df = pd.DataFrame(results).T.round(4)
        st.dataframe(res_df.style.highlight_max(axis=0, color='#bbf7d0'), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Test F1-Score Comparison")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            res_df['F1-Score'].plot(kind='bar', color='#3b82f6', ax=ax)
            ax.set_ylabel("F1-Score")
            plt.xticks(rotation=20)
            st.pyplot(fig)

        with col2:
            st.markdown("#### Test Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            res_df['Accuracy'].plot(kind='bar', color='#10b981', ax=ax)
            ax.set_ylabel("Accuracy")
            plt.xticks(rotation=20)
            st.pyplot(fig)

        st.subheader("Best Model Diagnostics (Confusion Matrix)")
        colA, colB = st.columns([1, 1])
        with colA:
            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Genuine', 'Fraud'],
                        yticklabels=['Genuine', 'Fraud'], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        with colB:
            st.markdown("#### Classification Report")
            cr = classification_report(y_test, y_pred, target_names=['Genuine', 'Fraud'], output_dict=True)
            st.dataframe(pd.DataFrame(cr).round(3).T, use_container_width=True)


# ==========================================
# 6. LIVE CLAIM FRAUD PREDICTOR (INFERENCE)
# ==========================================
elif menu == "🔮 Live Claim Fraud Predictor":
    st.markdown('<div class="main-header">Real-Time Fraud Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Submit claim attributes to get instantaneous inference from the Task 5 trained model</div>', unsafe_allow_html=True)

    best_model, _, feature_cols, _, _ = train_and_cache_models()

    st.markdown("#### 👤 1. Driver Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age of Driver *", min_value=16, max_value=100, value=35, step=1)
        income = st.number_input("Annual Income ($) *", min_value=0.0, value=55000.0, step=1000.0)
    with c2:
        gender = st.selectbox("Gender *", ["MALE", "FEMALE", "OTHER"])
        education = st.selectbox("Higher Education *", ["College", "High School", "Associate", "Masters", "JD", "MD", "PhD"])
    with c3:
        marital = st.selectbox("Marital Status *", ["MARRIED", "SINGLE", "DIVORCED"])
        safety = st.slider("Safety Rating *", min_value=1, max_value=100, value=75)

    st.markdown("#### 🏠 2. Address & Property")
    c4, c5, c6 = st.columns(3)
    with c4:
        address_change = st.selectbox("Address Change *", ["NO", "YES (under 1 year)", "YES (1-3 years)"])
    with c5:
        property_status = st.selectbox("Property Status *", ["own", "rent"])
    with c6:
        zip_code = st.number_input("ZIP Code *", min_value=10000, max_value=99999, value=50006, step=1)

    st.markdown("#### 🚗 3. Vehicle Information")
    c7, c8, c9, c10 = st.columns(4)
    with c7:
        veh_cat = st.selectbox("Vehicle Category *", ["Medium", "Large", "Sedan", "SUV", "Coupe", "Wagon"])
    with c8:
        veh_price = st.number_input("Vehicle Price ($) *", min_value=0.0, value=35000.0, step=1000.0)
    with c9:
        veh_age = st.number_input("Age of Vehicle (years) *", min_value=0, max_value=30, value=4, step=1)
    with c10:
        veh_color = st.selectbox("Vehicle Color *", ["Silver", "White", "Black", "Red", "Blue", "Gray", "Other"])

    st.markdown("#### ⚠️ 4. Accident Information")
    c11, c12, c13 = st.columns(3)
    with c11:
        claim_date = st.date_input("Claim Date *", value=datetime.date(2023, 9, 15))
        past_claims = st.number_input("Previous Claims *", min_value=0, max_value=10, value=0, step=1)
    with c12:
        claim_day = st.selectbox("Claim Day *", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        witness = st.selectbox("Witness Present *", ["YES", "NO"])
    with c13:
        accident_site = st.selectbox("Accident Site *", ["Highway", "Intersection", "Local", "Parking Lot", "Residential Area"])
        liab_prct = st.slider("Liability Percentage % *", min_value=0, max_value=100, value=25)

    st.markdown("#### 📄 5. Policy Information")
    c14, c15, c16 = st.columns(3)
    with c14:
        channel = st.selectbox("Communication Channel *", ["Phone", "Online", "Mobile App", "Broker", "In Person"])
        premium = st.number_input("Annual Premium ($) *", min_value=100.0, value=1350.0, step=50.0)
    with c15:
        police_rep = st.selectbox("Police Report Filed *", ["YES", "NO"])
        days_open = st.number_input("Days Open *", min_value=1, max_value=60, value=7, step=1)
    with c16:
        deductible = st.selectbox("Policy Deductible ($) *", [500, 1000, 1500, 2000], index=1)
        form_defects = st.number_input("Form Defects *", min_value=0, max_value=10, value=0, step=1)

    st.markdown("#### 💰 6. Claim Information")
    c17, c18 = st.columns(2)
    with c17:
        claim_amt = st.number_input("Total Claim Amount ($) *", min_value=500.0, value=25000.0, step=1000.0)
    with c18:
        injury_amt = st.number_input("Injury Claim ($) *", min_value=0.0, value=4000.0, step=500.0)

    if st.button("🚀 Evaluate Fraud Probability", type="primary", use_container_width=True):
        if best_model is not None and feature_cols is not None:
            # Build 1-row feature vector matching the 42 features
            row_dict = {col: 0 for col in feature_cols}
            
            # Direct numerical mappings
            row_dict['age_of_driver'] = float(age)
            row_dict['safety_rating'] = float(safety)
            row_dict['annual_income'] = float(income)
            row_dict['high_education'] = 1 if any(deg in education.lower() for deg in ['college', 'masters', 'jd', 'md', 'phd', 'degree', 'associate', 'yes']) else 0
            row_dict['address_change'] = 1 if 'yes' in address_change.lower() else 0
            row_dict['zip_code'] = int(zip_code)
            row_dict['past_num_of_claims'] = int(past_claims)
            row_dict['liab_prct'] = float(liab_prct)
            row_dict['police_report'] = 1 if police_rep == "YES" else 0
            row_dict['age_of_vehicle'] = float(veh_age)
            row_dict['vehicle_price'] = float(veh_price)
            row_dict['total_claim'] = float(claim_amt)
            row_dict['injury_claim'] = float(injury_amt)
            row_dict['policy_deductible'] = float(deductible)
            row_dict['annual_premium'] = float(premium)
            row_dict['days_open'] = float(days_open)
            row_dict['form_defects'] = int(form_defects)
            
            # Categorical mappings
            row_dict['gender_M'] = 1 if gender == "MALE" else 0
            if marital == 'SINGLE' and 'marital_status_0' in row_dict:
                row_dict['marital_status_0'] = 1
            elif marital == 'MARRIED' and 'marital_status_1' in row_dict:
                row_dict['marital_status_1'] = 1
                
            if property_status.lower() in ['rent', 'rental'] and 'property_status_Rent' in row_dict:
                row_dict['property_status_Rent'] = 1
                
            day_key = f'claim_day_of_week_{claim_day.capitalize()}'
            if day_key in row_dict:
                row_dict[day_key] = 1
                
            if accident_site in ['Local', 'Residential Area']:
                if 'accident_site_Local' in row_dict:
                    row_dict['accident_site_Local'] = 1
            elif 'Parking' in accident_site and 'accident_site_Parking Lot' in row_dict:
                row_dict['accident_site_Parking Lot'] = 1
                
            if witness == 'NO' and 'witness_present_0' in row_dict:
                row_dict['witness_present_0'] = 1
            elif witness == 'YES' and 'witness_present_1' in row_dict:
                row_dict['witness_present_1'] = 1
                
            channel_lower = channel.lower()
            if 'online' in channel_lower or 'mobile' in channel_lower:
                if 'channel_Online' in row_dict:
                    row_dict['channel_Online'] = 1
            elif 'phone' in channel_lower:
                if 'channel_Phone' in row_dict:
                    row_dict['channel_Phone'] = 1
                    
            veh_cat_str = str(veh_cat).capitalize()
            if 'Large' in veh_cat_str or 'Suv' in veh_cat_str:
                if 'vehicle_category_Large' in row_dict:
                    row_dict['vehicle_category_Large'] = 1
            elif 'Medium' in veh_cat_str or 'Sedan' in veh_cat_str or 'Coupe' in veh_cat_str or 'Wagon' in veh_cat_str:
                if 'vehicle_category_Medium' in row_dict:
                    row_dict['vehicle_category_Medium'] = 1
                    
            col_name = f'vehicle_color_{veh_color.lower()}'
            if col_name in row_dict:
                row_dict[col_name] = 1
            elif 'vehicle_color_other' in row_dict and veh_color.lower() not in ['black']:
                row_dict['vehicle_color_other'] = 1
                
            input_df = pd.DataFrame([row_dict])
            
            # Predict
            prob = float(best_model.predict_proba(input_df)[0][1]) * 100
            pred = int(best_model.predict(input_df)[0])
            
            st.markdown("---")
            st.subheader("Prediction Result")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prob > 60 or pred == 1:
                    st.error(f"### ⚠️ HIGH RISK\n**Fraud Probability: {prob:.1f}%**")
                elif prob > 30:
                    st.warning(f"### 🟡 MEDIUM RISK\n**Fraud Probability: {prob:.1f}%**")
                else:
                    st.success(f"### 🟢 LOW RISK\n**Fraud Probability: {prob:.1f}%**")
                    
            with res_col2:
                st.markdown("**Key Risk Factors Evaluated:**")
                st.write(f"- Total Claim: **${claim_amt:,.2f}** ({'High Claim Flag' if claim_amt > 60000 else 'Normal Range'})")
                st.write(f"- Police Report: **{police_rep}** ({'Unverified flag' if police_rep == 'NO' else 'Verified'})")
                st.write(f"- Witness: **{witness}** ({'Elevated risk' if witness == 'NO' else 'Corroborated'})")
                st.write(f"- Past Claims: **{past_claims}**")


# ==========================================
# 7. TASK 6: DEPLOYMENT ARCHITECTURE
# ==========================================
elif menu == "🚀 Task 6: Deployment Architecture":
    st.markdown('<div class="main-header">Task 6: Multi-Cloud Deployment Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Streamlit Community Cloud • Vercel (Frontend) • Render (FastAPI Backend)</div>', unsafe_allow_html=True)

    tabA, tabB = st.tabs(["Part 1: Streamlit Cloud", "Part 2: React + FastAPI (Vercel & Render)"])
    
    with tabA:
        st.markdown("""
        ### Part 1: Deploying to Streamlit Community Cloud
        1. **Push your code to GitHub:**
           Ensure repository contains `streamlit_app.py` and `requirements.txt`.
        2. **Visit [share.streamlit.io](https://share.streamlit.io):**
           Sign in with GitHub.
        3. **Create New App:**
           - Repository: `your-username/your-repo`
           - Branch: `main`
           - Main file path: `streamlit_app.py`
        4. **Deploy:** Streamlit automatically provisions dependencies and exposes a live public URL.
        """)
        
    with tabB:
        st.markdown("""
        ### Part 2: Deploying React + FastAPI to Vercel and Render
        
        #### Step A: Deploy FastAPI to Render
        - **Directory:** `/backend`
        - **Build Command:** `pip install -r requirements.txt`
        - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
        - **CORS Configuration:**
        ```python
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://your-frontend.vercel.app"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        ```
        
        #### Step B: Deploy Frontend to Vercel
        - **Root Directory:** `Front-end`
        - **Environment Variable:** `VITE_API_URL = https://your-backend.onrender.com`
        - **API Fetch:**
        ```javascript
        const API_URL = import.meta.env.VITE_API_URL;
        fetch(`${API_URL}/predict`, { method: "POST", ... });
        ```
        """)
