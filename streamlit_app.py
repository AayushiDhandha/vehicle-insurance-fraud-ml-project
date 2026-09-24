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
st.sidebar.caption("Vehicle Insurance Fraud Detection")

menu = st.sidebar.radio(
    "Navigate:",
    [
        "🔮 Live Claim Fraud Predictor"
    ]
)



# ==========================================
# LIVE CLAIM FRAUD PREDICTOR (INFERENCE)
# ==========================================
if menu == "🔮 Live Claim Fraud Predictor":
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


