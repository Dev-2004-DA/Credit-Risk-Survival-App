import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import gdown

# ==========================================================
# Load preprocessing pipeline
# ==========================================================

preprocessing = joblib.load(r'models/Ct_for_preprocessing.csv')

# ==========================================================
# Classification models
# ==========================================================

lr = joblib.load(r'models/Final_lr_model.pkl')

gb = joblib.load(r'models/Final_gb_model.pkl')

# ==========================================================
# Survival models
# ==========================================================

os.makedirs("models", exist_ok=True)
if not os.path.exists("models/cox_model.pkl"):
    gdown.download(
        id="1HLCFDl17JL69qyzMWPaUfVG4vwFaM-h8",
        output="models/cox_model.pkl",
        quiet=False
    )

cox_model = joblib.load("models/cox_model.pkl")

xgb_model = joblib.load(r'models/xgb_model.pkl')

# ==========================================================
# Column transformers
# ==========================================================
cox_model_ct = joblib.load(r'models/CT_cox_model.pkl')

xgb_model_ct = joblib.load(r'models/CT_xgb_model.pkl')

# ==========================================================
# SHAP masker data
# ==========================================================

masker_data = pd.read_csv(r'masker_data_for_lr_pd.csv')


masker_data = pd.DataFrame(
    preprocessing.transform(masker_data),
    columns=preprocessing.get_feature_names_out()
)

masker = shap.maskers.Independent(masker_data,max_samples=100)

# ==========================================================
# Prediction Function
# ==========================================================

def predict_customer(input_data):

    results = {}

    # Make copies of input for different models
    lr_model_input = input_data.copy()
    cox_model_input = input_data.copy()

    # ======================================================
    # Logistic Regression preprocessing
    # ======================================================

    lr_model_input['bin_annual_inc'] = pd.cut(lr_model_input['annual_inc'],
        bins=[-1, 18000, 40000, 80000, 120000, float('inf')],labels=['<18K', '18K-40K', '40K-80K', '80K-120K', '>120K'],
        include_lowest=True
    )

    lr_model_input['inq_last_6mths_bin'] = pd.cut(lr_model_input['inq_last_6mths'],
        bins=[-1, 0, 1, 2, float('inf')],labels=['0', '0-1', '1-2', '2+'],
        include_lowest=True
    )

    lr_model_input['total_acc_bin'] = pd.cut(lr_model_input['total_acc'],
        bins=[-1, 10, 16, 20, 26, 35, float('inf')],labels=['0-10', '10-16', '16-20', '20-26', '26-35', '35+'],
        include_lowest=True
    )

    lr_model_input['bin_int_ratec'] = pd.cut(lr_model_input['int_rate'],
        bins=[-1, 8, 12, 15, 21, 24, float('inf')], labels=['<8%', '8-12%', '12-15%', '15-21%', '21-24%', '>24%'],
        include_lowest=True
    )

    lr_model_input['bin_dti'] = pd.cut(lr_model_input['dti'],
        bins=[-1, 5, 10, 15, 20, 25, 30, 35, float('inf')],labels=['<5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '>35'],
        include_lowest=True
    )

    lr_model_input['bin_revol_util'] = pd.cut(lr_model_input['revol_util'],
        bins=[-1, 20, 40, 60, 80, 100, float('inf')],labels=['<20%', '20-40%', '40-60%', '60-80%', '80-100%', '>100%'],
        include_lowest=True
    )

    lr_model_input.drop(['annual_inc', 'inq_last_6mths', 'total_acc', 'int_rate', 'dti', 'revol_util'],axis=1,inplace=True)

    # ======================================================
    # Logistic Regression & Gradient Boosting Prediction
    # ======================================================

    # Transform input
    X_lr = preprocessing.transform(lr_model_input)

    # Predictions
    lr_def_pred = lr.predict_proba(X_lr)[:, 1][0] * 100
    gb_def_pred = gb.predict_proba(X_lr)[:, 1][0] * 100

    # Store predictions
    results["lr_probability"] = lr_def_pred
    results["gb_probability"] = gb_def_pred

        # ======================================================
    # SHAP Values (LR & GB)
    # ======================================================

    X_trans = pd.DataFrame(X_lr, columns=preprocessing.get_feature_names_out())

    shap_lr = shap.LinearExplainer(lr, masker)(X_trans)

    shap_gb = shap.TreeExplainer(gb)(X_trans)

    results["lr_shap"] = shap_lr
    results["gb_shap"] = shap_gb

        # ======================================================
    # Cox Proportional Hazards Model
    # ======================================================

    # Collapse sub_grade into letter groups
    for grade in ['A','B','C','D','E','F','G']:
        cox_model_input.loc[
            cox_model_input['sub_grade'].str.startswith(grade),
            'sub_grade'
        ] = grade

    # Rename purpose labels
    cox_model_input.loc[
        cox_model_input['purpose_group'] == "Debt Management",
        'purpose_group'
    ] = "Debt_Management"

    cox_model_input.loc[
        cox_model_input['purpose_group'] == "Business & Education",
        'purpose_group'
    ] = "Business_Education"

    cox_model_input.loc[
        cox_model_input['purpose_group'] == "Personal Expenses",
        'purpose_group'
    ] = "Personal"

    cox_model_input.loc[
        cox_model_input['purpose_group'] == "Home & Property",
        'purpose_group'
    ] = "Asset_Home"

    cox_model_input.loc[
        cox_model_input['purpose_group'] == "Vehicle & Major Purchases",
        'purpose_group'
    ] = "Major_Purchase"

    # Rename home ownership
    cox_model_input.loc[
        cox_model_input["home_ownership"].isin(["OTHER","OWN"]),
        "home_ownership"
    ] = "NON_MORTGAGE"

    # Clean variables
    cox_model_input["term"] = cox_model_input["term"].str.strip()

    cox_model_input.rename(columns={"emp_status_unverifiable":"Employement"},inplace=True)

    # Transform
    cox_transformed = pd.DataFrame(cox_model_ct.transform(cox_model_input),columns=cox_model_ct.get_feature_names_out())

    # Hazard Ratio
    cox_risk = cox_model.predict_partial_hazard(cox_transformed)[0]

    # Survival Curve
    surv_func = cox_model.predict_survival_function(cox_transformed)

    # Store results
    results["cox_risk"] = cox_risk
    results["survival_curve"] = surv_func

        # ======================================================
    # XGBoost Survival Model
    # ======================================================

    # Prepare input
    xgb_model_input = cox_model_input.copy()

    xgb_transformed = pd.DataFrame(xgb_model_ct.transform(xgb_model_input), columns=xgb_model_ct.get_feature_names_out())

    # Predict risk score
    xgb_risk = xgb_model.predict(xgb_transformed)[0]

    # SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(xgb_transformed)

    # Raw score
    raw_score = (shap_values.base_values[0] + shap_values.values[0].sum())

    # Store results
    results["xgb_risk"] = xgb_risk
    results["xgb_shap"] = shap_values
    results["raw_score"] = raw_score

    return results
