import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import lifelines
import seaborn as sns
import numpy as np 
import os
import gdown

# import
# Preprocessing pipeline
preprocessing = joblib.load('models/Ct_for_preprocessing.pkl')

#Models
lr = joblib.load('models/Final_lr_model.pkl')
gb = joblib.load('models/Final_gb_model.pkl')

st.title("🏦 Credit Risk Analytics Dashboard")
st.markdown(
    """
This application predicts a borrower's **probability of default** using **Logistic Regression** and **Gradient Boosting** models,
and estimates the **default risk over time** using **Cox Proportional Hazards** and **XGBoost Survival** models.
The dashboard also provides **SHAP-based model explainability** to highlight the key factors influencing each prediction.
"""
)
# Inputs 
loan_amnt = st.sidebar.slider('Loan Amount', min_value=1000, max_value=35000, value=10000, step=1000)
term = st.sidebar.selectbox('Term', options=[' 36 months', ' 60 months'])
int_rate = st.sidebar.slider('Interest Rate (%)', min_value=1.0, max_value=45.0, value=3.5, step=0.05 )
sub_grade = st.sidebar.selectbox('Sub_Grade', options= ['A1','A2','A3','A4','A5',
                                                        'B1','B2','B3','B4','B5',
                                                        'C1','C2','C3','C4','C5',
                                                        'D1','D2','D3','D4','D5',
                                                        'E1','E2','E3','E4','E5',
                                                        'F1','F2','F3','F4','F5',
                                                        'G1','G2','G3','G4','G5'])

home_ownership = st.sidebar.selectbox('Home Ownership', options= ['RENT','OWN', 'MORTGAGE', 'OTHER'])
annual_inc = st.sidebar.number_input('Annual Income', min_value = 500, max_value=2000000, step = 500, value = 50000)
verification_status = st.sidebar.selectbox('Verification Status', options= [ 'Verified', 'Not Verified', 'Source Verified']) 
dti = st.sidebar.slider('Debt to Income Ratio', min_value=0.0, max_value=100.0, value=20.5, step=0.01 )
delinq_2yrs = st.sidebar.number_input('Delinquency in 2 years', min_value = 0, max_value=40, step = 1, value = 5)
inq_last_6mths  = st.sidebar.number_input('Inquire in last 6 months', min_value = 0, max_value= 40, step = 1, value = 5)
total_acc = st.sidebar.number_input('Total number of accounts', min_value = 0, max_value=50, step = 1, value = 5)
revol_util = st.sidebar.slider('Revolving utilizatiation', min_value=0.0, max_value=200.0, value=20.5, step=0.01 )
revol_bal = st.sidebar.number_input('Revolving balance', min_value = 0, max_value= 1000000, step = 100, value = 5000)
open_acc = st.sidebar.number_input('Total number of open accounts', min_value = 0, max_value=50, step = 1, value = 5)
pub_rec = st.sidebar.number_input('Total number of public records', min_value = 0, max_value=50, step = 1, value = 5)
mths_since_last_delinq = st.sidebar.number_input('Month since last delinquency', min_value = 0, max_value=50, step = 1, value = 5)
mths_since_last_record  = st.sidebar.number_input('Month since last public record', min_value = 0, max_value=50, step = 1, value = 5)
mths_since_last_major_derog = st.sidebar.number_input('Month since last derogatry remark', min_value = 0, max_value=50, step = 1, value = 5)
initial_list_status = st.sidebar.selectbox('Initial list status', options = ['f','w'])
emp_status_unverifiable = st.sidebar.selectbox('Is Employed , 0 = no m 1 - yes', options = [0,1])
purpose_group = st.sidebar.selectbox('Purpose of Loan', options = ['Debt Management', 'Home & Property',  'Personal Expenses',  'Business & Education', 'Vehicle & Major Purchases'])
delinq_2yrs_flag = st.sidebar.selectbox('Delinquency in last 2 year? , 1 = yes , 0 no ', options = [0,1])

input_data = pd.DataFrame({
    'loan_amnt': [loan_amnt], 'term': [term], 'int_rate': [int_rate], 'sub_grade': [sub_grade],
    'home_ownership': [home_ownership], 'annual_inc': [annual_inc], 'verification_status': [verification_status],
    'dti': [dti], 'delinq_2yrs': [delinq_2yrs], 'inq_last_6mths': [inq_last_6mths], 'total_acc': [total_acc],
    'revol_util': [revol_util], 'revol_bal': [revol_bal], 'open_acc': [open_acc], 'pub_rec': [pub_rec],
    'mths_since_last_delinq': [mths_since_last_delinq], 'mths_since_last_record': [mths_since_last_record],
    'mths_since_last_major_derog': [mths_since_last_major_derog], 'initial_list_status': [initial_list_status],
    'emp_status_unverifiable': [emp_status_unverifiable], 'purpose_group': [purpose_group], 'delinq_2yrs_flag': [delinq_2yrs_flag]
})


# Default prediction LR model
lr_model_input = input_data.copy()

lr_model_input['bin_annual_inc'] = pd.cut(lr_model_input['annual_inc'],
    bins=[-1, 18000, 40000, 80000, 120000, float('inf')], labels=['<18K', '18K-40K', '40K-80K', '80K-120K', '>120K'],
    include_lowest=True
)

lr_model_input['inq_last_6mths_bin'] = pd.cut(lr_model_input['inq_last_6mths'],
    bins=[-1, 0, 1, 2, float('inf')], labels=['0', '0-1', '1-2', '2+'],
    include_lowest=True
)

lr_model_input['total_acc_bin'] = pd.cut(lr_model_input['total_acc'],
    bins=[-1, 10, 16, 20, 26, 35, float('inf')], labels=['0-10', '10-16', '16-20', '20-26', '26-35', '35+'],
    include_lowest=True
)

lr_model_input['bin_int_ratec'] = pd.cut(lr_model_input['int_rate'],
    bins=[-1, 8, 12, 15, 21, 24, float('inf')], labels=['<8%', '8-12%', '12-15%', '15-21%', '21-24%', '>24%'],
    include_lowest=True
)

lr_model_input['bin_dti'] = pd.cut(lr_model_input['dti'],
    bins=[-1, 5, 10, 15, 20, 25, 30, 35, float('inf')], labels=['<5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '>35'],
    include_lowest=True
)

lr_model_input['bin_revol_util'] = pd.cut(lr_model_input['revol_util'],
    bins=[-1, 20, 40, 60, 80, 100, float('inf')], labels=['<20%', '20-40%', '40-60%', '60-80%', '80-100%', '>100%'],
    include_lowest=True
)

# Drop original continuous variables
lr_model_input.drop(['annual_inc', 'inq_last_6mths', 'total_acc', 'int_rate', 'dti', 'revol_util'], axis=1, inplace=True)

# Predictions
lr_def_pred = lr.predict_proba(preprocessing.transform(lr_model_input))[:,1][0] * 100
gb_def_pred = gb.predict_proba(preprocessing.transform(lr_model_input))[:,1][0] * 100

st.header("📊 Default Risk Prediction")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Logistic Regression")

    st.metric(
        label="Predicted Default Probability",
        value=f"{lr_def_pred:.2f}%"
    )

    st.progress(float(lr_def_pred / 100))

    st.markdown("#### Model Performance")
    st.table(pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "ROC-AUC"],
        "Value": ["71.83%", "28.21%", "59.99%", "70.06%"]
    }))

with col2:
    st.subheader("Gradient Boosting")

    st.metric(
        label="Predicted Default Probability",
        value=f"{gb_def_pred:.2f}%"
    )

    st.progress(float(gb_def_pred / 100))

    st.markdown("#### Model Performance")
    st.table(pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "ROC-AUC"],
        "Value": ["65.97%", "29.04%", "62.98%", "69.61%"]
    }))

    st.markdown("---")
st.header("🔍 Global Feature Importance (SHAP)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Logistic Regression")
    st.image(
        "images/lr_shap_bar.png",
        caption="Global SHAP Feature Importance - Logistic Regression",
        use_container_width=True
    )

with col2:
    st.subheader("Gradient Boosting")
    st.image(
        "images/gb_shap_bar.png",
        caption="Global SHAP Feature Importance - Gradient Boosting",
        use_container_width=True
    )

# Compute SHAP values
# For masker


input = pd.DataFrame(preprocessing.transform(lr_model_input), columns=preprocessing.get_feature_names_out())
X_train = pd.read_csv('masker_data_for_lr_pd.csv')
X_train = pd.DataFrame( preprocessing.transform(X_train), columns = preprocessing.get_feature_names_out())
masker = shap.maskers.Independent(X_train ,max_samples=1000)
shap_lr = shap.LinearExplainer(lr, masker)(input)
shap_gb = shap.TreeExplainer(gb)(input)

# Waterfall plots
fig, axes = plt.subplots(2, 1, figsize=(10, 14))

# Logistic Regression SHAP
plt.sca(axes[0])

masker = shap.maskers.Independent(X_train, max_samples=100)
shap.plots.waterfall(shap_lr[0], max_display=10, show=False)
axes[0].set_title("Logistic Regression SHAP Waterfall")

# Gradient Boosting SHAP
plt.sca(axes[1])
shap.plots.waterfall(shap_gb[0], max_display=10, show=False)
axes[1].set_title("Gradient Boosting SHAP Waterfall")

plt.tight_layout()

# Display in Streamlit
st.header("🔍 Model Explainability")
st.pyplot(fig)


st.info(
    """
**Note:** The value **f(x)** shown in the SHAP waterfall plot represents the **log-odds (logit)** output of the Logistic Regression model, not the predicted probability.  
The corresponding probability is computed using the sigmoid function:

[ PD = 1/(1 + exp(-f(x)))]



where **f(x)** is the final SHAP output obtained by summing the base value and all feature contributions.
"""
)


# ===========================
# Survival Analysis (Cox PH)
# ===========================
# models
os.makedirs("models", exist_ok=True)
if not os.path.exists("models/cox_model.pkl"):
    gdown.download(
        id="1HLCFDl17JL69qyzMWPaUfVG4vwFaM-h8",
        output="models/cox_model.pkl",
        quiet=False
    )

cox_model = joblib.load("models/cox_model.pkl")
xgb_model = joblib.load('models/xgb_model.pkl')
# CT
cox_model_ct = joblib.load('models/CT_cox_model.pkl')
xgb_model_ct = joblib.load('models/CT_xgb_model.pkl')

st.markdown("---")
st.header("📈 Survival Risk Prediction")

# Copy input
cox_model_input = input_data.copy()

# Collapse sub_grade into letter groups
for grade in ['A','B','C','D','E','F','G']:
    cox_model_input.loc[
        cox_model_input['sub_grade'].str.startswith(grade),
        'sub_grade'
    ] = grade

# Rename purpose labels
cox_model_input.loc[cox_model_input['purpose_group']=="Debt Management","purpose_group"]="Debt_Management"
cox_model_input.loc[cox_model_input['purpose_group']=="Business & Education","purpose_group"]="Business_Education"
cox_model_input.loc[cox_model_input['purpose_group']=="Personal Expenses","purpose_group"]="Personal"
cox_model_input.loc[cox_model_input['purpose_group']=="Home & Property","purpose_group"]="Asset_Home"
cox_model_input.loc[cox_model_input['purpose_group']=="Vehicle & Major Purchases","purpose_group"]="Major_Purchase"

# Rename home ownership
cox_model_input.loc[
    cox_model_input["home_ownership"].isin(["OTHER","OWN"]),
    "home_ownership"
] = "NON_MORTGAGE"

# Clean variables
cox_model_input["term"] = cox_model_input["term"].str.strip()
cox_model_input.rename(
    columns={"emp_status_unverifiable":"Employement"},
    inplace=True
)

# Transform
cox_transformed = pd.DataFrame(
    cox_model_ct.transform(cox_model_input),
    columns=cox_model_ct.get_feature_names_out()
)

# Prediction
cox_risk = cox_model.predict_partial_hazard(cox_transformed)[0]

# Display
col1, col2 = st.columns([1,2])

with col1:

    st.metric(
        label="Cox Hazard Ratio",
        value=f"{cox_risk:.3f}"
    )

    st.markdown("""
**Interpretation**

- Hazard Ratio = **1** → Average Risk
- Hazard Ratio > **1** → Higher Default Risk
- Hazard Ratio < **1** → Lower Default Risk
""")

with col2:

    surv_func = cox_model.predict_survival_function(cox_transformed)

    fig, ax = plt.subplots(figsize=(7,5))

    sns.lineplot(
        x=surv_func.index,
        y=surv_func.iloc[:,0],
        ax=ax
    )

    ax.set_title("Predicted Survival Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    plt.show()
    
st.subheader("📌 Key Hazard Ratio Interpretations")

st.markdown("""
Based on the fitted Cox Proportional Hazards model, the following variables have the largest impact on the borrower's hazard of default:

- **Interest Rate (+70.15%)**  
  Increasing the interest rate substantially increases the borrower's hazard of default, making it the strongest predictor in the model.

- **Annual Income (−19.41%)**  
  Higher annual income reduces the hazard of default, indicating that financially stronger borrowers are less likely to default.

- **Home Ownership – RENT (+12.21%)**  
  Borrowers living in rented accommodation have a higher hazard of default than the reference home ownership category.

- **Loan Amount (+12.01%)**  
  Larger loan amounts are associated with an increased hazard of default, suggesting greater repayment risk.

- **Recent Credit Inquiries (+11.60%)**  
  A higher number of credit inquiries in the last six months increases the hazard of default, reflecting higher credit-seeking behaviour.
""")

st.info("""
**Note:** The percentages above represent the percentage change in the **hazard of default** for a one-unit increase in the corresponding covariate (or relative to the reference category), while keeping all other variables constant.
""")

st.info(
    """
**Note:** The Cox Proportional Hazards model estimates the borrower's **relative hazard (hazard ratio)** compared with the baseline borrower.
The accompanying survival curve represents the predicted probability that the borrower **has not defaulted** up to each point in time.
"""
)

# ============================================================
#               XGBoost Survival Model Prediction
# ============================================================

st.markdown("---")
st.header("🌲 XGBoost Survival Prediction")

# Prepare input
xgb_model_input = cox_model_input.copy()

xgb_transformed = pd.DataFrame(
    xgb_model_ct.transform(xgb_model_input),
    columns=xgb_model_ct.get_feature_names_out()
)

# Predict risk score
xgb_risk = xgb_model.predict(xgb_transformed)[0]

# Compute SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(xgb_transformed)

# Compute raw score
raw_score = shap_values.base_values[0] + shap_values.values[0].sum()

# ------------------------------------------------------------
# Prediction Summary
# ------------------------------------------------------------

col1, col2 = st.columns([1, 2])

with col1:

    st.metric(
        label="Predicted Risk Score",
        value=f"{xgb_risk:.3f}"
    )

    st.markdown("### Model Performance")

    perf = pd.DataFrame({
        "Metric": ["Concordance Index"],
        "Value": ["68.33%"]
    })

    st.table(perf)

    st.markdown("### Interpretation")

    st.markdown("""
- Higher **Risk Score** indicates a greater likelihood of experiencing the event (default) earlier.
- Lower **Risk Score** indicates a lower relative default risk.
- The predicted value is a **relative risk score** and **not** a probability of default.
""")

# ------------------------------------------------------------
# SHAP Waterfall Plot
# ------------------------------------------------------------

with col2:

    shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False
    )

    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------
# SHAP Explanation
# ------------------------------------------------------------

st.subheader("🔍 SHAP Prediction Explanation")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Raw Score (f(x))",
        f"{raw_score:.3f}"
    )

with m2:
    st.metric(
        "exp(f(x))",
        f"{np.exp(raw_score):.3f}"
    )

with m3:
    st.metric(
        "Predicted Risk Score",
        f"{xgb_risk:.3f}"
    )

st.info("""
**Note:** The SHAP waterfall plot explains the model's **raw prediction** \(f(x)\), also known as the **log-risk score**.

The final XGBoost Survival prediction is obtained by exponentiating the raw score:

**Risk Score = exp(f(x))**

Therefore,

- **f(x)** displayed in the waterfall plot is the raw model output.
- **exp(f(x))** converts the raw score into the final relative risk score.
- The **Predicted Risk Score** equals **exp(f(x))**, confirming the correctness of the prediction.
""")

st.info("""
**Note:** Although both the **Cox Proportional Hazards** and **XGBoost Survival** models produce a **risk score**, the interpretation of these scores is different.

- **Cox Proportional Hazards:** The predicted risk score is a **Hazard Ratio**, representing how many times the borrower's instantaneous default risk changes relative to the baseline borrower. For example, a hazard ratio of **2** indicates twice the baseline hazard, while **0.5** indicates half the baseline hazard.

- **XGBoost Survival:** The predicted risk score is a **relative ranking score** learned by the model. It is useful for comparing borrowers (higher score implies higher relative risk), but it **does not represent a hazard ratio** and cannot be interpreted as a multiple of the baseline risk.

Therefore, while **higher values indicate greater default risk in both models**, the numerical values from the Cox model and XGBoost Survival model **should not be compared directly** because they represent different quantities.
""")
# python -m streamlit run app.py
#  C:\Users\Dev\OneDrive\Desktop\data\Streamlite
