import streamlit as st
import pandas as pd
from predictor import predict_customer
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Compare Customers", layout="wide")

st.title("👥 Customer Comparison")
st.write("Compare the predictions of two borrowers using the trained models.")


# ======================================================
# Customer Input Function
# ======================================================

def customer_input(prefix):

    loan_amnt = st.slider(
        f"{prefix} Loan Amount",
        min_value=1000,
        max_value=35000,
        value=10000,
        step=1000,
        key=f"{prefix}_loan"
    )

    term = st.selectbox(
        f"{prefix} Term",
        [' 36 months', ' 60 months'],
        key=f"{prefix}_term"
    )

    int_rate = st.slider(
        f"{prefix} Interest Rate",
        min_value=1.0,
        max_value=45.0,
        value=10.0,
        step=0.05,
        key=f"{prefix}_rate"
    )

    sub_grade = st.selectbox(
        f"{prefix} Sub Grade",
        ['A1','A2','A3','A4','A5',
         'B1','B2','B3','B4','B5',
         'C1','C2','C3','C4','C5',
         'D1','D2','D3','D4','D5',
         'E1','E2','E3','E4','E5',
         'F1','F2','F3','F4','F5',
         'G1','G2','G3','G4','G5'],
        key=f"{prefix}_grade"
    )

    home_ownership = st.selectbox(
        f"{prefix} Home Ownership",
        ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
        key=f"{prefix}_home"
    )

    annual_inc = st.number_input(
        f"{prefix} Annual Income",
        min_value=500,
        max_value=2000000,
        value=50000,
        step=500,
        key=f"{prefix}_income"
    )

    verification_status = st.selectbox(
        f"{prefix} Verification Status",
        ['Verified', 'Not Verified', 'Source Verified'],
        key=f"{prefix}_verify"
    )

    dti = st.slider(
        f"{prefix} DTI",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        key=f"{prefix}_dti"
    )

    delinq_2yrs = st.number_input(
        f"{prefix} Delinq 2 Years",
        min_value=0,
        max_value=40,
        value=0,
        key=f"{prefix}_delinq"
    )

    inq_last_6mths = st.number_input(
        f"{prefix} Inquiries Last 6 Months",
        min_value=0,
        max_value=40,
        value=0,
        key=f"{prefix}_inq"
    )

    total_acc = st.number_input(
        f"{prefix} Total Accounts",
        min_value=0,
        max_value=100,
        value=10,
        key=f"{prefix}_acc"
    )

    revol_util = st.slider(
        f"{prefix} Revolving Utilization",
        min_value=0.0,
        max_value=200.0,
        value=30.0,
        key=f"{prefix}_util"
    )

    revol_bal = st.number_input(
        f"{prefix} Revolving Balance",
        min_value=0,
        max_value=1000000,
        value=5000,
        key=f"{prefix}_bal"
    )

    open_acc = st.number_input(
        f"{prefix} Open Accounts",
        min_value=0,
        max_value=50,
        value=5,
        key=f"{prefix}_open"
    )

    pub_rec = st.number_input(
        f"{prefix} Public Records",
        min_value=0,
        max_value=50,
        value=0,
        key=f"{prefix}_pub"
    )

    mths_since_last_delinq = st.number_input(
        f"{prefix} Months Since Last Delinquency",
        min_value=0,
        max_value=100,
        value=0,
        key=f"{prefix}_msd"
    )

    mths_since_last_record = st.number_input(
        f"{prefix} Months Since Last Record",
        min_value=0,
        max_value=100,
        value=0,
        key=f"{prefix}_msr"
    )

    mths_since_last_major_derog = st.number_input(
        f"{prefix} Months Since Last Major Derogatory",
        min_value=0,
        max_value=100,
        value=0,
        key=f"{prefix}_major"
    )

    initial_list_status = st.selectbox(
        f"{prefix} Initial List Status",
        ['f', 'w'],
        key=f"{prefix}_status"
    )

    emp_status_unverifiable = st.selectbox(
        f"{prefix} Employment Status",
        [0, 1],
        key=f"{prefix}_emp"
    )

    purpose_group = st.selectbox(
        f"{prefix} Purpose",
        [
            'Debt Management',
            'Home & Property',
            'Personal Expenses',
            'Business & Education',
            'Vehicle & Major Purchases'
        ],
        key=f"{prefix}_purpose"
    )

    delinq_2yrs_flag = st.selectbox(
        f"{prefix} Delinquency Flag",
        [0, 1],
        key=f"{prefix}_flag"
    )

    return pd.DataFrame({
        "loan_amnt": [loan_amnt],
        "term": [term],
        "int_rate": [int_rate],
        "sub_grade": [sub_grade],
        "home_ownership": [home_ownership],
        "annual_inc": [annual_inc],
        "verification_status": [verification_status],
        "dti": [dti],
        "delinq_2yrs": [delinq_2yrs],
        "inq_last_6mths": [inq_last_6mths],
        "total_acc": [total_acc],
        "revol_util": [revol_util],
        "revol_bal": [revol_bal],
        "open_acc": [open_acc],
        "pub_rec": [pub_rec],
        "mths_since_last_delinq": [mths_since_last_delinq],
        "mths_since_last_record": [mths_since_last_record],
        "mths_since_last_major_derog": [mths_since_last_major_derog],
        "initial_list_status": [initial_list_status],
        "emp_status_unverifiable": [emp_status_unverifiable],
        "purpose_group": [purpose_group],
        "delinq_2yrs_flag": [delinq_2yrs_flag]
    })

# ======================================================
# Customer Inputs
# ======================================================

st.header("Enter Borrower Details")

tab1, tab2 = st.tabs(["👤 Customer A", "👤 Customer B"])

with tab1:
    st.subheader("Customer A Information")
    customerA = customer_input("A")

with tab2:
    st.subheader("Customer B Information")
    customerB = customer_input("B")


# ======================================================
# Compare Button
# ======================================================

compare = st.button(
    "🔍 Compare Customers",
    key="compare_btn",
    use_container_width=True
)

# ======================================================
# Compare Customers
# ======================================================

if compare:

    # ======================================================
    # Predictions
    # ======================================================

    resultA = predict_customer(customerA)
    resultB = predict_customer(customerB)

    # ======================================================
    # Dashboard Summary
    # ======================================================

    st.header("🏆 Dashboard Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Safer Borrower",
            "Customer A"
            if resultA["xgb_risk"] < resultB["xgb_risk"]
            else "Customer B"
        )

    with col2:
        st.metric(
            "Difference in XGBoost Risk",
            f"{abs(resultA['xgb_risk'] - resultB['xgb_risk']):.3f}"
        )

    # ======================================================
    # Prediction Comparison
    # ======================================================

    st.header("📊 Prediction Comparison")

    comparison = pd.DataFrame({
        "Metric": [
            "LR Default Probability (%)",
            "GB Default Probability (%)",
            "Cox Hazard Ratio",
            "XGBoost Risk Score"
        ],
        "Customer A": [
            round(resultA["lr_probability"], 2),
            round(resultA["gb_probability"], 2),
            round(resultA["cox_risk"], 3),
            round(resultA["xgb_risk"], 3)
        ],
        "Customer B": [
            round(resultB["lr_probability"], 2),
            round(resultB["gb_probability"], 2),
            round(resultB["cox_risk"], 3),
            round(resultB["xgb_risk"], 3)
        ]
    })

    st.dataframe(comparison, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Safer Customer (LR)",
            "Customer A"
            if resultA["lr_probability"] < resultB["lr_probability"]
            else "Customer B"
        )

    with col2:
        st.metric(
            "Safer Customer (Cox)",
            "Customer A"
            if resultA["cox_risk"] < resultB["cox_risk"]
            else "Customer B"
        )

    # ======================================================
    # Survival Curve Comparison
    # ======================================================

    st.header("📈 Survival Curve Comparison")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        resultA["survival_curve"].index,
        resultA["survival_curve"].iloc[:, 0],
        linewidth=2,
        label="Customer A"
    )

    ax.plot(
        resultB["survival_curve"].index,
        resultB["survival_curve"].iloc[:, 0],
        linewidth=2,
        label="Customer B"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Predicted Survival Curves")
    ax.grid(alpha=0.3)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.info("""
**Interpretation**

- A higher survival curve indicates a lower probability of early default.
- A rapidly declining curve indicates higher credit risk.
- The borrower whose curve remains higher over time is expected to survive longer.
""")

    # ======================================================
    # Logistic Regression SHAP
    # ======================================================

    st.markdown("---")
    st.header("🔍 Logistic Regression SHAP Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer A")

        shap.plots.waterfall(
            resultA["lr_shap"][0],
            max_display=10,
            show=False
        )

        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Customer B")

        shap.plots.waterfall(
            resultB["lr_shap"][0],
            max_display=10,
            show=False
        )

        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    # ======================================================
    # Gradient Boosting SHAP
    # ======================================================

    st.markdown("---")
    st.header("🌳 Gradient Boosting SHAP Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer A")

        shap.plots.waterfall(
            resultA["gb_shap"][0],
            max_display=10,
            show=False
        )

        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Customer B")

        shap.plots.waterfall(
            resultB["gb_shap"][0],
            max_display=10,
            show=False
        )

        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    # ======================================================
    # XGBoost SHAP
    # ======================================================

    st.markdown("---")
    st.header("🌲 XGBoost Survival SHAP Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer A")

        shap.plots.waterfall(
            resultA["xgb_shap"][0],
            max_display=10,
            show=False
        )

        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Customer B")

        shap.plots.waterfall(
            resultB["xgb_shap"][0],
            max_display=10,
            show=False
        )

        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    # ======================================================
    # Overall Summary
    # ======================================================

    st.markdown("---")
    st.header("🏆 Overall Comparison Summary")

    if resultA["lr_probability"] < resultB["lr_probability"]:
        st.success("✔ Logistic Regression: Customer A is predicted to be safer.")
    else:
        st.success("✔ Logistic Regression: Customer B is predicted to be safer.")

    if resultA["gb_probability"] < resultB["gb_probability"]:
        st.success("✔ Gradient Boosting: Customer A is predicted to be safer.")
    else:
        st.success("✔ Gradient Boosting: Customer B is predicted to be safer.")

    if resultA["cox_risk"] < resultB["cox_risk"]:
        st.success("✔ Cox Proportional Hazards: Customer A has the lower hazard ratio.")
    else:
        st.success("✔ Cox Proportional Hazards: Customer B has the lower hazard ratio.")

    if resultA["xgb_risk"] < resultB["xgb_risk"]:
        st.success("✔ XGBoost Survival: Customer A has the lower predicted risk score.")
    else:
        st.success("✔ XGBoost Survival: Customer B has the lower predicted risk score.")
