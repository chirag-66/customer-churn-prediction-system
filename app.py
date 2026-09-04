import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import PIL.Image as Image

page_icon = Image.open("C:\\Users\\chira\\OneDrive\\Desktop\\customer-churn-predictor\\Screenshot 2026-09-04 142720.png")

project_root = Path().cwd()

model = joblib.load(project_root/"models"/"churn_model.pkl")
threshold = joblib.load(project_root/"models"/"threshold.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon=page_icon,
    layout="wide"
)



model = joblib.load(project_root/"models"/"churn_model.pkl")
threshold = joblib.load(project_root/"models"/"threshold.pkl")


with st.sidebar:
    st.image("https://images.openai.com/static-rsc-4/k1s6bTN_gFnD3MKQTShmDjRx4DvD_DmkLhnzTJgHgB8u15l-NQ0kXbjr8xp1bjNwS1PuGyCEMpCXChsg9JwsJ03TSQqpuDpfMu_75Y455E0Rnki2Fv1iuUOqNDJV1iigpwammylTNSKHE_mEPUMUpMJdGPRqUkDKa_ORFgm8p_zUCKov-tMPpPYrOBnAD_eG?purpose=fullsize",
             width=50,)
    st.header("Customer Churn Prediction System")

    page = st.sidebar.radio(
    "Navigation",
    ["Dashboard","Prediction"]
    )


    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** XGBoost")
    st.write(f"**Decision Threshold:** {threshold:.2f}")
    st.write("**Explainability:** SHAP")

    st.divider()

    st.subheader("Model Performance")
    st.write("**ROC-AUC:** 0.850")
    st.write("**Recall:** 0.818")
    st.write("**F1 Score:** 0.639")
    st.write("**Accuracy:** 0.755")

    st.divider()

    st.caption(
        "Customer Churn Prediction System"
    )

    if st.button("Reset"):
        st.rerun()


st.image("https://images.openai.com/static-rsc-4/k1s6bTN_gFnD3MKQTShmDjRx4DvD_DmkLhnzTJgHgB8u15l-NQ0kXbjr8xp1bjNwS1PuGyCEMpCXChsg9JwsJ03TSQqpuDpfMu_75Y455E0Rnki2Fv1iuUOqNDJV1iigpwammylTNSKHE_mEPUMUpMJdGPRqUkDKa_ORFgm8p_zUCKov-tMPpPYrOBnAD_eG?purpose=fullsize",
             width=50,)
st.title("Customer Churn Prediction System")


# Customer Dashboard
st.divider()

if page == "Dashboard":
    st.set_page_config(
    page_title="Churn Dashboard",
    layout="wide"
)

    st.title("Customer Churn Dashboard")
    st.caption("Customer behavior and model insights")

# Load dataset
    df = pd.read_csv(
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

# Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    st.divider()

    total_customers = len(df)

    churned_customers = (df["Churn"] == "Yes").sum()

    churn_rate = (churned_customers / total_customers) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
        "Total Customers",
        f"{total_customers:,}"
        )

    with col2:
        st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
        )

    with col3:
        st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
        )

    with col4:
        st.metric(
        "Model ROC-AUC",
        "0.850"
        )

    st.subheader("Customer Churn Distribution")

    churn_counts = df["Churn"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(churn_counts.index,churn_counts.values)

    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Churn by Contract Type")

    contract_churn = pd.crosstab(df["Contract"],df["Churn"])

    st.bar_chart(contract_churn)

    st.subheader("Churn by Internet Service")

    internet_churn = pd.crosstab(df["InternetService"],df["Churn"])

    st.bar_chart(internet_churn)

    st.subheader("Churn by Tenure")

    st.line_chart(df.groupby("tenure")["Churn"].value_counts(normalize=True).unstack().fillna(0))

    st.subheader("Tenure vs Total Charges")

    
    fig = go.Figure()

    for churn_status in df["Churn"].unique():
        df_churn = df[df["Churn"] == churn_status]
        fig.add_trace(go.Scatter(x=df_churn["tenure"], y=df_churn["TotalCharges"], mode="markers", name=churn_status))

    fig.update_layout(title="Tenure vs Total Charges", xaxis_title="Tenure", yaxis_title="Total Charges")

    st.plotly_chart(fig)

    st.divider()

    st.subheader("Tenure vs Monthly Charges")

    fig = go.Figure()
    
    for churn_status in df["Churn"].unique():
        df_churn = df[df["Churn"] == churn_status]
        fig.add_trace(go.Scatter(x=df_churn["tenure"], y=df_churn["MonthlyCharges"], mode="markers", name=churn_status))
    
    fig.update_layout(title="Tenure vs Monthly Charges", xaxis_title="Tenure", yaxis_title="Monthly Charges")
    
    st.plotly_chart(fig)

    st.subheader("Top Model Features")

    features = [
    "Contract — Month-to-month",
    "Tenure",
    "Online Security — No",
    "Monthly Charges",
    "Internet Service — Fiber optic",
    "Tech Support — No",
    "Payment Method — Electronic check",
    "Contract — Two year",
    "Paperless Billing — No",
    "Total Charges"
    ]

    for i, feature in enumerate(features, start=1):
        st.write(f"**{i}.** {feature}")



st.header(" Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

with col2:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )



st.subheader("Services")

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


st.subheader("Charges")

col1, col2 = st.columns(2)

with col1:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

with col2:
    st.metric(
        "monthly charges",
        f"${monthly_charges:.2f}"
    )



customer_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

# SHAP Explainer and Preprocessor

explainer = shap.TreeExplainer(
    model.named_steps["model"]
)
preprocessor = model.named_steps["preprocessor"]



st.divider()

if page == "Prediction":
    if st.button("Predict Churn", type="secondary"):

        probability = model.predict_proba(customer_data)[0, 1]

        prediction = int(probability >= threshold)

    # SHAP
        customer_transformed = preprocessor.transform(customer_data)

        feature_names = preprocessor.get_feature_names_out()

        customer_shap = explainer(customer_transformed)

   
    # Prediction Result

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
            "Churn Probability",
            f"{probability:.2%}"
            )

        with col2:
            if prediction == 1:
                st.error("Churn")
            else:
                st.success("customer stay")

        with col3:
            if probability >= 0.70:
                risk = "very High"
            elif probability >= 0.50:
                risk = "High"
            elif probability >= 0.30:
                risk = "medium"
            else:
                risk = "Low"

            st.metric(
                "Risk Level",
                risk
            )


    # Recommendation

        st.subheader("Recommendation")

        if prediction == 1:

            st.warning(
                "This customer has been identified as being at risk "
                "of churn. Consider targeted retention actions such "
                "as personalized offers, service support, contract, monthly charges should be quite low."
            )

        else:

            st.success(
                "This customer currently has a relatively low predicted "
                "churn risk. Continue normal customer engagement."
            )

   
    # SHAP Explanation
    

        st.divider()

        st.subheader("Why did the model make this prediction?")


        st.subheader("Individual Explanation")

        shap.plots.waterfall(
            customer_shap[0],
            max_display=10,
            show=False
            )

        st.pyplot(plt.gcf())
        plt.close()