import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 3rem;
    font-size: 18px;
    border-radius: 10px;
}

div[data-testid="metric-container"] {
    border: 1px solid #dcdcdc;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = load_model("model.h5")

with open("scalar.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encode_geo.pkl", "rb") as file:
    onehot_encode_geo = pickle.load(file)

# ---------------- HEADER ----------------
st.title("🏦 Customer Churn Prediction Dashboard")
st.write("Predict whether a customer is likely to leave the bank.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📋 Customer Information")

geography = st.sidebar.selectbox(
    "🌍 Geography",
    onehot_encode_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "👤 Gender",
    label_encoder_gender.classes_
)

age = st.sidebar.slider(
    "🎂 Age",
    18,
    92,
    35
)

credit_score = st.sidebar.number_input(
    "💳 Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

balance = st.sidebar.number_input(
    "💰 Balance",
    min_value=0.0,
    value=50000.0
)

estimated_salary = st.sidebar.number_input(
    "💵 Estimated Salary",
    min_value=0.0,
    value=50000.0
)

tenure = st.sidebar.slider(
    "📅 Tenure",
    0,
    10,
    5
)

num_of_products = st.sidebar.slider(
    "📦 Number of Products",
    1,
    4,
    2
)

has_cr_card = st.sidebar.selectbox(
    "💳 Has Credit Card",
    [0, 1]
)

is_active_member = st.sidebar.selectbox(
    "✅ Active Member",
    [0, 1]
)

# ---------------- CUSTOMER SUMMARY ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Credit Score", credit_score)

with col2:
    st.metric("Balance", f"${balance:,.0f}")

with col3:
    st.metric("Salary", f"${estimated_salary:,.0f}")

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Churn"):

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encode_geo.transform(
        [[geography]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encode_geo.get_feature_names_out(
            ['Geography']
        )
    )

    input_data = pd.concat(
        [input_data.reset_index(drop=True),
         geo_encoded_df],
        axis=1
    )

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(
        input_data_scaled,
        verbose=0
    )

    prediction_proba = float(prediction[0][0])

    st.subheader("📊 Prediction Result")

    st.metric(
        "Churn Probability",
        f"{prediction_proba*100:.2f}%"
    )

    st.progress(prediction_proba)

    if prediction_proba < 0.30:
        st.success(
            f"🟢 Low Risk Customer ({prediction_proba*100:.1f}%)"
        )

    elif prediction_proba < 0.70:
        st.warning(
            f"🟡 Medium Risk Customer ({prediction_proba*100:.1f}%)"
        )

    else:
        st.error(
            f"🔴 High Risk Customer ({prediction_proba*100:.1f}%)"
        )

    st.markdown("---")

    st.subheader("📝 Customer Details Used")

    st.dataframe(
        input_data,
        use_container_width=True
    )