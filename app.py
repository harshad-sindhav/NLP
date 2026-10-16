
import pandas as pd
import numpy as np
import pickle as p
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import tensorflow as tf
import streamlit as st

print("Libraries imported")

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = p.load(f)

# Load gender encoder
with open("label_gender.pkl", "rb") as f:
    label_encoder_gender = p.load(f)

# Load geography encoder
with open("one_hot_geo.pkl", "rb") as f:
    one_hot_encoder_geo = p.load(f)

print("Encoders and scaler are loaded")

# Streamlit app
st.title("Customer Churn Prediction App")

# Input fields
geography = st.selectbox(
    "Geography",
    options=one_hot_encoder_geo.categories_[0].tolist()
)

gender = st.selectbox(
    "Gender",
    options=label_encoder_gender.classes_.tolist()
)

age = st.slider("Age", 18, 100)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=60000.0
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

credit_score = st.number_input(
    "Credit Score",
    min_value=0,
    value=650
)

tenure = st.slider("Tenure", 0, 10)

numof_products = st.slider(
    "Number of Products",
    1, 4
)

hascards = st.selectbox("Has Credit Card", [0, 1])

Isactive_member = st.selectbox("Is Active Member", [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [
        label_encoder_gender.transform([gender])[0]
    ],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [numof_products],
    "HasCrCard": [hascards],
    "IsActiveMember": [Isactive_member],
    "EstimatedSalary": [estimated_salary]
})

# One-hot encode Geography
geo_encoded = one_hot_encoder_geo.transform(
    [[geography]]
).toarray()

geo_encode_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder_geo.get_feature_names_out(
        ["Geography"]
    )
)

# Combine input data and geography columns
input_data = pd.concat(
    [
        input_data.reset_index(drop=True),
        geo_encode_df.reset_index(drop=True)
    ],
    axis=1
)

# Match the training feature order when available
if hasattr(scaler, "feature_names_in_"):
    input_data = input_data.reindex(
        columns=scaler.feature_names_in_
    )

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Predict customer churn
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled, verbose=0)
    prediction_proba = float(prediction[0][0])

    st.write(
        f"Churn Probability: {prediction_proba:.2%}"
    )

    if prediction_proba >= 0.5:
        st.error("Customer is likely to leave.")
    else:
        st.success("Customer is likely to stay.")
