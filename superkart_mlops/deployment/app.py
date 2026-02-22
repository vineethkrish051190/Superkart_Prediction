import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="vnsonly05/Superkart-Prediction", filename="superkart_best_model.joblib")
model = joblib.load(model_path)

# Streamlit UI for Superkart Sales Prediction
st.title("SuperKart Sales Prediction App")
st.write("Enter product and store details to forecast total sales.")

# 2. Get inputs and save into a DataFrame
col1, col2 = st.columns(2)

with col1:
    weight = st.number_input("Product Weight", min_value=0.0, value=12.0)
    sugar = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    area = st.number_input("Allocated Area", min_value=0.0, max_value=1.0, value=0.05)
    p_type = st.selectbox("Product Type", ['Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Others'])
    mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)

with col2:
    est_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2024, value=2000)
    s_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    city_type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
    s_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

# Convert inputs to DataFrame (must match training column names exactly)
input_df = pd.DataFrame([{
    'Product_Weight': weight,
    'Product_Sugar_Content': sugar,
    'Product_Allocated_Area': area,
    'Product_Type': p_type,
    'Product_MRP': mrp,
    'Store_Establishment_Year': est_year,
    'Store_Size': s_size,
    'Store_Location_City_Type': city_type,
    'Store_Type': s_type
}])

if st.button("Predict Sales"):
    # The pipeline handles the One-Hot Encoding internally
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Total Sales: ${prediction:,.2f}")
