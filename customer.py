import streamlit as st
import pickle
import numpy as np

# Load the KMeans model
model = pickle.load(open('customer_segmentation_model.pkl', 'rb'))

st.title("Customer Segmentation Prediction")

# Input Fields
income = st.number_input('Enter Annual Income (in $K)', min_value=0)
spending_score = st.number_input('Enter Spending Score ', min_value=1, max_value=100)

cluster_labels = {
    0: 'Medium Income - Medium Spender',
    1: 'High Income - High Spender',
    2: 'High Income - Low/Medium Spender',
    3: 'Low Income - Mixed Spending Behavior',
    4: 'Low Income - High Spender'
}
# Predict Button
if st.button('Predict Cluster'):
    input_data = np.array([[income, spending_score]])
    cluster = model.predict(input_data)[0]
    st.success(f'You are belongs to {cluster_labels[cluster]}')
    st.info(f"Income Entered: ₹{income * 1000}")
    st.info(f"Estimated Spending Score: {round(spending_score, 2)}/100")

