import streamlit as st
import pickle
import pandas as pd

# Load Model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🌸 Iris Flower Classification")

# st.write("Enter the flower measurements below.")

# User Inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0)

# Predict Button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "SepalLengthCm": [sepal_length],
        "SepalWidthCm": [sepal_width],
        "PetalLengthCm": [petal_length],
        "PetalWidthCm": [petal_width]
    })

    prediction = model.predict(input_data)

    st.success(f"🌸 Predicted Flower: {prediction[0]}")