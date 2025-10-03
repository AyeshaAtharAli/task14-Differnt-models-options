import streamlit as st
import joblib
import numpy as np

# Load the saved models - assuming they are in the same directory as app.py
loaded_lr_model = joblib.load('logistic_regression_model.pkl')
loaded_dt_model = joblib.load('decision_tree_model.pkl')
loaded_knn_model = joblib.load('knn_model.pkl')

# Define a prediction function
def predict(model, input_features):
    """
    Makes a prediction using the selected model and input features.

    Args:
        model: The trained machine learning model.
        input_features: A numpy array of input features.

    Returns:
        The predicted class.
    """
    prediction = model.predict(input_features)
    return prediction[0]

st.title("Wine Quality Prediction")

# Create a sidebar for model selection
st.sidebar.header("Select Model")
selected_model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    ['Logistic Regression', 'Decision Tree', 'KNN']
)

# Map the selected model name to the loaded model object
if selected_model_name == 'Logistic Regression':
    selected_model = loaded_lr_model
elif selected_model_name == 'Decision Tree':
    selected_model = loaded_dt_model
else:
    selected_model = loaded_knn_model

st.sidebar.write(f"You selected: {selected_model_name}")

st.header("Enter Wine Features")

# Add input fields for the 13 features
# Referencing the Wine dataset documentation for feature descriptions and typical ranges
alcohol = st.number_input("Alcohol", min_value=11.0, max_value=15.0, value=13.0, step=0.01)
malic_acid = st.number_input("Malic Acid", min_value=0.5, max_value=6.0, value=2.0, step=0.01)
ash = st.number_input("Ash", min_value=1.0, max_value=4.0, value=2.5, step=0.01)
alcalinity_of_ash = st.number_input("Alcalinity of Ash", min_value=10.0, max_value=30.0, value=20.0, step=0.1)
magnesium = st.number_input("Magnesium", min_value=70, max_value=160, value=100, step=1)
total_phenols = st.number_input("Total Phenols", min_value=1.0, max_value=4.0, value=2.5, step=0.01)
flavanoids = st.number_input("Flavanoids", min_value=0.5, max_value=5.0, value=2.5, step=0.01)
nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", min_value=0.1, max_value=0.6, value=0.3, step=0.01)
proanthocyanins = st.number_input("Proanthocyanins", min_value=0.3, max_value=3.0, value=1.5, step=0.01)
color_intensity = st.number_input("Color Intensity", min_value=1.0, max_value=14.0, value=5.0, step=0.1)
hue = st.number_input("Hue", min_value=0.5, max_value=1.8, value=1.0, step=0.01)
od280_od315_of_diluted_wines = st.number_input("OD280/OD315 of diluted wines", min_value=1.0, max_value=4.0, value=2.5, step=0.01)
proline = st.number_input("Proline", min_value=100, max_value=2000, value=700, step=10)

# Create a button to make predictions
if st.button("Predict"):
    # Collect input features into a numpy array
    input_features = np.array([[
        alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
        total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
        color_intensity, hue, od280_od315_of_diluted_wines, proline
    ]])

    # Make prediction using the selected model
    prediction = predict(selected_model, input_features)

    # Display the prediction result
    st.subheader("Prediction Result")
    st.write(f"The predicted wine class is: {prediction}")
