# streamlit_app.py
import streamlit as st
import pickle
# import pypickle as pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
# import pandas as pd
# D:\hackthon\spg\workflow-lithofacies classificaation\best_models
# Load models
models = {
    "CatBoost": "catboost_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "LightGBM": "lightgbm_model.pkl",
    # "KNN": "./workflow-lithofacies classificaation/best_models/knn_model.pkl",
    # "SVM": "./workflow-lithofacies classificaation/best_models/svm_model.pkl"
}

loaded_models = {}
for name, path in models.items():
    with open(path, "rb") as f:
        loaded_models[name] = pickle.load(f)

# Streamlit UI
st.title("SPG Hackathon Fluvial-facies Classification")

st.write("Enter 4 features to predict the target:")

# Take input
f1 = st.number_input("Permeability(0 500)", value=150.0 ,format="%.6f")
f2 = st.number_input("Gamma (0-150) 50-130 for best result", value=60.0,format="%.6f")
f3 = st.number_input("porosity(0-.5)", value=.2,format="%.6f")
f4 = st.number_input("netgross(0-1)", value=0.5,format="%.6f")
# df = pd.read_csv('./workflow-lithofacies classificaation/processed_data/las_data/A1.csv')
# fluvial_actual = df['FLUVIALFACIES']
# df = df.drop(columns=['FLUVIALFACIES'])
# df = [df.iloc[229]]
input_data = np.array([[f1, f2, f3, f4]])
# input_data = np.array(df)

# Standard scaling
scaler = StandardScaler()
  # ⚠️ Ideally fit on training data

if st.button("Predict"):
    st.subheader("Predictions")
    for name, model in loaded_models.items():
        try:
            if name == 'SVM' or name == 'KNN':
                scaled_data = scaler.fit_transform(input_data)
                pred = model.predict(scaled_data)[0]
            else:
                pred = model.predict(input_data)[0]
            if name == "CatBoost":
                st.write(f"**{name} Prediction:** {pred[0]}")
            else:
                st.write(f"**{name} Prediction:** {pred}")
            # print(f"{name} Prediction: {pred}")
        except Exception as e:
            st.write(f"Error with {name}: {e}")
            # print(f"Error with {name}: {e}")
