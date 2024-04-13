import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle

def prediction(lst, model):
    filename = 'website/predictor.pickle'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    pred_value = model.predict([lst])
    return pred_value[0]

def preprocess_input(data):
    categorical_cols = ['long_hair', 'nose_wide', 'nose_long', 'lips_thin', 'distance_nose_to_lip_long']
    data_encoded = pd.get_dummies(data, columns=categorical_cols)
    return data_encoded

def predict_gender(input_data, model):
    # Preprocess input data to ensure it has the same columns as during training
    input_data_encoded = preprocess_input(input_data)
    
    # Get the feature names used during training
    training_feature_names = model.feature_names_in_
    
    # Reorder columns to match training feature names
    input_data_encoded = input_data_encoded.reindex(columns=training_feature_names, fill_value=0)
    
    # Predict gender
    prediction = model.predict(input_data_encoded)
    return prediction





def main():
    st.title("Gender Prediction App")
    st.header("Enter Facial Features")
    long_hair = st.selectbox("Long Hair", [0, 1])
    forehead_width = st.number_input("Forehead Width (cm)", min_value=0.0)
    forehead_height = st.number_input("Forehead Height (cm)", min_value=0.0)
    nose_wide = st.selectbox("Wide Nose", [0, 1])
    nose_long = st.selectbox("Long Nose", [0, 1])
    lips_thin = st.selectbox("Thin Lips", [0, 1])
    distance_nose_to_lip_long = st.selectbox("Long Distance Nose to Lip", [0, 1])
    
    input_data = {
        'long_hair': long_hair,
        'forehead_width_cm': forehead_width,
        'forehead_height_cm': forehead_height,
        'nose_wide': nose_wide,
        'nose_long': nose_long,
        'lips_thin': lips_thin,
        'distance_nose_to_lip_long': distance_nose_to_lip_long
    }
    
    # Load the model
    with open('website/predictor.pickle', 'rb') as file:
        model = pickle.load(file)
    
    if st.button("Predict Gender"):
        prediction = predict_gender(pd.DataFrame(input_data, index=[0]), model)
        st.success(f"Predicted Gender: {prediction[0]}")

if __name__ == '__main__':
    main()
