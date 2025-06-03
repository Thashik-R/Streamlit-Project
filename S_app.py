# combined_app.py

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import json


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config
config = load_config(r'Final_Check\data.json')
    
# Constants
IMG_HEIGHT = config["IMG_HEIGHT"]
IMG_WIDTH = config["IMG_WIDTH"]
CLASS_NAMES = ['Bald', 'NotBald']
MODEL_PATH = config["final_model_path"]#r'Working/mobilenet_bald_classifier_augmented1.h5'
Excel_file=config["Excel_path"]

@st.cache_data
def load_data():
    df = pd.read_csv(Excel_file)
    return df

def preprocess_data(df):
    df = df.drop(columns=['Timestamp', 'What is your name ?'], errors='ignore')
    le = LabelEncoder()
    df['Do you have hair fall problem ?'] = le.fit_transform(df['Do you have hair fall problem ?'])
    df['What is your gender ?'] = le.fit_transform(df['What is your gender ?'])

    encode_cols = [
        'Is there anyone in your family having a hair fall problem or a baldness issue?',
        'Did you face any type of chronic illness in the past?',
        'Do you stay up late at night?',
        'Do you have any type of sleep disturbance?',
        'Do you think that in your area water is a reason behind hair fall problems?',
        'Do you use chemicals, hair gel, or color in your hair?',
        'Do you have anemia?',
        'Do you have too much stress',
        'What is your food habit'
    ]

    le = LabelEncoder()
    for col in encode_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    return df

def prepare_user_input(user_data, training_columns):
    user_df = pd.DataFrame(user_data, index=[0])
    return user_df.reindex(columns=training_columns, fill_value=0)

# === Image Classifier ===
@st.cache_resource
def load_image_model():
    return load_model(MODEL_PATH)

def predict_image_from_upload(uploaded_file, model):
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image.resize((IMG_WIDTH, IMG_HEIGHT))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    label = CLASS_NAMES[int(pred > 0.5)]
    confidence = pred if label == "NotBald" else 1 - pred
    return label, confidence

def combine_predictions(lr_pred, mn_pred):
    if lr_pred == 1 and mn_pred == 'Bald':
        return "🔴 High Hair Problem"
    elif lr_pred == 0 and mn_pred == 'Bald':
        return "🟠 Medium Hair Problem"
    elif lr_pred == 1 and mn_pred == 'NotBald':
        return "🟡 Low Hair Problem"
    else:
        return "🟢 No Hair Problem"

# === Streamlit App ===
def main():
    st.set_page_config("Hair Fall + Baldness Predictor", layout="wide")

    st.title("🧠 Hair Fall & Baldness Analyzer")
    st.markdown("This app uses **form data** and **image processing** to assess your hair condition.")

    df = load_data()
    df = preprocess_data(df)
    X = df.drop('Do you have hair fall problem ?', axis=1)
    y = df['Do you have hair fall problem ?']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_lr = LogisticRegression(max_iter=500)
    model_lr.fit(X_train, y_train)
    acc = accuracy_score(y_test, model_lr.predict(X_test))
    st.sidebar.success(f"Logistic Regression Accuracy: {acc:.2f}")

    image_model = load_image_model()

    with st.form("input_form"):
        st.subheader("📋 Lifestyle Questionnaire")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            age = st.slider('Age', 15, 100, 24)
            family_history = st.selectbox('Family history of hair fall?', ['Yes', 'No'])
            chronic_illness = st.selectbox('Chronic illness in past?', ['Yes', 'No'])
            stay_up_late = st.selectbox('Stay up late at night?', ['Yes', 'No'])

        with col2:
            sleep_disturbance = st.selectbox('Sleep disturbance?', ['Yes', 'No'])
            water_quality = st.selectbox('Is water a reason?', ['Yes', 'No'])
            use_chemicals = st.selectbox('Use hair chemicals?', ['Yes', 'No'])
            anemia = st.selectbox('Do you have anemia?', ['Yes', 'No'])
            stress = st.selectbox('Too much stress?', ['Yes', 'No'])
            food_habit = st.selectbox('Food habit', ['Nutritious', 'Junk', 'Both'])

        uploaded_image = st.file_uploader("📷 Upload a scalp image", type=['jpg', 'jpeg', 'png'])

        submit = st.form_submit_button("Analyze")

    if submit:
        user_input = {
            'What is your gender ?': 1 if gender == 'Male' else 0,
            'What is your age ?': age,
            'Is there anyone in your family having a hair fall problem or a baldness issue?': 1 if family_history == 'Yes' else 0,
            'Did you face any type of chronic illness in the past?': 1 if chronic_illness == 'Yes' else 0,
            'Do you stay up late at night?': 1 if stay_up_late == 'Yes' else 0,
            'Do you have any type of sleep disturbance?': 1 if sleep_disturbance == 'Yes' else 0,
            'Do you think that in your area water is a reason behind hair fall problems?': 1 if water_quality == 'Yes' else 0,
            'Do you use chemicals, hair gel, or color in your hair?': 1 if use_chemicals == 'Yes' else 0,
            'Do you have anemia?': 1 if anemia == 'Yes' else 0,
            'Do you have too much stress': 1 if stress == 'Yes' else 0,
            'What is your food habit': 0 if food_habit == 'Nutritious' else 1 if food_habit == 'Junk' else 2
        }

        input_df = prepare_user_input(user_input, X.columns)
        lr_prediction = model_lr.predict(input_df)[0]  # 1 or 0

        if uploaded_image is not None:
            mn_label, mn_conf = predict_image_from_upload(uploaded_image, image_model)

            st.subheader("📊 Combined Prediction Result")
            final_result = combine_predictions(lr_prediction, mn_label)
            st.success(f"Result: {final_result}")
            st.write(f"LR Prediction: {'Have Hair Loss' if lr_prediction == 1 else 'No Hair Loss'}")
            st.write(f"Image Prediction: {mn_label} (Confidence: {mn_conf:.2f})")

            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        else:
            st.error("Please upload a scalp image for combined prediction.")

if __name__ == "__main__":
    main()
