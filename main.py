import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from PIL import Image
import streamlit as st
import requests

favicon = Image.open(requests.get('https://raw.githubusercontent.com/swarnya246/farmclean/main/favicon.png', stream=True).raw)
st.set_page_config(page_title='Farm Clean', page_icon=favicon)
image = Image.open(requests.get('https://raw.githubusercontent.com/swarnya246/farmclean/main/field.jpg', stream=True).raw)
st.image(image, use_column_width=True)
st.title("Farm Clean")

df = pd.read_csv('https://raw.githubusercontent.com/swarnya246/farmclean/main/Crop_recommendation.csv')

def get_user_input():
    nitrogen = st.sidebar.number_input('Nitrogen Content of Soil', 0.0, 140.0, 0.0, 0.5)
    phosphorus = st.sidebar.number_input('Phosphorus Content of Soil', 5.0, 145.0, 5.0, 0.5)
    potassium = st.sidebar.number_input('Potassium Content of Soil', 5.0, 205.0, 5.0, 0.5)
    temperature = st.sidebar.number_input('Temperature of Environment (C)', 0.0, 60.0, 0.0, 0.1)
    humidity = st.sidebar.number_input('Humidity of Environment (%)', 10.0, 100.0, 10.0, 0.5)
    ph = st.sidebar.number_input('pH of Soil', 0.0, 7.0, 0.0, 0.1)
    rainfall = st.sidebar.number_input('Annual Rainfall (mm)', 0.0, 300.0, 0.0, 0.5)

    user_data = {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

X = df.iloc[:, 0:7].values
Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sgd_classifier = SGDClassifier(tol=-np.infty, loss='log')
sgd_classifier.fit(X_train, Y_train)

labels = df.label.unique().tolist()
labels.sort()

st.markdown("#### Based on your soil and evironment, we recommend planting the following:")

results = pd.DataFrame(sgd_classifier.predict_proba(user_input).transpose())
results.rename(columns={0: "score"}, inplace=True)
results.reset_index(level=0, inplace=True)
results.sort_values(by="score", ascending=False, inplace=True, ignore_index=True)

for i in range(22):
    if results.loc[i, "score"] > 0.001:
        st.markdown("- " + labels[results.loc[i, "index"]] + " (" + str(round(100 * results.loc[i, "score"])) + "%)")

st.write("Using precision agriculture to diversify the crop rotation is a method " +
         "used to prevent soil degradation. Precision agriculture is the farming " +
         "management concept that uses technological tools such as apps to measure, " +
         "predict, and control the crops. Diversifying the crop rotation disrupts " +
         "pests’ lifecycles, adds fertility to the soil, and maintains the organic " +
         "content of the soil year round.")
st.write("The crop recommender using machine learning to determine the crop or crops " +
         "that would be the best for the soil and environment, with the confidence " +
         "displayed next to each crop.")