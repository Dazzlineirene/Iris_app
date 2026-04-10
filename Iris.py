import streamlit as st
import pandas as pd
import pickle

# 1. Load the pre-trained model
# Ensure the .pkl file is in the same directory as this script
model = pickle.load(open("model.pkl", "rb"))


st.write("""
# Iris Specie Prediction App
This app uses a **pre-trained pickle model** to predict the Iris species.
""")

st.sidebar.header('User Input Features')


def user_input_features():
    # Using the sample features provided in your prompt
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

    data = {
        'SepalLengthCm': sepal_length,
        'SepalWidthCm': sepal_width,
        'PetalLengthCm': petal_length,
        'PetalWidthCm': petal_width
    }
    return pd.DataFrame(data, index=[0])


# Capture user input
df = user_input_features()

st.subheader('Input Summary')
st.write(df)

# 2. Make Prediction
# The model expects the same feature names/order used during training
prediction = model.predict(df)

# Mapping dictionary (Update this based on your specific model's labels)
target_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
result = target_names.get(prediction[0], "Unknown")

# 3. Display Results
st.subheader('Prediction')
st.success(f"The predicted species is: **{result}**")

