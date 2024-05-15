import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# Load the dataset
@st.cache_data
def load_data():
    iris = load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])
    return data

# Train the model
def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Main function
def main():
    st.title('Machine Learning App')
    st.write('Diese App nutzt den Random Forest Classifier um die Iris-Art vorherzusagen.')

    # Load the data
    data = load_data()

    # Train the model
    model = train_model(data)

    # User input
    sepal_length = st.slider('Sepal Length', 0.0, 10.0, 5.0)
    sepal_width = st.slider('Sepal Width', 0.0, 10.0, 5.0)
    petal_length = st.slider('Petal Length', 0.0, 10.0, 5.0)
    petal_width = st.slider('Petal Width', 0.0, 10.0, 5.0)

    # Make a prediction
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)

    # Map the prediction to the corresponding Iris species
    iris_species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    predicted_species = iris_species[prediction[0]]

    st.write('Die vorhergesagte Iris-Art ist: ', predicted_species)
    st.image(f'{predicted_species}.jpg')

# Run the main function
if __name__ == "__main__":
    main()