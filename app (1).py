import streamlit as st
import joblib
import numpy as np

# Load the model
# Ensure 'iris_model.pkl' is available in the Colab environment or path is correct
model = joblib.load('iris_model.pkl')

# Page title
st.title('Iris Species Prediction App')

# Input labels
st.header('Enter Iris Flower Measurements')
sepals_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepals_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petals_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0, step=0.1)
petals_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Prediction button
if st.button('Predict Species'):
    # Create a numpy array from the inputs
    input_data = np.array([[sepals_length, sepals_width, petals_length, petals_width]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction
    st.success(f'The predicted Iris species is: {prediction[0]}')

# Instructions for running in Colab:
# 1. Save the content of this cell to a file named `app.py` in your Colab files:
#    %%writefile app.py
#    (then copy and paste the code above, or run this cell, then the next if you want to write this to a file)
# 2. In a new cell, run: `!streamlit run app.py & npx localtunnel --port 8501`
# 3. Click the public URL to open your Streamlit app.
