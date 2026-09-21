import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('log.sav')

# Get feature names (ensure this order matches your training data)
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                 'Warehouse_Processing_Time']

st.title('Delivery Delay Prediction App')
st.write('Enter the values for the delivery features to predict if there will be a delay.')

# Create input widgets for each feature
input_features = {}
for feature in feature_names:
    input_features[feature] = st.number_input(f'{feature.replace("_", " ").title()}', value=0.0)

if st.button('Predict'):
    # Collect inputs into a DataFrame
    input_df = pd.DataFrame([input_features])

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('The delivery is predicted to be **Delayed**.')
    else:
        st.success('The delivery is predicted to be **On Time**.')

    st.write(f"Probability of 'On Time' (0): {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of 'Delayed' (1): {prediction_proba[0][1]:.2f}")

st.write('---
To run this Streamlit app, execute the following commands in your Colab notebook (in separate cells or after installation):
```bash
!pip install streamlit
!streamlit run app.py &>/dev/null&
```
After running the `streamlit run` command, a public URL will be provided that you can click to access the app.')
