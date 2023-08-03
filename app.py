import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained XGBoost model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
def main():
    st.title('Bike Rental Prediction')

    st.write('Enter the input features:')

    # Get input values from the user
    hour = st.number_input('Hour', min_value=0, max_value=23, value=0)
    temperature = st.number_input('Temperature (°C)', value=0.0)
    wind_speed = st.number_input('Wind speed (m/s)', value=0.0)
    visibility = st.number_input('Visibility (10m)', value=0.0)
    dew_point_temperature = st.number_input('Dew point temperature (°C)', value=0.0)
    solar_radiation = st.number_input('Solar Radiation (MJ/m2)', value=0.0)
    rainfall = st.number_input('Rainfall (mm)', value=0.0)
    seasons = st.selectbox('Seasons', ['Winter', 'Summer', 'Spring', 'Autumn'])
    weekday = st.selectbox('WeekDay', ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
    day = st.number_input('Day', min_value=1, max_value=31, value=1)
    year = st.number_input('Year', min_value=2000, max_value=2100, value=2000)
    no_holiday = st.checkbox('No Holiday')

    # Convert the input data to a DataFrame
    input_data = {
        'Hour': [hour],
        'Temperature(°C)': [temperature],
        'Wind speed (m/s)': [wind_speed],
        'Visibility (10m)': [visibility],
        'Dew point temperature(°C)': [dew_point_temperature],
        'Solar Radiation (MJ/m2)': [solar_radiation],
        'Rainfall(mm)': [rainfall],
        'Seasons': [seasons],
        'WeekDay': [weekday],
        'Day': [day],
        'Year': [year],
        'No Holiday': [no_holiday]
    }

    input_df = pd.DataFrame(input_data)

    # Convert string features to integers
    seasons_mapping = {'Winter': 1, 'Summer': 2, 'Spring': 3, 'Autumn': 4}
    weekday_mapping = {'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7}
    input_df['Seasons'] = input_df['Seasons'].map(seasons_mapping)
    input_df['WeekDay'] = input_df['WeekDay'].map(weekday_mapping)
    input_df['No Holiday'] = input_df['No Holiday'].astype(int)  # Convert checkbox value to integer (0 or 1)

    # Perform scaling on numerical features
    numeric_cols = ['Hour', 'Temperature(°C)', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)',
                    'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Day', 'Year']
    sc = StandardScaler()
    input_df[numeric_cols] = sc.fit_transform(input_df[numeric_cols])

    if st.button('Predict'):
        # Use the model to make predictions
        predictions = model.predict(input_df)
        st.write(f'Predicted bike rentals: {predictions[0]}')

if __name__ == '__main__':
    main()
