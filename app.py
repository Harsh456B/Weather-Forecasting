import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime, timedelta
import pytz

# Constants
API_KEY = 'c5385722edf9ea23965d802ab9842e35'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# Helper Functions
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'wind_speed': data['wind']['speed'],
        'pressure': data['main']['pressure']
    }

def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    x = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    return x, y, le

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Streamlit App
st.title("Weather Forecasting App")

# City Input
city = st.text_input("Enter City Name:", "Delhi")
if st.button("Get Weather Forecast"):
    with st.spinner("Fetching current weather and loading models..."):
        # Current Weather
        current_weather = get_current_weather(city)

        # Load historical data
        historical_data = pd.read_csv('indian_weather_prediction_dataset.csv')

        # Load or Train Models
        try:
            rain_model = load_model('rain_model.pkl')
            temp_model = load_model('temp_model.pkl')
            hum_model = load_model('hum_model.pkl')
            le = load_model('label_encoder.pkl')
        except FileNotFoundError:
            st.warning("Training models for the first time. This may take a while...")
            # Prepare Data and Train Models
            x, y, le = prepare_data(historical_data)
            rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rain_model.fit(x, y)
            save_model(rain_model, 'rain_model.pkl')
            save_model(le, 'label_encoder.pkl')

            # Train Regression Models
            x_temp, y_temp = x['Temp'], y
            x_hum, y_hum = x['Humidity'], y
            temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
            temp_model.fit(np.array(x_temp).reshape(-1, 1), y_temp)
            hum_model = RandomForestRegressor(n_estimators=100, random_state=42)
            hum_model.fit(np.array(x_hum).reshape(-1, 1), y_hum)

            # Save Regression Models
            save_model(temp_model, 'temp_model.pkl')
            save_model(hum_model, 'hum_model.pkl')

        # Make Predictions
        future_temp = [current_weather['temp_min']]
        future_humidity = [current_weather['humidity']]

        for i in range(5):
            next_temp = temp_model.predict(np.array([[future_temp[-1]]]))[0]
            future_temp.append(next_temp)

            next_hum = hum_model.predict(np.array([[future_humidity[-1]]]))[0]
            future_humidity.append(next_hum)

        future_temp = future_temp[1:]  # Remove the initial current value
        future_humidity = future_humidity[1:]

        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

    # Display Results
    st.subheader(f"Weather in {city}, {current_weather['country']}")
    st.write(f"**Temperature:** {current_weather['current_temp']}\u00b0C")
    st.write(f"**Feels Like:** {current_weather['feels_like']}\u00b0C")
    st.write(f"**Description:** {current_weather['description']}")
    st.write(f"**Humidity:** {current_weather['humidity']}%")
    st.write(f"**Pressure:** {current_weather['pressure']} hPa")
    rain_pred = rain_model.predict(pd.DataFrame([current_weather]))[0]
    st.write(f"**Rain Prediction:** {'Yes' if rain_pred else 'No'}")

    st.subheader("Future Predictions")
    st.write("### Temperature (Next 5 Hours):")
    for time, temp in zip(future_times, future_temp):
        st.write(f"{time}: {round(temp, 1)}\u00b0C")
    st.write("### Humidity (Next 5 Hours):")
    for time, humidity in zip(future_times, future_humidity):
        st.write(f"{time}: {round(humidity, 1)}%")
