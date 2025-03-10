import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

# Function to get coordinates from city name using Nominatim
def get_coordinates(city_name):
    url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json&limit=1"
    headers = {
        "User-Agent": "WindSpeedPredictor/1.0"  # Updated app name
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        location_data = response.json()
        if location_data:
            location = location_data[0]
            return float(location['lat']), float(location['lon'])
        else:
            st.warning("City not found. Please check the spelling or try adding the country name (e.g., 'San Francisco, USA').")
            return None, None
    else:
        st.error(f"API request failed with status code {response.status_code}: {response.text}")
        return None, None

# Function to get weather data from Open-Meteo
def get_weather_data(lat, lon, hours):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation,cloud_cover,wind_direction_10m&forecast_days=2"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to retrieve data")
        return None

# Streamlit UI for downloading dataset
def download_dataset():
    st.title("Download Weather Dataset")
    city_name = st.text_input("Enter City Name")
    forecast_duration = st.slider("Select Forecast Duration (Hours)", min_value=12, max_value=48, value=24, step=12)

    if st.button("Get Weather Data"):
        lat, lon = get_coordinates(city_name)
        if lat and lon:
            data = get_weather_data(lat, lon, forecast_duration)
            if data:
                times = [datetime.now() + timedelta(hours=i) for i in range(forecast_duration)]
                df = pd.DataFrame({
                    "Time": times,
                    "temperature": data['hourly']['temperature_2m'][:forecast_duration],
                    "humidity": data['hourly']['relative_humidity_2m'][:forecast_duration],
                    "pressure": data['hourly']['pressure_msl'][:forecast_duration],
                    "precipitation": data['hourly']['precipitation'][:forecast_duration],
                    "cloud_cover": data['hourly']['cloud_cover'][:forecast_duration],
                    "wind_speed": data['hourly']['wind_speed_10m'][:forecast_duration],
                    "wind_direction": data['hourly']['wind_direction_10m'][:forecast_duration]
                })
                st.write(df)
                st.download_button(
                    label="Download Dataset as CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="weather_data.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    download_dataset()
