# import streamlit as st
# import requests
# import pandas as pd
# from datetime import datetime, timedelta

# # Function to get coordinates from city name using Nominatim
# def get_coordinates(city_name):
#     url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json&limit=1"
#     headers = {
#         "User-Agent": "WindSpeedPredictor/1.0"  # Updated app name
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         location_data = response.json()
#         if location_data:
#             location = location_data[0]
#             return float(location['lat']), float(location['lon'])
#         else:
#             st.warning("City not found. Please check the spelling or try adding the country name (e.g., 'San Francisco, USA').")
#             return None, None
#     else:
#         st.error(f"API request failed with status code {response.status_code}: {response.text}")
#         return None, None

# # Function to get weather data from Open-Meteo
# def get_weather_data(lat, lon, hours):
#     url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation,cloud_cover,wind_direction_10m&forecast_days=2"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         st.error("Failed to retrieve data")
#         return None

# # Streamlit UI for downloading dataset
# def download_dataset():
#     st.title("Download Weather Dataset")
#     city_name = st.text_input("Enter City Name")
#     forecast_duration = st.slider("Select Forecast Duration (Hours)", min_value=12, max_value=48, value=24, step=12)

#     if st.button("Get Weather Data"):
#         lat, lon = get_coordinates(city_name)
#         if lat and lon:
#             data = get_weather_data(lat, lon, forecast_duration)
#             if data:
#                 times = [datetime.now() + timedelta(hours=i) for i in range(forecast_duration)]
#                 df = pd.DataFrame({
#                     "Time": times,
#                     "temperature": data['hourly']['temperature_2m'][:forecast_duration],
#                     "humidity": data['hourly']['relative_humidity_2m'][:forecast_duration],
#                     "pressure": data['hourly']['pressure_msl'][:forecast_duration],
#                     "precipitation": data['hourly']['precipitation'][:forecast_duration],
#                     "cloud_cover": data['hourly']['cloud_cover'][:forecast_duration],
#                     "wind_speed": data['hourly']['wind_speed_10m'][:forecast_duration],
#                     "wind_direction": data['hourly']['wind_direction_10m'][:forecast_duration]
#                 })
#                 st.write(df)
#                 st.download_button(
#                     label="Download Dataset as CSV",
#                     data=df.to_csv(index=False).encode('utf-8'),
#                     file_name="weather_data.csv",
#                     mime="text/csv"
#                 )

# if __name__ == "__main__":
#     download_dataset()


import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta

# OpenWeatherMap API Key
API_KEY = "e0100edeedd99f5ae298581c486626a4"

# Function to get coordinates from city name using OpenWeatherMap
def get_coordinates(city_name):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        location_data = response.json()
        if location_data:
            location = location_data[0]
            return location['lat'], location['lon']
        else:
            st.warning(f"City not found: {city_name}. Please check the spelling or try adding the country name.")
            return None, None
    else:
        st.error(f"API request failed with status code {response.status_code}: {response.text}")
        return None, None

# Function to get weather data from OpenWeatherMap
def get_weather_data(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['list']  # Returns 5-day forecast in 3-hour intervals
    else:
        st.error(f"Failed to retrieve data for lat={lat}, lon={lon}: {response.status_code} - {response.text}")
        return None

# Streamlit UI for downloading dataset
def download_dataset():
    st.title("Download Weather Dataset")
    city_name = st.text_input("Enter City Name")
    forecast_duration = st.slider("Select Forecast Duration (Hours)", min_value=12, max_value=48, value=24, step=12)

    if st.button("Get Weather Data"):
        lat, lon = get_coordinates(city_name)
        if lat and lon:
            weather_data = get_weather_data(lat, lon)
            if weather_data:
                times = [datetime.fromtimestamp(hour['dt']) for hour in weather_data[:forecast_duration]]
                df = pd.DataFrame({
                    "Time": times,
                    "temperature": [hour['main']['temp'] for hour in weather_data[:forecast_duration]],
                    "humidity": [hour['main']['humidity'] for hour in weather_data[:forecast_duration]],
                    "pressure": [hour['main']['pressure'] for hour in weather_data[:forecast_duration]],
                    "precipitation": [hour.get('rain', {}).get('3h', 0) for hour in weather_data[:forecast_duration]],
                    "cloud_cover": [hour['clouds']['all'] for hour in weather_data[:forecast_duration]],
                    "wind_speed": [hour['wind']['speed'] for hour in weather_data[:forecast_duration]],
                    "wind_direction": [hour['wind']['deg'] for hour in weather_data[:forecast_duration]],
                    "visibility": [hour.get('visibility', 0) for hour in weather_data[:forecast_duration]],  # Visibility in meters
                    "dew_point": [hour.get('dew_point', 0) for hour in weather_data[:forecast_duration]],  # Dew point in Celsius
                    "uvi": [hour.get('uvi', 0) for hour in weather_data[:forecast_duration]],  # UV index
                    "wind_gust": [hour['wind'].get('gust', 0) for hour in weather_data[:forecast_duration]]  # Wind gust speed
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
