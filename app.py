# import streamlit as st
# import requests
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# import joblib

# # Function to get coordinates from city name using Nominatim
# def get_coordinates(city_name):
#     url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json&limit=1"
#     headers = {
#         "User-Agent": "WindSpeedPredictionApp/1.0 (contact@example.com)"  # Replace with your app name and contact info
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
#     url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation,cloud_cover&forecast_days=2"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         st.error("Failed to retrieve data")
#         return None

# # Function to train and evaluate a model
# def train_model(X_train, y_train, X_test, y_test, model_type="Random Forest"):
#     if model_type == "Random Forest":
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#     elif model_type == "Gradient Boosting":
#         model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#     elif model_type == "AdaBoost":
#         model = AdaBoostRegressor(n_estimators=100, random_state=42)
#     elif model_type == "Support Vector Machine":
#         model = SVR(kernel='rbf')
#     elif model_type == "Neural Network":
#         model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
#     else:
#         raise ValueError("Invalid model type")

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     return model, mse, r2

# # Streamlit UI
# st.set_page_config(page_title="Wind Speed Prediction", layout="wide")
# st.title("Wind Speed Prediction Using Advanced AI Models")
# st.markdown("""
#     <div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px;">
#         <p style="color: #004080; font-size: 16px;">
#             This application predicts wind speed using real-time weather data and advanced AI models. 
#             Users can select multiple AI models to compare predictions and visualize the results. 
#             The app leverages data from Open-Meteo and provides accurate forecasts for the next 12 to 48 hours.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# # User Inputs for City Name and Forecast Duration
# city_name = st.text_input("Enter City Name", value="San Francisco")
# forecast_duration = st.slider("Select Forecast Duration (Hours)", min_value=12, max_value=48, value=24, step=12)
# model_types = st.multiselect("Select AI Models", options=["Random Forest", "Gradient Boosting", "AdaBoost", "Support Vector Machine", "Neural Network"])

# if st.button("Get Weather Data and Predict Wind Speed"):
#     lat, lon = get_coordinates(city_name)
#     if lat and lon:
#         data = get_weather_data(lat, lon, forecast_duration)
#         if data:
#             # Prepare time and parameter data
#             times = [datetime.now() + timedelta(hours=i) for i in range(forecast_duration)]
#             df = pd.DataFrame({"Time": times})

#             # Display current weather data in a summary
#             st.subheader("Current Weather Summary")
#             col1, col2, col3, col4, col5, col6 = st.columns(6)
#             col1.metric("Temperature", f"{data['hourly']['temperature_2m'][0]}°C")
#             col2.metric("Humidity", f"{data['hourly']['relative_humidity_2m'][0]}%")
#             col3.metric("Wind Speed", f"{data['hourly']['wind_speed_10m'][0]} m/s")
#             col4.metric("Pressure", f"{data['hourly']['pressure_msl'][0]} hPa")
#             col5.metric("Precipitation", f"{data['hourly']['precipitation'][0]} mm")
#             col6.metric("Cloud Cover", f"{data['hourly']['cloud_cover'][0]}%")

#             # Prepare dataset for AI model
#             weather_df = pd.DataFrame({
#                 "temperature": data['hourly']['temperature_2m'],
#                 "humidity": data['hourly']['relative_humidity_2m'],
#                 "pressure": data['hourly']['pressure_msl'],
#                 "precipitation": data['hourly']['precipitation'],
#                 "cloud_cover": data['hourly']['cloud_cover'],
#                 "wind_speed": data['hourly']['wind_speed_10m']
#             })

#             # Features and target
#             X = weather_df.drop(columns=["wind_speed"])
#             y = weather_df["wind_speed"]

#             # Train-test split
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             # Train and evaluate models
#             results = []
#             for model_type in model_types:
#                 model, mse, r2 = train_model(X_train, y_train, X_test, y_test, model_type)
#                 results.append((model_type, model, mse, r2))

#             # Predict wind speed for the next `forecast_duration` hours
#             future_data = pd.DataFrame({
#                 "temperature": data['hourly']['temperature_2m'][:forecast_duration],
#                 "humidity": data['hourly']['relative_humidity_2m'][:forecast_duration],
#                 "pressure": data['hourly']['pressure_msl'][:forecast_duration],
#                 "precipitation": data['hourly']['precipitation'][:forecast_duration],
#                 "cloud_cover": data['hourly']['cloud_cover'][:forecast_duration]
#             })

#             # Visualize predicted wind speed with advanced dynamic chart
#             st.subheader("Predicted Wind Speed Over Time")
#             fig = go.Figure()
#             for model_type, model, mse, r2 in results:
#                 predicted_wind_speed = model.predict(future_data)
#                 df[f"Predicted Wind Speed (m/s) - {model_type}"] = predicted_wind_speed
#                 fig.add_trace(go.Scatter(x=df["Time"], y=df[f"Predicted Wind Speed (m/s) - {model_type}"], mode="lines+markers", name=f"{model_type}"))

#             fig.update_layout(
#                 title="Predicted Wind Speed (m/s)",
#                 xaxis_title="Time",
#                 yaxis_title="Wind Speed (m/s)",
#                 template="plotly_white",
#                 hovermode="x unified"
#             )
#             st.plotly_chart(fig, use_container_width=True)

#             # Additional Visualizations
#             st.subheader("Additional Visualizations")
#             fig2 = px.line(df, x="Time", y=[f"Predicted Wind Speed (m/s) - {model_type}" for model_type in model_types], title="Comparison of Predicted Wind Speeds")
#             st.plotly_chart(fig2, use_container_width=True)

#             fig3 = px.bar(df, x="Time", y=[f"Predicted Wind Speed (m/s) - {model_type}" for model_type in model_types], title="Bar Chart of Predicted Wind Speeds")
#             st.plotly_chart(fig3, use_container_width=True)

#             # Display dataset
#             st.subheader("Dataset Used for Prediction")
#             st.write(df)
#             st.download_button(
#                 label="Download Dataset as CSV",
#                 data=df.to_csv(index=False).encode('utf-8'),
#                 file_name="wind_speed_prediction_data.csv",
#                 mime="text/csv"
#             )

#             st.write("Forecast Source: Open-Meteo | AI Models: " + ", ".join(model_types))


import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

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
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation,cloud_cover&forecast_days=2"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to retrieve data")
        return None

# Function to train and evaluate a model
def train_model(X_train, y_train, X_test, y_test, model_type="Random Forest"):
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == "AdaBoost":
        model = AdaBoostRegressor(n_estimators=100, random_state=42)
    elif model_type == "Support Vector Machine":
        model = SVR(kernel='rbf')
    elif model_type == "Neural Network":
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2

# Streamlit UI
st.set_page_config(page_title="Wind Speed Prediction", layout="wide")
st.title("Wind Speed Prediction Using AI Models")
st.markdown("""
    <div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px;">
        <p style="color: #004080; font-size: 16px;">
            This application predicts wind speed using real-time weather data and advanced AI models. 
            Users can select multiple AI models to compare predictions and visualize the results.
        </p>
    </div>
    """, unsafe_allow_html=True)

# User Inputs
city_name = st.text_input("Enter City Name")  # Removed default value
model_types = st.multiselect("Select AI Models", options=["Random Forest", "Gradient Boosting", "AdaBoost", "Support Vector Machine", "Neural Network"])
forecast_duration = st.slider("Select Forecast Duration (Hours)", min_value=12, max_value=48, value=24, step=12)

if st.button("Get Wind Speed and Weather Data"):  # Changed button text
    lat, lon = get_coordinates(city_name)
    if lat and lon:
        data = get_weather_data(lat, lon, forecast_duration)
        if data:
            # Prepare time and parameter data
            times = [datetime.now() + timedelta(hours=i) for i in range(forecast_duration)]
            df = pd.DataFrame({"Time": times})

            # Display current weather data in a summary
            st.subheader("Current Weather Summary")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Temperature", f"{data['hourly']['temperature_2m'][0]}°C")
            col2.metric("Humidity", f"{data['hourly']['relative_humidity_2m'][0]}%")
            col3.metric("Wind Speed", f"{data['hourly']['wind_speed_10m'][0]} m/s")
            col4.metric("Pressure", f"{data['hourly']['pressure_msl'][0]} hPa")
            col5.metric("Precipitation", f"{data['hourly']['precipitation'][0]} mm")
            col6.metric("Cloud Cover", f"{data['hourly']['cloud_cover'][0]}%")

            # Prepare dataset for AI model
            weather_df = pd.DataFrame({
                "temperature": data['hourly']['temperature_2m'],
                "humidity": data['hourly']['relative_humidity_2m'],
                "pressure": data['hourly']['pressure_msl'],
                "precipitation": data['hourly']['precipitation'],
                "cloud_cover": data['hourly']['cloud_cover'],
                "wind_speed": data['hourly']['wind_speed_10m']
            })

            # Features and target
            X = weather_df.drop(columns=["wind_speed"])
            y = weather_df["wind_speed"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train and evaluate models
            results = []
            for model_type in model_types:
                model, mse, r2 = train_model(X_train, y_train, X_test, y_test, model_type)
                results.append((model_type, model, mse, r2))

            # Predict wind speed for the next `forecast_duration` hours
            future_data = pd.DataFrame({
                "temperature": data['hourly']['temperature_2m'][:forecast_duration],
                "humidity": data['hourly']['relative_humidity_2m'][:forecast_duration],
                "pressure": data['hourly']['pressure_msl'][:forecast_duration],
                "precipitation": data['hourly']['precipitation'][:forecast_duration],
                "cloud_cover": data['hourly']['cloud_cover'][:forecast_duration]
            })

            # Visualize predicted wind speed
            st.subheader("Predicted Wind Speed Over Time")
            fig = go.Figure()
            for model_type, model, mse, r2 in results:
                predicted_wind_speed = model.predict(future_data)
                df[f"Predicted Wind Speed (m/s) - {model_type}"] = predicted_wind_speed
                fig.add_trace(go.Scatter(x=df["Time"], y=df[f"Predicted Wind Speed (m/s) - {model_type}"], mode="lines+markers", name=f"{model_type}"))

            fig.update_layout(
                title="Predicted Wind Speed (m/s)",
                xaxis_title="Time",
                yaxis_title="Wind Speed (m/s)",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Additional Visualizations
            st.subheader("Comparison of Predicted Wind Speeds")
            fig2 = px.line(df, x="Time", y=[f"Predicted Wind Speed (m/s) - {model_type}" for model_type in model_types], title="Line Chart of Predicted Wind Speeds")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Bar Chart of Predicted Wind Speeds")
            fig3 = px.bar(df, x="Time", y=[f"Predicted Wind Speed (m/s) - {model_type}" for model_type in model_types], title="Bar Chart of Predicted Wind Speeds")
            st.plotly_chart(fig3, use_container_width=True)

            # Display dataset
            st.subheader("Dataset Used for Prediction")
            st.write(df)
            st.download_button(
                label="Download Dataset as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="wind_speed_prediction_data.csv",
                mime="text/csv"
            )
