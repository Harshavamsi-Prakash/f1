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
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
# import joblib

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
#     return model

# # Streamlit UI
# st.set_page_config(page_title="Wind Speed Prediction", layout="wide")
# st.title("Wind Speed Prediction Using AI Models")
# st.markdown("""
#     <div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px;">
#         <p style="color: #004080; font-size: 16px;">
#             This application predicts wind speed using real-time weather data and advanced AI models, providing accurate forecasts for 12 to 48 hours. Users can compare multiple machine learning models and analyze trends through interactive visualizations. By leveraging key meteorological factors like temperature, humidity, and pressure, the system enhances prediction accuracy. Ideal for weather forecasting, energy management, and outdoor planning, this AI-driven tool ensures reliable wind speed insights.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

# # Sidebar - Project Overview
# st.sidebar.title("Project Overview")
# st.sidebar.markdown("""
# This app predicts wind speed for the next 12 to 48 hours using AI models trained on real-time weather data. Key meteorological factors like temperature, humidity, pressure, precipitation, and cloud cover are used for accurate predictions.

# ### AI Models Used:
# - **Random Forest**: Ensemble of decision trees for robust predictions.
# - **Gradient Boosting**: Sequential learning to minimize errors.
# - **AdaBoost**: Adaptive boosting to improve weak learners.
# - **Support Vector Machine**: Kernel-based regression for non-linear data.
# - **Neural Network**: Multi-layer perceptron for complex pattern recognition.

# ### How It Works:
# 1. **Enter a city name** → Fetches latitude & longitude.
# 2. **Fetch weather data** → Uses Open-Meteo API.
# 3. **Train AI models** → Predicts wind speed trends.
# 4. **Visualize & compare** → Interactive charts show model performance.
# 5. **Download results** → Export predictions for further analysis.

# ### Applications:
# - **Renewable Energy**: Optimizes wind turbine efficiency.
# - **Aviation & Maritime**: Ensures safer flights and shipping.
# - **Disaster Management**: Provides early warnings for storms.
# - **Agriculture & Environment**: Helps in crop and infrastructure protection.
# - **Outdoor Planning**: Supports scheduling based on wind conditions.

# Select models and compare their performance to find the best predictor.
# """)

# # User Inputs
# city_name = st.text_input("Enter City Name")
# model_types = st.multiselect("Select AI Models", options=["Random Forest", "Gradient Boosting", "AdaBoost", "Support Vector Machine", "Neural Network"])
# forecast_duration = st.slider("Select Forecast Duration (Hours)", min_value=12, max_value=48, value=24, step=12)

# if st.button("Get Wind Speed and Weather Data"):
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

#             # Train models
#             results = []
#             for model_type in model_types:
#                 model = train_model(X_train, y_train, X_test, y_test, model_type)
#                 results.append((model_type, model))

#             # Predict wind speed for the next `forecast_duration` hours
#             future_data = pd.DataFrame({
#                 "temperature": data['hourly']['temperature_2m'][:forecast_duration],
#                 "humidity": data['hourly']['relative_humidity_2m'][:forecast_duration],
#                 "pressure": data['hourly']['pressure_msl'][:forecast_duration],
#                 "precipitation": data['hourly']['precipitation'][:forecast_duration],
#                 "cloud_cover": data['hourly']['cloud_cover'][:forecast_duration]
#             })

#             # Visualize predicted wind speed
#             st.subheader("Predicted Wind Speed Over Time")
#             fig = go.Figure()
#             for model_type, model in results:
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

#             # Bar Chart of Predicted Wind Speeds
#             st.subheader("Bar Chart of Predicted Wind Speeds")
#             fig2 = px.bar(df, x="Time", y=[f"Predicted Wind Speed (m/s) - {model_type}" for model_type in model_types], title="Bar Chart of Predicted Wind Speeds")
#             st.plotly_chart(fig2, use_container_width=True)

#             # Correlation Heatmap of Weather Parameters
#             st.subheader("Correlation Heatmap of Weather Parameters")
#             correlation_matrix = weather_df.corr()
#             fig3 = px.imshow(correlation_matrix, text_auto=True, title="Correlation Heatmap")
#             st.plotly_chart(fig3, use_container_width=True)

#             # New Chart 1: Scatter Plot of Wind Speed vs Temperature
#             st.subheader("Wind Speed vs Temperature")
#             fig4 = px.scatter(weather_df, x="temperature", y="wind_speed", trendline="ols", title="Wind Speed vs Temperature")
#             st.plotly_chart(fig4, use_container_width=True)

#             # New Chart 2: 3D Scatter Plot of Wind Speed, Humidity, and Pressure
#             st.subheader("3D Scatter Plot: Wind Speed, Humidity, and Pressure")
#             fig5 = px.scatter_3d(weather_df, x="humidity", y="pressure", z="wind_speed", color="wind_speed", title="3D Scatter Plot: Wind Speed, Humidity, and Pressure")
#             st.plotly_chart(fig5, use_container_width=True)

#             # Display dataset
#             st.subheader("Dataset Used for Prediction")
#             st.write(df)
#             st.download_button(
#                 label="Download Dataset as CSV",
#                 data=df.to_csv(index=False).encode('utf-8'),
#                 file_name="wind_speed_prediction_data.csv",
#                 mime="text/csv"
#             )

#----
# import streamlit as st
# import requests
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import plotly.express as px
# import plotly.graph_objects as go
# import joblib
# import dataset

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

# # Streamlit UI
# st.set_page_config(page_title="Wind Speed Prediction", layout="wide")
# st.title("Wind Speed Prediction Using AI Models")

# # Sidebar - Project Overview
# st.sidebar.title("Project Overview")
# st.sidebar.markdown("""
# This app predicts wind speed for the next 12 to 48 hours using AI models trained on real-time weather data. Key meteorological factors like temperature, humidity, pressure, precipitation, and cloud cover are used for accurate predictions.

# ### AI Models Used:
# - **XGBoost**: Optimized gradient boosting for high accuracy.
# - **LightGBM**: Fast and efficient gradient boosting.
# - **CatBoost**: Handles categorical data natively.
# - **LSTM**: Long Short-Term Memory for time series forecasting.

# ### How It Works:
# 1. **Enter a city name** → Fetches latitude & longitude.
# 2. **Fetch weather data** → Uses Open-Meteo API.
# 3. **Load pre-trained models** → Predicts wind speed trends.
# 4. **Visualize & compare** → Interactive charts show model performance.
# 5. **Download results** → Export predictions for further analysis.
# """)

# # Navigation to dataset download interface
# if st.sidebar.button("Download Dataset"):
#     st.sidebar.write("Redirecting to dataset download interface...")
#     dataset.download_dataset()

# # User Inputs
# city_name = st.text_input("Enter City Name")
# forecast_duration = st.slider("Select Forecast Duration (Hours)", min_value=12, max_value=48, value=24, step=12)

# if st.button("Get Wind Speed and Weather Data"):
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
#                 "wind_direction": data['hourly']['wind_direction_10m'],
#                 "wind_speed": data['hourly']['wind_speed_10m']
#             })

#             # Features and target
#             X = weather_df.drop(columns=["wind_speed"])
#             y = weather_df["wind_speed"]

#             # Load pre-trained models
#             models = {
#                 "XGBoost": joblib.load("xgboost_model.pkl"),
#                 "LightGBM": joblib.load("lightgbm_model.pkl"),
#                 "CatBoost": joblib.load("catboost_model.pkl"),
#                 "LSTM": joblib.load("lstm_model.pkl")
#             }

#             # Predict wind speed for the next `forecast_duration` hours
#             future_data = X[:forecast_duration]
#             results = {}
#             for model_name, model in models.items():
#                 results[model_name] = model.predict(future_data)

#             # Visualize predicted wind speed
#             st.subheader("Predicted Wind Speed Over Time")
#             fig = go.Figure()
#             for model_name, predictions in results.items():
#                 df[f"Predicted Wind Speed (m/s) - {model_name}"] = predictions
#                 fig.add_trace(go.Scatter(x=df["Time"], y=df[f"Predicted Wind Speed (m/s) - {model_name}"], mode="lines+markers", name=f"{model_name}"))

#             fig.update_layout(
#                 title="Predicted Wind Speed (m/s)",
#                 xaxis_title="Time",
#                 yaxis_title="Wind Speed (m/s)",
#                 template="plotly_white",
#                 hovermode="x unified"
#             )
#             st.plotly_chart(fig, use_container_width=True)

#             # Bar Chart of Predicted Wind Speeds
#             st.subheader("Bar Chart of Predicted Wind Speeds")
#             fig2 = px.bar(df, x="Time", y=[f"Predicted Wind Speed (m/s) - {model_name}" for model_name in models.keys()], title="Bar Chart of Predicted Wind Speeds")
#             st.plotly_chart(fig2, use_container_width=True)

#             # Correlation Heatmap of Weather Parameters
#             st.subheader("Correlation Heatmap of Weather Parameters")
#             correlation_matrix = weather_df.corr()
#             fig3 = px.imshow(correlation_matrix, text_auto=True, title="Correlation Heatmap")
#             st.plotly_chart(fig3, use_container_width=True)

#             # Scatter Plot of Wind Speed vs Temperature
#             st.subheader("Wind Speed vs Temperature")
#             fig4 = px.scatter(weather_df, x="temperature", y="wind_speed", trendline="ols", title="Wind Speed vs Temperature")
#             st.plotly_chart(fig4, use_container_width=True)

#             # 3D Scatter Plot of Wind Speed, Humidity, and Pressure
#             st.subheader("3D Scatter Plot: Wind Speed, Humidity, and Pressure")
#             fig5 = px.scatter_3d(weather_df, x="humidity", y="pressure", z="wind_speed", color="wind_speed", title="3D Scatter Plot: Wind Speed, Humidity, and Pressure")
#             st.plotly_chart(fig5, use_container_width=True)

#             # Display dataset
#             st.subheader("Dataset Used for Prediction")
#             st.write(df)
#             st.download_button(
#                 label="Download Dataset as CSV",
#                 data=df.to_csv(index=False).encode('utf-8'),
#                 file_name="wind_speed_prediction_data.csv",
#                 mime="text/csv"
#             )



import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# OpenWeatherMap API Key
API_KEY = "e0100edeedd99f5ae298581c486626a4"  # Replace with your actual API key

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

# Streamlit UI
st.set_page_config(page_title="Wind Speed Prediction", layout="wide")
st.title("Wind Speed Prediction Using AI Models")

# Sidebar - Project Overview
st.sidebar.title("Project Overview")
st.sidebar.markdown("""
This app predicts wind speed for the next 12 to 48 hours using AI models trained on real-time weather data. Key meteorological factors like temperature, humidity, pressure, precipitation, and cloud cover are used for accurate predictions.

### AI Models Used:
- **Linear Regression**: Simple linear regression model.
- **Random Forest Regressor**: Ensemble of decision trees for regression.
- **Gradient Boosting Regressor**: Boosting technique for regression.
- **Support Vector Regressor (SVR)**: Support vector machine for regression.
- **K-Nearest Neighbors (KNN) Regressor**: K-nearest neighbors for regression.

### How It Works:
1. **Enter a city name** → Fetches latitude & longitude.
2. **Fetch weather data** → Uses OpenWeatherMap API.
3. **Train models on the fly** → Predicts wind speed trends.
4. **Visualize & compare** → Interactive charts show model performance.
5. **Download results** → Export predictions for further analysis.
""")

# User Inputs
city_name = st.text_input("Enter City Name")
forecast_duration = st.slider("Select Forecast Duration (Hours)", min_value=12, max_value=48, value=24, step=12)

if st.button("Get Wind Speed and Weather Data"):
    lat, lon = get_coordinates(city_name)
    if lat and lon:
        weather_data = get_weather_data(lat, lon)
        if weather_data:
            # Prepare time and parameter data
            times = [datetime.fromtimestamp(hour['dt']) for hour in weather_data]
            df = pd.DataFrame({
                "Time": times,
                "temperature": [hour['main']['temp'] for hour in weather_data],
                "humidity": [hour['main']['humidity'] for hour in weather_data],
                "pressure": [hour['main']['pressure'] for hour in weather_data],
                "precipitation": [hour.get('rain', {}).get('3h', 0) for hour in weather_data],
                "cloud_cover": [hour['clouds']['all'] for hour in weather_data],
                "wind_speed": [hour['wind']['speed'] for hour in weather_data],
                "wind_direction": [hour['wind']['deg'] for hour in weather_data]
            })

            # Feature Engineering
            df['hour'] = df['Time'].dt.hour
            df['day'] = df['Time'].dt.day
            df['month'] = df['Time'].dt.month

            # Display current weather data in a summary
            st.subheader("Current Weather Summary")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Temperature", f"{df['temperature'][0]}°C")
            col2.metric("Humidity", f"{df['humidity'][0]}%")
            col3.metric("Wind Speed", f"{df['wind_speed'][0]} m/s")
            col4.metric("Pressure", f"{df['pressure'][0]} hPa")
            col5.metric("Precipitation", f"{df['precipitation'][0]} mm")
            col6.metric("Cloud Cover", f"{df['cloud_cover'][0]}%")

            # Prepare dataset for AI model
            X = df.drop(columns=["Time", "wind_speed"])
            y = df["wind_speed"]

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Initialize models
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
                "Support Vector Regressor (SVR)": SVR(kernel='rbf', C=10, gamma='scale'),
                "K-Nearest Neighbors (KNN) Regressor": KNeighborsRegressor(n_neighbors=5)
            }

            # Train models and predict
            results = {}
            accuracies = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[model_name] = y_pred
                accuracies[model_name] = r2

            # Display accuracies
            st.subheader("Model Accuracies")
            accuracy_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "R2 Score"])
            st.write(accuracy_df)

            # Visualize predicted wind speed
            st.subheader("Predicted Wind Speed Over Time")
            fig = go.Figure()
            for model_name, predictions in results.items():
                fig.add_trace(go.Scatter(x=X_test.index, y=predictions, mode="lines+markers", name=f"{model_name}"))

            fig.update_layout(
                title="Predicted Wind Speed (m/s)",
                xaxis_title="Time",
                yaxis_title="Wind Speed (m/s)",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Bar Chart of Predicted Wind Speeds
            st.subheader("Bar Chart of Predicted Wind Speeds")
            fig2 = px.bar(pd.DataFrame(results), title="Bar Chart of Predicted Wind Speeds")
            st.plotly_chart(fig2, use_container_width=True)

            # Correlation Heatmap of Weather Parameters
            st.subheader("Correlation Heatmap of Weather Parameters")
            correlation_matrix = df.corr()
            fig3 = px.imshow(correlation_matrix, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig3, use_container_width=True)

            # Scatter Plot of Wind Speed vs Temperature
            st.subheader("Wind Speed vs Temperature")
            fig4 = px.scatter(df, x="temperature", y="wind_speed", trendline="ols", title="Wind Speed vs Temperature")
            st.plotly_chart(fig4, use_container_width=True)

            # 3D Scatter Plot of Wind Speed, Humidity, and Pressure
            st.subheader("3D Scatter Plot: Wind Speed, Humidity, and Pressure")
            fig5 = px.scatter_3d(df, x="humidity", y="pressure", z="wind_speed", color="wind_speed", title="3D Scatter Plot: Wind Speed, Humidity, and Pressure")
            st.plotly_chart(fig5, use_container_width=True)

            # Display dataset
            st.subheader("Dataset Used for Prediction")
            st.write(df)
            st.download_button(
                label="Download Dataset as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="wind_speed_prediction_data.csv",
                mime="text/csv"
            )
