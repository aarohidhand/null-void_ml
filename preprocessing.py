import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = "D:/Technex/weather_data.csv"
df = pd.read_csv(file_path)

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    
df = df.dropna(subset=["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["year"] = df["timestamp"].dt.year
df["day_of_week"] = df["timestamp"].dt.dayofweek

df.fillna(method='ffill', inplace=True)

# Normalize the numerical data
numerical_features = ["temperature_C", "wind_speed_mps", "solar_radiation_Wm2"]
if all(feature in df.columns for feature in numerical_features):
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

df.to_csv("preprocessed_weather_data.csv", index=False)

print("Preprocessed weather data saved successfully!")
