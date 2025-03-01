import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("rajasthan_solar_wind_data_updated.csv")

df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.set_index("DateTime")

features = ["Solar Irradiance (W/m²)", "Temperature (°C)", "Cloud Cover (%)", "Wind Speed (m/s)", "Air Pressure (hPa)"]
targets = ["Solar Power Output (kWh)", "Wind Power Output (kWh)"]

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features + targets])

df_scaled = pd.DataFrame(df_scaled, columns=features + targets, index=df.index)

sequence_length = 24

def create_sequences(data, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][features].values)
        y.append(data.iloc[i+seq_length][target_columns].values)
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, targets, sequence_length)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(sequence_length, len(features))),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(50, activation="relu"),
    Dense(len(targets), activation="linear")  
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))