# Energy Forecasting Model

This project predicts solar and wind power output based on weather data using an LSTM-based deep learning model.

## Features
- Uses past weather data to train an LSTM model
- Fetches real-time weather data based on user location
- Predicts solar and wind power output for the next period
- Saves and loads trained models for future use

## Requirements
Install the necessary dependencies using:
```sh
pip install tensorflow pandas numpy scikit-learn joblib requests python-dotenv
```

## Dataset
The model requires a CSV file (`rajasthan_solar_wind_data_updated.csv`) containing:
- DateTime (timestamp)
- Solar Irradiance (W/m²)
- Temperature (°C)
- Cloud Cover (%)
- Wind Speed (m/s)
- Air Pressure (hPa)
- Solar Power Output (kWh)
- Wind Power Output (kWh)

## How It Works
1. **Data Preprocessing**
   - The dataset is normalized using MinMaxScaler.
   - Weather features are used to create input sequences of 24-hour data points.
2. **Model Training**
   - A deep learning model with LSTM layers is trained using past weather data.
   - The model is saved as `energy_forecast_model.h5`.
   - Scalers for feature normalization are saved using `joblib`.
3. **Fetching Real-Time Weather Data**
   - The user inputs a location.
   - The script fetches latitude and longitude using OpenCage API.
   - Historical weather data for the past 24 hours is retrieved using Open-Meteo API.
4. **Making Predictions**
   - The weather data is normalized.
   - The LSTM model predicts solar and wind power output.
   - The predictions are converted back to the original scale and displayed.

## Setup
1. Add an `.env` file containing:
   ```sh
   OPENCAGE_API_KEY=your_api_key_here
   ```
2. Run the script:
   ```sh
   python script.py
   ```
3. Enter the location when prompted.

## Output
The model prints:
```sh
Predicted Solar Power Output: X.XX kWh
Predicted Wind Power Output: X.XX kWh
```

## Notes
- Ensure `.env` is properly configured with an OpenCage API key.
- If weather data is insufficient, try again later.

## Future Improvements
- Enhance model accuracy with more training data.
- Add real-time forecasting capabilities.
- Improve handling of missing weather data.



