import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set up the app title
st.title("Tesla Stock Price Prediction")

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('tesla_stock.csv')  # Replace with your file's actual name
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()

# Display the dataset
st.subheader("Tesla Stock Data")
st.write(data.head())

# Visualization
st.subheader("Stock Price Over Time")
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Close'], label="Close Price", color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# Feature Engineering
st.subheader("Feature Engineering")
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
features = ['Year', 'Month']
target = 'Close'

# Train/Test Split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Model Mean Squared Error: {mse:.2f}")

# User Input for Prediction
st.subheader("Make a Prediction")
year = st.number_input("Enter Year", min_value=2010, max_value=2030, value=2023)
month = st.number_input("Enter Month", min_value=1, max_value=12, value=1)
input_data = np.array([[year, month]])
prediction = model.predict(input_data)

st.write(f"Predicted Close Price for {year}-{month:02d}: ${prediction[0]:.2f}")
