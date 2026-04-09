import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title
st.title("💼 Salary Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Salary_Data.csv")

df = load_data()

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Features and target
X = df[['YearsExperience']]
y = df['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Sidebar input
st.sidebar.header("Enter Experience")

years = st.sidebar.slider("Years of Experience", 0.0, 15.0, 2.0)

# Prediction
input_data = np.array([[years]])
prediction = model.predict(input_data)[0]

# Output
st.subheader("Prediction Result")
st.write(f"💰 Predicted Salary: **${prediction:.2f}**")

# Plot
st.subheader("Visualization")

fig, ax = plt.subplots()

# Scatter plot
ax.scatter(df['YearsExperience'], df['Salary'], label='Actual Data')

# Regression line
x_range = np.linspace(df['YearsExperience'].min(),
                      df['YearsExperience'].max(), 100)

y_line = model.predict(x_range.reshape(-1, 1))

ax.plot(x_range, y_line, label='Regression Line')

# Predicted point
ax.scatter(years, prediction, s=100, label='Prediction')

ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.legend()

st.pyplot(fig)

# Model details
st.subheader("Model Details")

st.write("Intercept:", model.intercept_)
st.write("Slope (Coefficient):", model.coef_[0])