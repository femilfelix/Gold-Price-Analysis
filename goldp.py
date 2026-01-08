import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Gold Price Profit/Loss & Forecast", layout="centered")
st.title("🏆 Gold Price Profit/Loss & Forecast Analyzer")

# -----------------------------
# Load dataset
# -----------------------------
try:
    df = pd.read_csv("gold_price.csv")
    st.success("✅ gold_price.csv loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading file: {e}")
    st.stop()

# -----------------------------
# Check for Date column
# -----------------------------
if 'Date' not in df.columns:    # Check if dataset has a column named 'Date'
    st.warning("⚠ 'Date' column not found — trying to detect automatically...") # Try to find any column name containing the word 'date
    possible_dates = [col for col in df.columns if 'date' in col.lower()]
    if possible_dates:
        df.rename(columns={possible_dates[0]: 'Date'}, inplace=True) # Rename it to 'Date'
    else:
        st.error("❌ Could not find a date column in the dataset.")
        st.stop() # Stop if no valid date column found

# Convert to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True) # Drop rows with invalid/missing dates
df.set_index('Date', inplace=True)  # Set Date as the DataFrame index
df.sort_index(inplace=True)  # Sort data by date (oldest to newest)

# -----------------------------
# Validate price column
# -----------------------------
if 'USD (PM)' not in df.columns:  # Ensure price column exists
    st.error("❌ The CSV must contain a column named 'USD (PM)'.")
    st.stop()   # Stop if column missing

# Calculate returns
df['Return'] = df['USD (PM)'].pct_change() * 100   # Calculate daily percentage change
df.dropna(inplace=True)   #Drop first row (since pct_change gives NaN)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("⚙️ User Controls")
min_year = int(df.index.min().year)  # Detect min and max years available in dataset
max_year = int(df.index.max().year)

# Input boxes and sliders for user input
start_year = st.sidebar.number_input("Start Year", min_value=min_year, max_value=max_year, value=2010)
end_year = st.sidebar.number_input("End Year", min_value=start_year, max_value=max_year, value=max_year)
forecast_year = st.sidebar.slider("📅 Predict prices till year", min_value=max_year, max_value=2025, value=2025)

# -----------------------------
# Profit / Loss analysis
# -----------------------------
selected_period = df.loc[str(start_year):str(end_year)] # Filter data between selected years

st.subheader(f"📆 Gold Price Analysis: {start_year} to {end_year}")

if selected_period.empty:
    st.warning("⚠ No data available for the selected years.")  # Warn if no data found
else:
    start_price = selected_period['USD (PM)'].iloc[0] # Get first and last prices in selected range
    end_price = selected_period['USD (PM)'].iloc[-1]
    profit_loss = ((end_price - start_price) / start_price) * 100  # Calculate profit/loss percentage


# Display profit or loss result
    if profit_loss > 0:
        st.success(f"💰 Profit of **{profit_loss:.2f}%** between {start_year} and {end_year}")
    else:
        st.error(f"📉 Loss of **{abs(profit_loss):.2f}%** between {start_year} and {end_year}")

# -----------------------------
# Forecasting using Linear Regression
# -----------------------------
st.subheader("🔮 Gold Price Prediction")

# Prepare features and target
df['Year'] = df.index.year  # Extract year from date
X = df[['Year']]    # Independent variable
y = df['USD (PM)']   # Dependent variable (price)

# Train the model
model = LinearRegression()
model.fit(X, y)

# Create future years for prediction
future_years = np.arange(df['Year'].max() + 1, forecast_year + 1)
future_df = pd.DataFrame({'Year': future_years})
future_df['Predicted_Price'] = model.predict(future_df[['Year']])  # Predict prices for those future years

# Create date index for future years
last_date = df.index.max()  # Last date in dataset
# Generate date range for upcoming years
future_dates = pd.date_range(start=last_date + pd.offsets.YearBegin(1),
                             periods=len(future_years),
                             freq='YS')
future_df.index = future_dates  # Assign as index

# Combine actual + forecast for continuous plotting
combined = pd.concat([
    df[['USD (PM)']].rename(columns={'USD (PM)': 'Gold Price'}),   # Actual prices
    future_df[['Predicted_Price']]  # Forecasted prices
])

# -----------------------------
# Plot actual + predicted chart
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(combined.index, combined['Gold Price'], color='gold', label='Actual Gold Price') # Plot actual prices
ax.plot(combined.index, combined['Predicted_Price'], '--r', label='Predicted Price (Forecast)') #plot predicted prices
ax.set_title(f"Gold Price Trend & Forecast till {forecast_year}")  #chart title
ax.set_xlabel("Year")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Show forecasted data
# -----------------------------
st.write("### 📈 Forecasted Prices (in USD)")  # Section title
st.dataframe(future_df[['Predicted_Price']].style.format("{:.2f}"))  # Display predicted prices table

# -----------------------------
# Dataset info
# -----------------------------
st.info(f"📅 Data covers {df.index.min().year}–{df.index.max().year} with {len(df)} records.")  # Show dataset summary