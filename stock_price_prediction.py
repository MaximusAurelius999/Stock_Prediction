import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from sklearn.linear_model import LinearRegression

def get_stock_symbols():
    stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]  # Add more symbols as needed
    return stock_symbols

def get_historical_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

def predict_stock_price(stock_data, days):
    df = stock_data.reset_index()
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')  # Convert Date column to string format

    X = df.index.values.reshape(-1, 1)
    y = df['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=days+1, closed='right')
    future_dates = future_dates.strftime('%Y-%m-%d').tolist()
    future_dates = future_dates[1:]  # Exclude the last date in the historical data

    future_prices = []
    for i in range(days):
        future_prices.append(model.predict([[df.index[-1] + i + 1]]))

    return future_dates, future_prices

def main():
    st.title("Stock Price Prediction")

    stock_symbols = get_stock_symbols()
    stock_symbol = st.sidebar.selectbox("Select a stock symbol", stock_symbols)

    today = date.today()
    start_date = st.sidebar.date_input("Start date", today.replace(year=today.year-1))
    end_date = st.sidebar.date_input("End date", today)

    if start_date >= end_date:
        st.error("Error: End date must be after start date.")
        return

    stock_data = get_historical_data(stock_symbol, start_date, end_date)

    if stock_data.empty:
        st.error("Error: No data available for the selected stock symbol.")
        return

    days = st.sidebar.number_input("Number of days for prediction", value=30)

    future_dates, future_prices = predict_stock_price(stock_data, days)

    st.subheader("Historical Data")
    st.line_chart(stock_data['Close'])

    st.subheader("Forecasted Data")
    forecast_data = {'Date': future_dates, 'Close': future_prices}
    forecast_df = stock_data.reset_index().append(forecast_data, ignore_index=True)
    forecast_df['Date'] = forecast_df['Date'].astype(str)  # Convert Date column to string format
    forecast_df = forecast_df.drop_duplicates(subset='Date')  # Drop duplicate rows based on 'Date' column
    forecast_df['Close'] = pd.to_numeric(forecast_df['Close'], errors='coerce')  # Convert 'Close' column to numeric type
    forecast_df = forecast_df.dropna(subset=['Close'])  # Remove rows with NaN values
    st.dataframe(forecast_df.tail(days))

    st.subheader("Forecasted Line Chart")
    st.line_chart(forecast_df.set_index('Date')['Close'])

    st.subheader("Forecasted Bar Chart")
    st.bar_chart(forecast_df.set_index('Date')['Close'])

if __name__ == "__main__":
    main()
