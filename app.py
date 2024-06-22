import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import torch
from chronos import ChronosPipeline
import numpy as np



st.title('Market Forecast Pro')
st.sidebar.info('Welcome to the Market Forecast Pro App using traditional ML model and Foundation Model/LLM. Choose your options below')
st.sidebar.info("Created and designed by [Bib Shukla](https://www.linkedin.com/in/bibshukla/)")

def main():
    option = st.sidebar.selectbox('Make a choice', ['Forecast','Visualize','Compare Forecast'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Compare Forecast':
        compare()
    else:
        predict()



@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



option = st.sidebar.text_input('Enter a Stock Symbol', value='AMZN')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')




data = download_data(option, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    st.header('Market Performance Indicators')
    option = st.radio('Choose a Market Performance Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.Close)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Expoenetial Moving Average')
        st.line_chart(ema)

    st.header('Recent Market Data')
    num = st.number_input('How many days recent data?', value=10)
    st.dataframe(data.tail(num))



def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor', 'ChronosPretrainedLanguageModel'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Forecast'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        elif model == 'ChronosPretrainedLanguageModel':
            model_engine_FM(num)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)


def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    print(x_forecast)
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1

def model_engine_FM(num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    
    pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
    torch_dtype=torch.bfloat16,
    )

    forecast = pipeline.predict(
        context=torch.tensor(data["Close"]),
        prediction_length=num,
        num_samples=20,
    )

    forecast_index = range(len(df), len(df) + num)
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    st.text('Forecasting using Chronos model')

    day = 1
    for i in median:
        st.text(f'Day {day}: {i}')
        day += 1

# Function to compare the selections
def compare():
    # Title of the app
    st.title("Compare Selections")

    # Dropdown select boxes
    option1 = st.selectbox("Select Option 1", ["LinearRegression", "RandomForestRegressor", "ExtraTreesRegressor", "KNeighborsRegressor", "XGBoostRegressor"])
    option2 = st.selectbox("Select Option 2", ["ChronosPretrainedLanguageModel"])

    num = st.number_input('How many days forecast?', value=5)

    num1 = num
    num2 = num

    if option1 == option2:
        st.write("Both selections are the same.")
    else:
        st.write(f"Selection 1: {option1}, Selection 2: {option2}")
    if st.button('Compare'):
        if option1 == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num1)
            model_engine_FM(num2)
        elif option1 == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num1)
            model_engine_FM(num2)
        elif option1 == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num1)
            model_engine_FM(num2)
        elif option1 == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num1)
            model_engine_FM(num2)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)
            model_engine_FM(num2)

if __name__ == '__main__':
    main()
