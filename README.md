# Market Forecast Pro App

Welcome to the Market Forecast Pro App! This app allows you to visualize stock price data, explore technical indicators, and make short-term price predictions using traditional machine learning models and language models.

Created and designed by [Bib Shukla](https://www.linkedin.com/in/bibshukla/).

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Technologies](#technologies)
- [License](#license)

## Description

The Market Forecast Pro App is a Streamlit-based web application that provides users with tools to analyze historical stock price data, visualize market technical indicators, and make short-term price predictions using different machine learning models and language models.

## Features

- **Visualize Technical Indicators**: Explore various technical indicators such as Bollinger Bands, MACD, RSI, SMA, and EMA to gain insights into stock price trends. Also view the most recent data of the selected stock, including the last 10 data points.

- **Forecast**: Predict future stock prices using machine learning models including Linear Regression, Random Forest Regressor, Extra Trees Regressor, KNeighbors Regressor, XGBoost Regressor & Chronos language model.

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/bibshukla/MarketForecastPro.git
   ```

2. Navigate to the project directory:
   ```sh
   cd marketforecastpro
   ```

3. Install the required Python packages using pip:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

2. The app will open in your default web browser. Use the sidebar to choose options for visualization or making price predictions.

3. Follow the on-screen instructions to input the stock symbol, select a date range, and choose technical indicators or prediction models.

## Technologies

- Python
- Streamlit
- pandas
- yfinance
- ta (Technical Analysis Library)
- scikit-learn
- XGBoost
- Chronos
    Chronos - https://github.com/amazon-science/chronos-forecasting
    Chronos is a family of pretrained time series forecasting models based on language model architectures. A time series is transformed into a sequence of tokens via scaling and quantization, and a language model is trained on these tokens using the cross-entropy loss. Once trained, probabilistic forecasts are obtained by sampling multiple future trajectories given the historical context.
- This code is an enhancement of code originally published - https://github.com/vikasharma005/Stock-Price-Prediction

  ## Author

<div id="header" align="center">
  <a href="https://www.linkedin.com/in/bibshukla">
  <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="100"/>
  </a>
</div>

<h3 align="center">Hi there 👋, I'm Bib</h3>
<h4 align="center">Always learning and curious😀</h4>

<div id="socials" align="center">
  <a href="https://www.linkedin.com/in/bibshukla">
    <img src="https://user-images.githubusercontent.com/76098066/186728913-a66ef85f-4644-4e3a-b847-98309c8cff42.svg">
  </a>
</div>

You can find more about me and my projects on my [GitHub profile](https://github.com/bibshukla).

## License

This project is licensed under the [MIT License](LICENSE).

---



