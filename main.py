import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from concurrent.futures import ThreadPoolExecutor
import requests

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Download VADER lexicon
nltk.download('vader_lexicon')

# Function to preprocess news
def preprocess_news(news_article, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences([news_article])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
sid = SentimentIntensityAnalyzer()

# Fetch data from Yahoo Finance with retries
def fetch_data_from_yahoo(ticker, period='10y', max_retries=3):
    for _ in range(max_retries):
        try:
            data = yf.download(ticker, period=period, interval='1d')
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            data.reset_index(inplace=True)
            data.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
            data['NetChange'] = data['Close'].diff()
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            time.sleep(2)
    raise ValueError(f"Failed to fetch data for {ticker} after {max_retries} attempts")

# Fetch live data from Yahoo Finance with retries
def fetch_live_data_from_yahoo(ticker, max_retries=3):
    for _ in range(max_retries):
        try:
            live_data = yf.download(ticker, period='1d', interval='1m')
            if live_data.empty:
                raise ValueError(f"No live data found for ticker {ticker}")
            live_data.reset_index(inplace=True)
            return live_data
        except Exception as e:
            print(f"Error fetching live data for {ticker}: {e}")
            time.sleep(2)
    raise ValueError(f"Failed to fetch live data for {ticker} after {max_retries} attempts")

# Function to handle multiple tickers
def process_ticker(ticker, news_article):
    print(f"Processing ticker: {ticker}")

    df = fetch_data_from_yahoo(ticker)

    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The fetched DataFrame is empty. Check the ticker symbol or the period specified.")

    # Normalize data
    sc_close = MinMaxScaler()
    sc_sentiment = MinMaxScaler()

    # Scale Close data
    df['Close_scaled'] = sc_close.fit_transform(df[['Close']].values)

    # Preprocess the news input
    max_length = 100  # Maximum length of input sequences
    if not tokenizer.word_index:  # Fit tokenizer if not already fitted
        tokenizer.fit_on_texts([news_article])
    preprocessed_news = preprocess_news(news_article, tokenizer, max_length)

    # Perform sentiment analysis
    sentiment_scores = sid.polarity_scores(news_article)
    print(f"Sentiment Scores: {sentiment_scores}")

    # Add sentiment scores to the DataFrame
    df['Sentiment_neg'] = sentiment_scores['neg']
    df['Sentiment_neu'] = sentiment_scores['neu']
    df['Sentiment_pos'] = sentiment_scores['pos']
    df['Sentiment_compound'] = sentiment_scores['compound']

    # Scale sentiment data consistently
    sentiment_columns = ['Sentiment_neg', 'Sentiment_neu', 'Sentiment_pos', 'Sentiment_compound']
    df[sentiment_columns] = sc_sentiment.fit_transform(df[sentiment_columns])

    # Combine close price data and sentiment data
    X_combined = np.column_stack((
        df['Close_scaled'],
        df['Sentiment_neg'],
        df['Sentiment_neu'],
        df['Sentiment_pos'],
        df['Sentiment_compound']
    ))

    # Split data into samples
    X_samples, y_samples = [], []
    TimeSteps = 200  # Next day's Price Prediction is based on last 200 past day's prices
    for i in range(TimeSteps, len(X_combined)):
        x_sample = X_combined[i - TimeSteps:i]
        y_sample = X_combined[i, 0]  # Next day's Close price
        X_samples.append(x_sample)
        y_samples.append(y_sample)

    X_data, y_data = np.array(X_samples), np.array(y_samples)

    # Train on the last 500 days of data
    X_train = X_data[-500:]
    y_train = y_data[-500:]

    # Define input shapes for LSTM
    TimeSteps = X_train.shape[1]
    TotalFeatures = X_train.shape[2]

    # Model initialization and training with performance tuning
    model_Close = Sequential([
        Bidirectional(LSTM(units=128, activation='relu', return_sequences=True), input_shape=(TimeSteps, TotalFeatures)),  # Reduced units
        Dropout(0.3),  # Adjusted dropout rate
        Bidirectional(LSTM(units=64, activation='relu', return_sequences=True)),  # Reduced units
        Dropout(0.3),  # Adjusted dropout rate
        LSTM(units=32, activation='relu'),
        Dense(units=16, activation='relu'),
        Dropout(0.2),  # Adjusted dropout rate
        Dense(units=1)
    ])

    model_Close.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping and model checkpoint
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'/content/best_model_{ticker}.keras', monitor='val_loss', save_best_only=True)

    # Model training with adjusted batch size
    StartTime = time.time()
    history = model_Close.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0, validation_split=0.1, callbacks=[early_stop, checkpoint])  # Further reduced batch size and epochs
    EndTime = time.time()
    print("Total Time Taken by the model: ", round((EndTime - StartTime) / 60, 2), 'Minutes')

    # Save the model after training
    model_Close.save(f'/content/best_model_{ticker}.keras')  # Save the model

    # Load best model
    model_Close = tf.keras.models.load_model(f'/content/best_model_{ticker}.keras')

    # Get the last date in the dataset
    last_date = df['Date'].iloc[-1]

    # Calculate the next date
    next_date = last_date + datetime.timedelta(days=1)

    # Extract features for the last available date
    last_X_sample = X_data[-1].reshape(1, TimeSteps, TotalFeatures)

    # Predict the close price for the next day
    next_day_prediction_scaled = model_Close.predict(last_X_sample)

    # Inverse transform the prediction to get the actual price
    next_day_prediction = sc_close.inverse_transform(next_day_prediction_scaled)[0][0]

    # Last trained close price
    last_trained_close_price_scaled = y_train[-1]
    last_trained_close_price = sc_close.inverse_transform([[last_trained_close_price_scaled]])[0][0]

    # Difference between predicted and last trained close price
    difference = next_day_prediction - last_trained_close_price

    # Explanation
    explanation = "The predicted price might differ due to various factors such as market sentiment, news, or unexpected events not captured in the training data."

    # Print the results
    print(f"Ticker: {ticker}")
    print(f"Last trained close price (around {last_date}): {last_trained_close_price:.2f}")
    print(f"Predicted Close price for {next_date}: {next_day_prediction:.2f}")
    print(f"Difference: {difference:.2f}")
    print("Reason:", explanation)

    # Plotting loss during training
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss over Epochs for {ticker}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main interactive loop for multiple tickers
def main():
    while True:
        # Input news article
        news_article = input("Paste your news article (press Enter to exit): ").strip()
        if news_article == "":
            break

        # Input tickers
        tickers = input("Enter ticker symbols separated by commas (e.g., AAPL, MSFT): ").strip().split(',')

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_ticker, ticker.strip(), news_article) for ticker in tickers]
            for future in futures:
                future.result()

        # Live Data Monitoring
        for ticker in tickers:
            print(f"\nMonitoring live data for {ticker}...")
            try:
                live_data = fetch_live_data_from_yahoo(ticker.strip())
                print(live_data.tail())
            except Exception as e:
                print(f"Failed to fetch live data for {ticker}: {e}")

if __name__ == "__main__":
    main()
