import pandas as pd
import yfinance as yf
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

#Functions for data manipulation and cleaning

def get_stock_data(ticker_symbol,start_date):
    # Fetch data from Yahoo Finance
    data = yf.download(ticker_symbol, start=start_date)
    data = pd.DataFrame(data)
    return data

def fetch_news(ticker_symbol):
    # This function should return a list of news articles for the given ticker symbol
    # For demonstration, this will return a dummy list
    return [
        {"title": "Company ABC reports record profits", "Date": "2023-04-01"},
        {"title": "Company ABC faces lawsuit over patent", "Date": "2023-04-02"},
        # Add more news articles
    ]

def analyze_sentiment(article):
    analysis = TextBlob(article['title'])  # You can also include article content if available
    return analysis.sentiment.polarity

'''Aggregate these sentiment scores by date'''

def aggregate_sentiments(news_articles):
    sentiments = {}
    for article in news_articles:
        date = article['Date']
        sentiment = analyze_sentiment(article)
        if date in sentiments:
            sentiments[date].append(sentiment)
        else:
            sentiments[date] = [sentiment]

    # Average the sentiments for each date
    #Can do this post function
    '''avg_sentiments = {date: sum(scores) / len(scores) for date, scores in sentiments.items()}
    return avg_sentiments'''
    return sentiments

def combine_data(stock_data, sentiment_data):
    # Convert sentiment_data dictionary to a DataFrame
    sentiment_df = pd.DataFrame(list(sentiment_data.items()), columns=['Date', 'Sentiment'])
    #Make Date a column instead of index
    stock_data = stock_data.reset_index()
    # Convert date columns to datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

    # Merge the DataFrames on the date column
    combined_data = pd.merge(stock_data, sentiment_df, on='Date', how='left')

    # Handle missing values
    combined_data['Sentiment'].fillna(0, inplace=True)

    return combined_data

def prepare_dataset(combined_data):

    return combined_data

def add_indicators(data):
    # Simple Moving Average (SMA) for the 'close' price
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def model_func(data):
    df = data.fillna(0)

    # 5. Split Data into Training and Test Sets
    X = df[['Open', 'High', 'Low', 'Volume','SMA_20','RSI','Sentiment']]
    y = df['Close']  # Assuming you want to predict the closing price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Select a Model
    # Here we use a Random Forest, but you can select another model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # 7. Train the Model
    model.fit(X_train, y_train)

    # 8. Evaluate the Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

    '''# Optional: Plotting feature importance
    features = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()'''