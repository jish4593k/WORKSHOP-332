import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression, LassoLars, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def convert_dates(row):
    row['Date'] = datetime.datetime.strptime(row['Date'], '%Y-%m-%d')
    return row

def load_data(stock):
    df = pd.read_csv('./datasets/' + stock + '.csv')
    mpl.rc('figure', figsize=(16, 12))
    return df.apply(convert_dates, axis=1)

def create_features(df):
    rolling_mean_window_size = 200
    dfreg = df.loc[:, ['Date', 'Adj Close', 'Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    dfreg['AVG_rolling_PCT_change'] = dfreg['PCT_change'].rolling(window=10).mean()
    dfreg['AVG_rolling_eighth'] = df['Adj Close'].rolling(window=rolling_mean_window_size//8).mean()
    dfreg['AVG_rolling_half'] = df['Adj Close'].rolling(window=rolling_mean_window_size//2).mean()
    dfreg['AVG_rolling_full'] = df['Adj Close'].rolling(window=rolling_mean_window_size).mean()
    return dfreg

def preprocess_data(dfreg, forecast_length=50):
    dfreg.set_index('Date', inplace=True)
    dfreg = dfreg[rolling_mean_window_size:]
    dfreg.fillna(value=-99999, inplace=True)

    dfreg['label'] = dfreg['Adj Close'].shift(-forecast_length)
    X = np.array(dfreg.drop('label', 1))
    X = scale(X)

    X_train = X[:-forecast_length]
    Y_train = np.array(dfreg['label'])[:-forecast_length]

    X_pred = X[-forecast_length:]

    return X_train, Y_train, X_pred

def train_linear_regression(X_train, Y_train):
    linreg = LinearRegression(n_jobs=-1)
    linreg.fit(X_train, Y_train)
    return linreg

def train_polynomial_regression(X_train, Y_train):
    quadreg = make_pipeline(PolynomialFeatures(2), LassoLars(alpha=0.01, max_iter=10000))
    quadreg.fit(X_train, Y_train)
    return quadreg

def train_bayesian_regression(X_train, Y_train):
    bayesreg = make_pipeline(PolynomialFeatures(2), BayesianRidge(n_iter=10000, compute_score=True))
    bayesreg.fit(X_train, Y_train)
    return bayesreg

def train_lstm(X_train_tensor, Y_train_tensor):
    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size
            self.lstm = nn.LSTM(input_size, hidden_layer_size)
            self.linear = nn.Linear(hidden_layer_size, output_size)
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                                torch.zeros(1, 1, self.hidden_layer_size))

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    Y_train_lstm = scaler.fit_transform(Y_train.reshape(-1, 1))
    X_train_lstm = scaler.fit_transform(X_train)

    
    X_train_tensor = torch.tensor(X_train_lstm, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_lstm, dtype=torch.float32)

   
    lstm_model = LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    for epoch in range(100):
        for seq, labels in zip(X_train_tensor, Y_train_tensor):
            optimizer.zero_grad()
            lstm_model.hidden_cell = (torch.zeros(1, 1, lstm_model.hidden_layer_size),
                                      torch.zeros(1, 1, lstm_model.hidden_layer_size))

            y_pred = lstm_model(seq)

            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer.step()

    return lstm_model

def make_forecasts(models, X_pred):
    forecasts = [model.predict(X_pred) for model in models]
    return forecasts

def display_results(lin_forecast, quad_forecast, bayes_forecast, lstm_forecast):
    # Display results or use the forecasts as needed
    print("Linear Regression Forecast:", lin_forecast)
    print("Quadratic Regression Forecast:", quad_forecast)
    print("Bayesian Regression Forecast:", bayes_forecast)
    print("LSTM Forecast:", lstm_forecast)

if __name__ == '__main__':
    stock_code = 'AAPL'  # replace with the desired stock code

    df = load_data(stock_code)
    dfreg = create_features(df)
    X_train, Y_train, X_pred = preprocess_data(dfreg)
    
    linreg = train_linear_regression(X_train, Y_train)
    quadreg = train_polynomial_regression(X_train, Y_train)
    bayesreg = train_bayesian_regression(X_train, Y_train)
    
    lstm_model = train_lstm(X_train, Y_train)
    
    models = [linreg, quadreg, bayesreg, lstm_model]
    forecasts = make_forecasts(models, X_pred)
    
    lin_forecast, quad_forecast, bayes_forecast, lstm_forecast = forecasts
    display_results(lin_forecast, quad_forecast, bayes_forecast, lstm_forecast)
