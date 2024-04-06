import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM
from keras.models import Sequential
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# from keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.layers import Dense, Dropout

input_file = "BitcoinHeistData.csv"


# # convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


#
#
# # fix random seed for reproducibility
np.random.seed(5)

# load the dataset ['address', 'year', 'day', 'length', 'weight', 'count', 'looped', 'neighbors', 'income', 'label']
df = read_csv(input_file, dtype='unicode')
nan_count = df.isna().sum()

print(nan_count)
# dropping the categorical columns
df1 = df.drop(['address', 'label'], axis=1)
print(df1.describe())
nan_count = df1.isna().sum()

print(nan_count)

df2 = df1.dropna()
print(df2.describe())
nan_count = df2.isna().sum()

print(nan_count)
# columns = df.columns.tolist()
# print(df.head(1))
# weight_list = df['weight'].tolist()
# print(weight_list)
# for column in columns:
#     print(column, ":", type(column))


# take close price as income
all_y = df2['income'].values
dataset = all_y.reshape(-1, 1)
#
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#
# # split into train and test sets, 50% test data, 50% training data
train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
#
# reshape into X=t and Y=t+1, timestep 240
look_back = 240
train_x, train_y = create_dataset(train, look_back)
test_x, test_y = create_dataset(test, look_back)
#
# reshape input to be [samples, time steps, features]
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
#
# # create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()
model.add(LSTM(25, input_shape=(1, look_back)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(train_x, train_y, epochs=10, batch_size=240, verbose=1)
#
# make predictions
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)
#
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])
#
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
print('Train Score: %.2f RMSE' % trainScore)
testScore = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
print('Test Score: %.2f RMSE' % testScore)
#
# # shift train predictions for plotting
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict
#
# shift test predictions for plotting
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataset) - 1, :] = test_predict
#
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(train_predict_plot)
print('testPrices:')
test_prices = scaler.inverse_transform(dataset[test_size + look_back:])
print(test_prices)
#
print('testPredictions:')
print(test_predict)
#
# # export prediction and actual prices
# df = pd.DataFrame(data={"prediction": np.around(list(test_predict.reshape(-1)), decimals=2),
#                         "test_price": np.around(list(test_prices.reshape(-1)), decimals=2)})
# df.to_csv("lstm_result.csv", sep=';', index=None)
#
# # plot the actual price, prediction in test data=red line, actual price=blue line
plt.plot(test_predict_plot)
plt.show()
