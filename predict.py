import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Reading in the data

data = pd.read_csv("sphist.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['Date'],axis=0, ascending=True)

# Creating the columns : average for last 5 days, average for last 365 days, standard deviation for the last 365 days

data['5_days'] = pd.rolling_mean(data['Close'],5)
data['365_days'] = pd.rolling_mean(data['Close'],365)
data['std_365'] = data['Close'].rolling(window=365,min_periods = 365).apply(lambda x: np.std(x))

# Shifting so that the data doesn't use the current data in its computation

data = data.shift(periods=1)

# Removing missing values

data_updated = data[data['Date'] > datetime(year=1951, month=1, day=3)]
data_updated.dropna(axis=0, inplace=True)

# Splitting the data

train = data_updated[data_updated['Date'] < datetime(year=2013, month=1, day=1)]
test = data_updated[data_updated['Date'] >= datetime(year=2013, month=1, day=1)]

# Linear regression model

cols = ['5_days','365_days','std_365']

model = LinearRegression()
model.fit(train[cols],train['Close'])
predictions = model.predict(test[cols])
mae = mean_absolute_error(test['Close'],predictions)
print(mae)