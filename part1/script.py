import pandas as pd
import quandl
import datetime
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


df = quandl.get('WIKI/GOOGL')


df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / \
    df['Adj. Open'] * 100.0


df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'], 1))
X = X[:-forecast_out]
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2)


clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(accuracy)
