<H1 ALIGN =CENTER> Ex.No: 6 --  HOLT WINTER'S METHOD...</H1>

### Date: 15/10/24 

### AIM :

To create and implement Holt Winter's Method Model using python.

### ALGORITHM :

#### Step 1 : 

You import the necessary libraries.

#### Step 2 : 

You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration.

#### Step 3 : 

You group the data by date and resample it to a monthly frequency (beginning of the month.

#### Step 4 : 

You plot the time series data.

#### Step 5 : 

You import the necessary 'statsmodels' libraries for time series analysis.

#### Step 6 : 

You decompose the time series data into its additive components and plot them.

#### Step 7 : 

You calculate the root mean squared error (RMSE) to evaluate the model's performance.

#### Step 8 : 

You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions.

#### Step 9 : 

You plot the original sales data and the predictions.

### PROGRAM :

```python

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

df=pd.read_csv('dailysales.csv',parse_dates=['date'])
df.info()
df.head()
df.isnull().sum()

df=df.groupby('date').sum()
df.head(10)
df=df.resample(rule='MS').sum()
df.head(10)
df.plot()

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

seasonal_decompose(df,model='additive').plot();
train=df[:19] #till Jul19
test=df[19:] # from aug19
train.tail()
test

from statsmodels.tsa.holtwinters import ExponentialSmoothing
hwmodel=ExponentialSmoothing(train.sales,trend='add', seasonal='mul', seasonal_periods=4).fit()

test_pred=hwmodel.forecast(5)
test_pred
train['sales'].plot(legend=True, label='Train', figsize=(10,6))
test['sales'].plot(legend=True, label='Test')
test_pred.plot(legend=True, label='predicted_test')

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test,test_pred))
df.sales.mean(), np.sqrt(df.sales.var())

final_model=ExponentialSmoothing(df.sales,trend='add', seasonal='mul', seasonal_periods=4).fit()

pred=final_model.forecast(10)
pred
df['sales'].plot(legend=True, label='sales', figsize=(10,6))
pred.plot(legend=True, label='prediction')

```

### OUTPUT :

#### SALES PLOT : 

![img1](https://github.com/anto-richard/TSA_EXP6/assets/93427534/522adc0c-fbd8-4e27-b02c-10346edbfd64)

#### SEASONAL DECOMPOSING (ADDITIVE) :

![img2](https://github.com/anto-richard/TSA_EXP6/assets/93427534/d95fb32f-ed0c-4fb1-abe3-cc3a24d278b5)

#### TEST_PREDICTION :

![img3](https://github.com/anto-richard/TSA_EXP6/assets/93427534/0e84a840-1042-4df2-864b-93a46a3c8346)

#### FINAL_PREDICTION :

![img4](https://github.com/anto-richard/TSA_EXP6/assets/93427534/14bdd28a-d86c-4fd1-8967-cd6757e062be)

### RESULT :

Thus, the program run successfully based on the Holt Winter's Method model.

