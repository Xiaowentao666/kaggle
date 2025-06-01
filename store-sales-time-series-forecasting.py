#%%
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
#%%
store_sales = pd.read_csv(
    'train.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
FAMILY = store_sales['family'].unique()
STORE_NBR = store_sales['store_nbr'].unique()
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
y = store_sales.unstack(['store_nbr', 'family']).loc["2017"]
#%%trend


dp_trend = DeterministicProcess(
    index=y.index,
    constant=True,
    order= 3,
    drop=True,
)
X_trend = dp_trend.in_sample()
model_trend = LinearRegression(fit_intercept=False)
model_trend.fit(X_trend, y)
y_trend = pd.DataFrame(model_trend.predict(X_trend), index=X_trend.index, columns=y.columns)
#%%
#%%
# Create training data
fourier = CalendarFourier(freq='M', order=4)
dp_seasonal = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X_seasonal = dp_seasonal.in_sample()
X_seasonal['NewYear'] = (X_seasonal.index.dayofyear == 1)
#%%
y_seasonal = y - y_trend
model_seasonal = LinearRegression(fit_intercept=False)
model_seasonal.fit(X_seasonal, y_seasonal)
y_pred_seasonal = pd.DataFrame(model_seasonal.predict(X_seasonal), index=X_seasonal.index, columns=y_seasonal.columns)

#%%
y_residuals = y - y_trend - y_pred_seasonal
#%%特征工程
oil = pd.read_csv('oil.csv',parse_dates=["date"])
store = pd.read_csv('stores.csv')
transaction = pd.read_csv("transactions.csv", parse_dates=["date"])
holiday = pd.read_csv("holidays_events.csv", parse_dates=["date"])
#%%
df_test = pd.read_csv(
    'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
df_test['date'] = df_test.date.dt.to_period('D')
df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()
#%%
# Create features for test set
X_test_trend = dp_trend.out_of_sample(steps=16)
X_test_seasonal = dp_seasonal.out_of_sample(steps=16)
X_test_trend.index.name = 'date'
X_test_seasonal.index.name = 'date'
X_test_seasonal['NewYear'] = (X_test_seasonal.index.dayofyear == 1)
#%%
y_test_trend = pd.DataFrame(model_trend.predict(X_test_trend),index=X_test_trend.index,columns=y.columns)
y_test_seasonal = pd.DataFrame(model_seasonal.predict(X_test_seasonal),index=X_test_seasonal.index,columns=y.columns)
y_test = y_test_seasonal + y_test_trend

# %%

y_test = y_test.stack(['store_nbr', 'family'])
y_test = y_test.join(df_test.id).reindex(columns=['id', 'sales'])
y_test.to_csv('submission_2.csv', index=False)

#%%trend图
import matplotlib.pyplot as plt
# moving_average = y.rolling(
#     window=7,       # 365-day window
#     center=True,      # puts the average at the center of the window
#     min_periods=3,  # choose about half the window size
# ).mean()

plt.rcParams['figure.dpi'] = 3000
STORE_NBR = '2'  # 1 - 54
FAMILY = 'PRODUCE'
ax = (y-y.mean()).loc(axis=1)['sales', STORE_NBR, FAMILY].plot()
ax = y_seasonal.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(ax=ax)
ax = y_pred_seasonal.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(ax=ax)
ax.legend()
#%%
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
STORE_NBR = '35'  # 1 - 54
FAMILY = 'PRODUCE'
ax = (y_residuals).loc(axis=1)['sales', STORE_NBR, FAMILY].plot()
# ax = y_test_seasonal.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(ax=ax)

# ax = y_test.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(ax=ax)