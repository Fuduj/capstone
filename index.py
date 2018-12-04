# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:19:25 2018

@author: fudu
"""
"                      Step 1: import Proquest data             "
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

A=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\A.csv")
B=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\B.csv")
C=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\C.csv")
D=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\D.csv")
E=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\E.csv")

df=A.append(B)
df=df.append(C)
df=df.append(D)
df=df.append(E)
df=df[df.year>1961]


"                   Step 2: create frequency indices           "

" Separate 'Month' from publication date"
df2=pd.DataFrame(df.pubdate.str.split(' ',1).tolist(),columns = ['Month',''])
                                  
"Adding month column to proquest dataframe"
df['Month'] = df2['Month']
df=df.sort_values('year')
annual_index=df.groupby('year').count()

" Drop other attributes"
annual_index=annual_index.drop(['pubdate', 'pubtitle', 'Month'],axis=1)


"Create year attribute"
annual_index['Year']=annual_index.index

"Convert annual index to integer"
annual_index=annual_index.astype(int)

"Upload total 'Policy' and 'Uncertainty' files "

file_name = 'C:\\Users\\fudu\\Desktop\\Capstone\\EP_articles\\Raw\\{}.csv'
policy = pd.concat([pd.read_csv(file_name.format(i)) for i in range(0, 48)])


"Create one dataframe with total uncertainty/policy count"

policy=policy.sort_values('year')
policy_index=policy.groupby('year').count()
policy_index=policy_index.drop(['pubdate', 'pubtitle'],axis=1)
policy_index['Year']=policy_index.index
policy_index=policy_index[policy_index.index>1963]


"Upload business investment"
annual_inv=pd.read_csv("C:\\Users\\fudu\\Desktop\\Capstone\\annual_business_investment.csv").astype(int)
annual_inv.columns=['year','total inv', 'total business inv', 'non-res', 'M&E','IP']
annual_inv.index=annual_inv['year']
annual_inv=annual_inv[annual_inv.index>1963]

"Step 3: Test for correlations"

"Test contemporaneous correlations between share of uncertainty and measures of BI"

correl_2=pd.DataFrame()
correl_2['EP']=policy_index['Title'].astype(int)
correl_2['EPU']=annual_index['title']
correl_2['share']=(correl_2['EPU']/correl_2['EP'])*100
correl_2 = correl_2.drop(['EP', 'EPU'],axis=1)
correl_2['TI']=annual_inv['total inv']
correl_2['BI']=annual_inv['total business inv']
correl_2['NR']=annual_inv['non-res']
correl_2['ME']=annual_inv['M&E']
correl_2['IP']=annual_inv['IP']

correl_2['TI+1']=annual_inv['total inv'].shift(1)
correl_2['BI+1']=annual_inv['total business inv'].shift(1)
correl_2['NR+1']=annual_inv['non-res'].shift(1)
correl_2['ME+1']=annual_inv['M&E'].shift(1)
correl_2['IP+1']=annual_inv['IP'].shift(1)

correl_2['TI+2']=annual_inv['total inv'].shift(2)
correl_2['BI+2']=annual_inv['total business inv'].shift(2)
correl_2['NR+2']=annual_inv['non-res'].shift(2)
correl_2['ME+2']=annual_inv['M&E'].shift(2)
correl_2['IP+2']=annual_inv['IP'].shift(2)

correl_2['TI+3']=annual_inv['total inv'].shift(3)
correl_2['BI+3']=annual_inv['total business inv'].shift(3)
correl_2['NR+3']=annual_inv['non-res'].shift(3)
correl_2['ME+3']=annual_inv['M&E'].shift(3)
correl_2['IP+3']=annual_inv['IP'].shift(3)

plt.plot(correl_2['share'])

"Check correlations"
correlation=correl_2.corr(method='pearson')

"Highest uncertainty periods"

Uncertainty=correl_2.sort_values(by=['share'])
print(Uncertainty)

"Results show peak correlations with....."

peak=correlation.sort_values(by=['share'])
peak=peak.drop(['TI','TI+1','TI+2','TI+3','BI','BI+1','BI+2','BI+3','NR','NR+1','NR+2','NR+3','ME','ME+1','ME+2','ME+3','IP','IP+1','IP+2','IP+3'], axis=1)

"Step 4: Predict business investment with uncertainty"

series_1= pd.DataFrame()
series_1['uncertainty']=correl_2['share']
series_1['TI'] = correl_2['TI']
series_1['BI'] = correl_2['BI']
series_1['ME'] = correl_2['ME']
series_1['IP'] = correl_2['IP']
series_1['NR'] = correl_2['NR']

"Check for stationarity"

from statsmodels.tsa.stattools import adfuller
from numpy import log

X = correl_2.TI
X = log(X)
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

B = correl_2.BI
B = log(X)
result = adfuller(B)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

M = correl_2.ME
M = log(M)
result = adfuller(M)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

U = correl_2.share
U = log(U)
result = adfuller(U)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

"All 3 investment variables are non-stationary, must be differenced. Uncertainty share is stationary."  

Y = series_1.uncertainty
Y = log(Y)
result = adfuller(Y)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
 

"New series with differenced investment data"
diff = X.diff()
diff.plot()
series_1['TI'] = diff
series_1['uncertainty'] = log(series_1['uncertainty'])
series_1['uncertainty'] = series_1['uncertainty'].diff()
series_1['BI'] = log(correl_2['BI'])
series_1['BI'] = series_1['BI'].diff()
series_1['ME'] = log(correl_2['ME'])
series_1['ME'] = series_1['ME'].diff()
series_1['IP'] = log(series_1['IP'])
series_1['IP'] = series_1['IP'].diff()
series_1['NR'] = log(series_1['NR'])
series_1['NR'] = series_1['IP'].diff()
series_1 = series_1[series_1.index>1964]



"Regression model - X changes in uncertainty lead to Y changes in investment"


import statsmodels.api as sm

"Include lagged uncertainty to estimate effect on total investment"

series_1['U-1'] = series_1['uncertainty'].shift(-1)
series_1['U-2'] = series_1['uncertainty'].shift(-2)
series_1['U-3'] = series_1['uncertainty'].shift(-3)
series_1['TI-1'] = series_1['TI'].shift(-1)
series_1['BI-1'] = series_1['BI'].shift(-1)
series_1['ME-1'] = series_1['ME'].shift(-1)
series_1['IP-1'] = series_1['IP'].shift(-1)
series_1['NR-1'] = series_1['NR'].shift(-1)

reg1 = sm.OLS(endog=series_1['TI'], exog=series_1[['uncertainty','U-1','U-2','U-3','TI-1']], missing='drop')
type(reg1)

results = reg1.fit()
type(results)
print(results.summary())

reg2 = sm.OLS(endog=series_1['BI'], exog=series_1[['uncertainty','U-1','U-2','U-3','BI-1']], missing='drop')
type(reg1)

results_2 = reg2.fit()
type(results_2)
print(results_2.summary())

reg3 = sm.OLS(endog=series_1['ME'], exog=series_1[['uncertainty','U-1','U-2','U-3','ME-1']], missing='drop')
type(reg1)

results_3 = reg3.fit()
type(results_3)
print(results_3.summary())

reg4 = sm.OLS(endog=series_1['IP'], exog=series_1[['uncertainty','U-1','U-2','U-3','IP-1']], missing='drop')
type(reg4)

results_4 = reg4.fit()
type(results_4)
print(results_4.summary())

reg5= sm.OLS(endog=series_1['NR'], exog=series_1[['uncertainty','U-1','U-2','U-3','NR-1']], missing='drop')
type(reg5)

results_5 = reg5.fit()
type(results_5)
print(results_5.summary())


from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt

forecast_series=pd.DataFrame(series_1.TI)
forecast_series['X']=series_1.uncertainty
y = forecast_series.TI
X_train, X_test, y_train, y_test = train_test_split(forecast_series.X, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
X_train = X_train.reshape((40,1))
Y_train = y_train.reshape(40,1)
X_test = X_test.reshape((11,1))
y_test = y_test.reshape((11,1))
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


error = pd.DataFrame(y_test)
error['observed'] = pd.DataFrame(y_test).astype(float)
error['prediction'] = predictions.astype(float)
error['RMSE_TI'] = error.prediction - error.observed
error['RMSE_TI'] = error.RMSE_TI**2
error['RMSE_TI'] = sum(error.RMSE_TI)/11


forecast_series['X']=series_1.uncertainty
y = series_1.BI
X_train, X_test, y_train, y_test = train_test_split(forecast_series.X, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
X_train = X_train.reshape((40,1))
Y_train = y_train.reshape(40,1)
X_test = X_test.reshape((11,1))
y_test = y_test.reshape((11,1))
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


error['observed'] = pd.DataFrame(y_test).astype(float)
error['prediction'] = predictions.astype(float)
error['RMSE_BI'] = error.prediction - error.observed
error['RMSE_BI'] = error.RMSE_BI**2
error['RMSE_BI'] = sum(error.RMSE_BI)/11

forecast_series['X']=series_1.uncertainty
y = series_1.IP
X_train, X_test, y_train, y_test = train_test_split(forecast_series.X, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
X_train = X_train.reshape((40,1))
Y_train = y_train.reshape(40,1)
X_test = X_test.reshape((11,1))
y_test = y_test.reshape((11,1))
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


error['observed'] = pd.DataFrame(y_test).astype(float)
error['prediction'] = predictions.astype(float)
error['RMSE_IP'] = error.prediction - error.observed
error['RMSE_IP'] = error.RMSE_IP**2
error['RMSE_IP'] = sum(error.RMSE_IP)/11

forecast_series['X']=series_1.uncertainty
y = series_1.ME
X_train, X_test, y_train, y_test = train_test_split(forecast_series.X, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
X_train = X_train.reshape((40,1))
Y_train = y_train.reshape(40,1)
X_test = X_test.reshape((11,1))
y_test = y_test.reshape((11,1))
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


error['observed'] = pd.DataFrame(y_test).astype(float)
error['prediction'] = predictions.astype(float)
error['RMSE_ME'] = error.prediction - error.observed
error['RMSE_ME'] = error.RMSE_ME**2
error['RMSE_ME'] = sum(error.RMSE_ME)/11


