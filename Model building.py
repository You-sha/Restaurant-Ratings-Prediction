# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:33:44 2023

@author: Yousha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dython import nominal


df = pd.read_csv('Zomato final 3.csv')
df.dtypes
df.columns

for i in df.columns:
    print(df[i].value_counts())

plt.figure(figsize=(50,50))
correlations = pd.DataFrame(df.corr()["Aggregate rating"].sort_values(ascending=False))
sns.heatmap(correlations,annot=True,square=True,cbar_kws={'shrink':0.96})
plt.show()

nominal.associations(df,display_columns='Aggregate rating',figsize=(50,50))
plt.savefig('Correlation.png',dpi=600)
plt.show()

X = pd.get_dummies(df.drop(['Aggregate rating','Rating color','Restaurant Name','Locality',
                            'Locality Verbose','Cuisines','Rating text','cuisine_list',
                            'Address','City'],axis=1))
y = df['Aggregate rating']

nominal.associations(X,figsize=(50,50))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

## MODELS ##

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 2)

rf.fit(X_train,y_train)
pred = rf.predict(X_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, pred)
rmse = metrics.mean_squared_error(y_test,pred,squared=False)
medae = metrics.median_absolute_error(y_test,pred)
mae = metrics.mean_absolute_error(y_test, pred)
mape = metrics.mean_absolute_percentage_error(y_test, pred)


print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Median Absolute Error {medae}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print(f'Variance {np.var(y_test)}')


rf.get_params()

## RANDOMIZED DEARCH CV

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf2 = RandomForestRegressor(random_state=2)

rf_rscv = RandomizedSearchCV(estimator=rf2, param_distributions=random_grid,\
                             n_iter = 100, cv = 3, verbose=2, random_state=2, n_jobs = -1)
rf_rscv.fit(X_train,y_train)
rf_rscv.best_params_

rf_random = RandomForestRegressor(
      n_estimators= 200,
      min_samples_split= 10,
      min_samples_leaf= 1,
      max_features='auto',
      max_depth = 10,
      bootstrap= True,
      random_state=2) # Best rscv params

rf_random.fit(X_train,y_train)
pred = rf_random.predict(X_test)

mse = metrics.mean_squared_error(y_test, pred)
rmse = metrics.mean_squared_error(y_test,pred,squared=False)
mae = metrics.mean_absolute_error(y_test, pred)
medae = metrics.median_absolute_error(y_test, pred)
mape = metrics.mean_absolute_percentage_error(y_test, pred)


print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Median Absolute Error (MEDAE): {medae}")
print(f'Mean Absolute Percentage Error (MAPE): {mape}')
print(np.var(y_test))

residuals = y_test - pred

# plot the residuals
plt.scatter(np.linspace(0,5,1479), residuals,c=residuals,cmap='magma')
plt.colorbar(label="Quality", orientation="vertical")
# plot a horizontal line at y = 0
plt.hlines(y = 0,
xmin = 0, xmax=5,
linestyle='--',colors='black')
# set xlim
plt.xlim((0, 5))
plt.xlabel('Aggregate Rating'); plt.ylabel('Residuals')
plt.show()
print(np.median(residuals))

## GRID SEARCH CV ##

# from sklearn.model_selection import GridSearchCV
# param_grid = {'n_estimators':[200,400,600],
#               'min_samples_split':[8,10,12],
#               'min_samples_leaf':[1,2,3],
#               'max_depth':[10,15,20],
#               'bootstrap':[True]}

# rf3 = RandomForestRegressor(random_state=1)

# from datetime import datetime

# print(datetime.now())
# rf_gscv = GridSearchCV(estimator=rf3, param_grid=param_grid,
#                        cv = 3)
# rf_gscv.fit(X_train,y_train)
# rf_gscv.best_params_
# print(datetime.now())

# rf_grid = RandomForestRegressor(
#     bootstrap = True,
#     max_depth=10,
#     max_features=3,
#     min_samples_leaf=2,
#     min_samples_split=12,
#     n_estimators=600,
#     random_state=2) # Best Gridsearch params

# rf_grid.fit(X_train,y_train)
# pred = rf_grid.predict(X_test)

# mse = metrics.mean_squared_error(y_test, pred)
# rmse = metrics.mean_squared_error(y_test,pred,squared=False)
# mae = metrics.mean_absolute_error(y_test, pred)
# mape = metrics.mean_absolute_percentage_error(y_test, pred)


# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(f"Mean Absolute Error (MAE): {mae}")
# print(f'Mean Absolute Percentage Error (MAPE): {mape}')

