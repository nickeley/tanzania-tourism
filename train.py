import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from feature_engineering import  calculate_metrics, remove_outliers, log_transform_target, inverse_log_transform_target, impute_package_columns, drop_package_columns, create_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

# import raw data
raw_data_df = pd.read_csv('data/original_zindi_data/Train.csv')

#cleaning data and preparing
X = raw_data_df.drop("total_cost", axis=1)
y = raw_data_df["total_cost"]


# splittin into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Feature engineering on train data")
# remove outliers only for train data
X_train, y_train = remove_outliers(X_train, y_train)

# optional feature engineering for package columns
#X_train = impute_package_columns(X_train)
X_train = drop_package_columns(X_train)
# log-transform target columns
y_train_log = log_transform_target(y_train)


print("Feature engineering on test data")
# optional feature engineering for package columns
#X_test = impute_package_columns(X_test)
X_test = drop_package_columns(X_test)

# randomized Grid Search for AdaBoostRegressor
param_grid = {
            'regressor__n_estimators': [10, 30, 50, 70, 90],
            'regressor__learning_rate': [0.3, 0.5, 0.7, 1.0],
            'regressor__loss': ['linear', 'square', 'exponential'],
            'regressor__base_estimator__max_depth': np.arange(5,70,5),
            'regressor__base_estimator__min_samples_split': np.arange(5,30, 5),
            'regressor__base_estimator__splitter': ['best', 'random']
            }

dtreg = DecisionTreeRegressor(criterion='squared_error')

# create preprocessor for pipeline without package columns
preprocessor = create_preprocessor(X_train, include_package_columns=False)

# setup pipeline
pipe_regressor = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', AdaBoostRegressor(base_estimator= dtreg, random_state=42))
    ])

print("Perform randomized search")
# perform randomized search
rand_search = RandomizedSearchCV(pipe_regressor, param_distributions=param_grid, scoring='neg_root_mean_squared_error', cv=5, verbose=1, n_jobs=-1, n_iter=100)


print("Training an optimized AdaBoost Model with Decision Tree")
# Train Model with Adaboost, without the package columns and with log-transformed target column
rand_search.fit(X_train, y_train_log) 

# save best model
best_model_ada = rand_search.best_estimator_

# print best parameters
print(rand_search.best_params_)

print("Predict on test data")
y_test_pred_log = best_model_ada.predict(X_test)
y_test_pred = inverse_log_transform_target(y_test_pred_log)

print("Predict on train data")
y_train_pred_log = best_model_ada.predict(X_train)
y_train_pred = inverse_log_transform_target(y_train_pred_log)

# calculate and print metrics
print("Calcutate metrics for test data")
calculate_metrics(y_test, y_test_pred, print_metrics=True)
print("Calcutate metrics for train data")
calculate_metrics(y_train, y_train_pred, print_metrics=True, calculate_r2=True)

#saving the model
print("Saving model in the model folder")
filename = 'models/adaboost_model.sav'
pickle.dump(best_model_ada, open(filename, 'wb'))