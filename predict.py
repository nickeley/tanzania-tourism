import sys
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import  calculate_metrics, remove_outliers, log_transform_target, inverse_log_transform_target, impute_package_columns, drop_package_columns, create_preprocessor

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv)) 

#in an ideal world this would validated
model = sys.argv[1]
X_test_path = sys.argv[2]
y_test_path = sys.argv[3]

# load the model from disk
loaded_model = pickle.load(open(model, 'rb'))
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

#feature eng on test data
print("Feature engineering")
#X_test = impute_package_columns(X_test)
X_train = drop_package_columns(X_test)
y_test_log = log_transform_target(y_test)

y_test_pred_log = loaded_model.predict(X_test)
y_test_pred = inverse_log_transform_target(y_test_pred_log)

calculate_metrics(y_test, y_test_pred, True, False)
