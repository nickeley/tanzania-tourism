import pandas as pd
import numpy as np

from scipy.stats import zscore

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# currency rate for Tanzania-Schilling
TZS_RATE = 2628.57
    

#'Unnamed: 0' and Quakers
def remove_outliers(X_train:pd.DataFrame, y_train:pd.DataFrame) -> pd.DataFrame:
    print("Remove outliers from train data...")
    # Find outliers with z-score
    df_zscore_total_female  = zscore(X_train.total_female.fillna(X_train.total_female.median()))
    df_zscore_total_male  = zscore(X_train.total_male.fillna(X_train.total_male.median()))
    df_zscore_night_mainland  = zscore(X_train.night_mainland.fillna(X_train.night_mainland.median()))
    df_zscore_night_zanzibar  = zscore(X_train.night_zanzibar.fillna(X_train.night_zanzibar.median()))
    #df_zscore_total_cost  = zscore(tourism_df.total_cost)

    df_scores = pd.concat([df_zscore_total_female, df_zscore_total_male, df_zscore_night_mainland, df_zscore_night_zanzibar], axis=1)

    # Calculate data loss based on 4 standard deviations of these 4 numeric features
    loss_percentage = df_scores.query('total_female > 4 or total_male > 4 or night_mainland > 4 or night_zanzibar > 4').count()['total_female'] / X_train.shape[0] * 100
    loss_rows = df_scores.query('total_female > 4 or total_male > 4 or night_mainland > 4 or night_zanzibar > 4')['total_female']

    print(f'Percentage of data loss of the training data based on 4 standard deviations for the numeric features: {loss_percentage.round(2)}%')
    print(f'Number of rows to be removed: {loss_rows.count().round(3)}')

    X_train.drop(loss_rows.index, axis=0, inplace=True)
    y_train.drop(loss_rows.index, axis=0, inplace=True)
    return X_train, y_train

def create_log_transformer():
    return FunctionTransformer(
        func=np.log1p,
        inverse_func=np.expm1
    )

def log_transform_target(y:pd.Series) -> pd.Series:
    print("Log-transform y")
    log_transformer = create_log_transformer()
    y_log = log_transformer.transform(y)
    return y_log

def inverse_log_transform_target(y_log:pd.Series) -> pd.Series:
    print("Inverse-log-transform y")
    log_transformer = create_log_transformer()
    y = log_transformer.inverse_func(y_log)
    return y

def get_package_columns(X):
    all_columns = list(X.columns)

    # get list of all columns that only concern Package Tours
    package_columns = [col for col in all_columns if 'package' in col]
    return package_columns


def impute_package_columns(X:pd.DataFrame) -> pd.DataFrame:
    print("Impute package-columns for X")
    package_columns = get_package_columns(X)
    X_imputed_package = X.copy()
    # Replace values of package columns with 'Irrelevant' vor Independent travellers
    X_imputed_package.loc[X['tour_arrangement'] == 'Independent', package_columns] = 'Irrelevant'
    return X_imputed_package

def drop_package_columns(X:pd.DataFrame) -> pd.DataFrame:
    print("Drop package-columns for X")
    package_columns = get_package_columns(X)
    X_without_package_cols= X.drop(package_columns, axis=1)
    return X_without_package_cols

def create_preprocessor(X_train:pd.DataFrame, include_package_columns = True):
    if include_package_columns:
        print(f"Create preprocessor")
    else:
        print(f"Create preprocessor without package columns")
    # Create feature lists for different kinds of pipelines
    impute_median_features = ['total_female', 'total_male']      # num_features
    impute_missing_features = ['travel_with']                    # cat_feature
    impute_no_comments_features = ['most_impressing']            # cat_feature

    # ID is a unique identifier for each tourist and therefore not relevant for the model
    drop_features = ['ID']                                      # cat_feature

    num_features = list(X_train.columns[X_train.dtypes!=object])
    # remove items that also need to go through imputation
    num_features = [x for x in num_features if x not in impute_median_features]

    cat_features = list(X_train.columns[X_train.dtypes==object])
    # remove items that also need to go through imputation or need to be dropped
    cat_features = [x for x in cat_features if x not in impute_missing_features and x not in impute_no_comments_features and x not in drop_features]

    # Create preprocessing pipelines
    impute_median_pipeline = Pipeline([
    ('imputer_num', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
    ])

    impute_missing_pipeline = Pipeline([
    ('imputer_cat', SimpleImputer(strategy='constant', fill_value='missing')),
    ('1hot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    impute_no_comments_pipeline = Pipeline([
    ('imputer_cat', SimpleImputer(strategy='constant', fill_value='No comments')),
    ('1hot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
    ('1hot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    if include_package_columns:
        # Create preprocessor with package columns
        preprocessor = ColumnTransformer([
            ('median', impute_median_pipeline, impute_median_features),
            ('missing', impute_missing_pipeline, impute_missing_features),
            ('nocomment', impute_no_comments_pipeline, impute_no_comments_features),
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
            ])
    else:
        # Create new preprocessor with dropped package columns
        cat_features_wopack = [feature for feature in cat_features if feature not in get_package_columns(X_train)]

        preprocessor = ColumnTransformer([
            ('median', impute_median_pipeline, impute_median_features),
            ('missing', impute_missing_pipeline, impute_missing_features),
            ('nocomment', impute_no_comments_pipeline, impute_no_comments_features),
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features_wopack)
            ])
    return preprocessor


def calculate_metrics(y_true, y_pred, print_metrics, calculate_r2=False):
    if calculate_r2:
        # R2 can only be used to assess train data
        r2 = r2_score(y_true, y_pred)
        if print_metrics:
            print(f'R2: {r2}')
        
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    metrics = {'rmse' : rmse,
                'mae' : mae}

    if print_metrics:
        print(f'RMSE in TZS: {rmse}')
        print(f'RMSE in EUR: {rmse / TZS_RATE}')

        print(f'MAE in TZS: {mae}')
        print(f'MAE in EUR: {mae / TZS_RATE}')
    return metrics