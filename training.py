from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_wrangling import *

DATASET_PATH = 'datasets/crx.data'


def feature_engineering():
    """
    Function to do feature engineering for the dataset including:
    - Data cleaning
    - Split data into train and test set
    - Data scaling (using MinMaxScaler)
    All consideration for doing the feature engineering has been analysed in `model_training_and_analysis.ipynb`

    :return x_train: DataFrame containing the features (X) for training set
    :return x_test: DataFrame containing the features (X) for test set
    :return y_train: DataFrame containing the label (Y) for training set
    :return y_test: DataFrame containing the label (Y) for test set
    :return ori_unique_val: List containing feature original unique value before feature encoding
    :return encode_unique_val: List containing feature unique value after feature encoding
    """

    # Read the dataset by calling function `open_df()`
    df = open_df(DATASET_PATH)
    # Handling missing data by calling function `missing_data_handling()`
    df = missing_data_handling(df)
    # Handling feature incosistent format by calling function `incosistent_format_handling()`
    df = incosistent_format_handling(df)

    # Get encoded dataset also both unique val before and after feature encoding by calling function `data_encoding()`
    df, ori_unique_val, encode_unique_val = data_encoding(df)

    # Handle outlier on specific features
    # The feature has been analysed in `model_training_and_analysis.ipynb`
    col_outlier = [1, 2, 7, 10, 14]
    df[col_outlier] = df[col_outlier].astype('float64')

    # Calling function `outlier_imputation()` to handle the outliers on specific features
    for col in col_outlier:
        df[col] = outlier_imputation(df, col)

    # Set `x` as feature and `y` as label for training
    # `x` consists of all feature except label (Feature 15 (Approval))
    # `y` consists of label (Feature 15 (Approval))
    x = df.drop(columns=[15])
    y = df[15]

    # Split the dataset into train and test set with ratio train:test = 80:20
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

    # Perform data scaling using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, ori_unique_val, encode_unique_val


def model_training(x_train, y_train):
    """
    Function to do model training using Random Forest model
    Consideration in choosing and tuning the model has been analysed in `model_training_and_analysis.ipynb`

    :param x_train: DataFrame containing the features (X) for training set
    :param y_train: DataFrame containing the label (Y) for training set
    :return model: Random Forest model which has been built with determined parameter
    """

    # Best parameter for Random Forest on this dataset obtained from analysis in `model_training_and_analysis.ipynb`
    params = {
        'criterion': 'gini',
        'max_depth': 30,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 6,
        'n_estimators': 50
    }

    # Initialized and built the Random Forest model
    model = RandomForestClassifier(
        criterion=params['criterion'],
        max_depth=params['max_depth'],
        max_features=params['max_features'],
        min_samples_leaf=params['min_samples_leaf'],
        min_samples_split=params['min_samples_split'],
        n_estimators=params['n_estimators'])
    model.fit(x_train, y_train)

    return model


def model_evaluation(model, x_test, y_test):
    """
    Function to do evaluation regarding the model on test set

    :param model: Random Forest model which has been built with determined parameter
    :param x_test: DataFrame containing the features (X) for test set
    :param y_test: DataFrame containing the label (Y) for test set
    :return acc: Accuracy score regarding the model on test set
    :return precision: Precision score regarding the model on test set
    :return recall: Recall score regarding the model on test set
    :return f1: F1-score regarding the model on test set
    """

    # Do prediction using the model with test set
    y_pred = model.predict(x_test)

    # Obtain the accuracy, precision, recall, and F1 score
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return acc, precision, recall, f1
