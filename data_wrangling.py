import pandas as pd
from sklearn.preprocessing import LabelEncoder


def open_df(path):
    """
    Function to open dataset and select necessary features based on analysis done in `model_training_and_analysis.ipynb`

    :param path: String containing path of dataset's file
    :return: DataFrame containing the dataset
    """

    df = pd.read_csv(path, header=None)
    # Remove features which has low correlation (based on previous analysis in)
    df.drop(columns=[0, 6, 11, 12, 13], axis=1, inplace=True)

    return df


def missing_data_handling(df):
    """
    Function to remove data which has value '?' in the row.
    Value '?' indicates that the feature value is NaN

    :param df: DataFrame containing the dataset
    :return: DataFrame containing the dataset after handling the missing data
    """

    # Remove row which at least one features has value '?'
    df = df[~(df == '?').any(axis=1)]
    df.reset_index(drop=True, inplace=True)

    return df


def incosistent_format_handling(df):
    """
    Function to change data type for feature 1 (Age) based on analysis done in `model_training_and_analysis.ipynb`

    :param df: DataFrame containing the dataset
    :return: DataFrame containing the dataset after handling the incosistent format
    """

    # Change feature 1 (Age) data type from object to float64
    df[1] = df[1].astype('float64')

    return df


def get_dict_unique_val(df_ori, df_copy):
    """
    Function to get unique value before and after label encoding

    :param df_ori: DataFrame containing original data value before feature encoding
    :param df_copy: DataFrame containing data value after feature encoding
    :return ori_unique_val: List containing unique value before feature encoding
    :return encode_unique_val: List containing unique value after feature encoding
    """

    ori_unique_val = {}
    encode_unique_val = {}

    for col in list(df_ori.columns):
        # Listing unique value on every feature except the label (Feature 15 (approval))
        if col != 15:
            ori_unique_val[col] = list(df_ori[col].unique())
            encode_unique_val[col] = list(df_copy[col].unique())

    return ori_unique_val, encode_unique_val


def data_encoding(df):
    """
    Function to do categorical features encoding (manually or automatically using LabelEncoder())

    :param df: DataFrame containing the dataset
    :return df_copy: DataFrame after feature encoding
    :return ori_unique_val: List containing unique value before feature encoding
    :return encode_unique_val: List containing unique value after feature encoding
    """

    df_copy = df.copy()

    # Manual encoding for features 8 (PriorDefault) and 9 (Employed)
    for col in [8, 9]:
        df_copy.loc[df_copy[col] == 't', col] = 1
        df_copy.loc[df_copy[col] == 'f', col] = 0

    # Manual encoding for feature 15 (Approved)
    df_copy.loc[df_copy[15] == '+', 15] = 1
    df_copy.loc[df_copy[15] == '-', 15] = 0

    # Change the data type of features 8, 9, 15 as integer
    df_copy[[8, 9, 15]] = df_copy[[8, 9, 15]].astype('int64')

    label_encoder = LabelEncoder()

    # Automatically do feature encoding for the remaining features which has object data type
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col] = label_encoder.fit_transform(df[col])

    # Get unique value both before and after feature encoding
    ori_unique_val, encode_unique_val = get_dict_unique_val(df, df_copy)

    return df_copy, ori_unique_val, encode_unique_val


def outlier_imputation(df, col):
    """
    Function to handle outlier based on the analysis done in `model_training_and_analysis.ipynb`.
    The outlier will be represented with upper extreme or lower extreme to maintain the meaning of the value.

    :param df: DataFrame containing the dataset
    :param col: Integer representing the feature number
    :return: DataFrame from feature which the outliers have been handled
    """

    # Get the first quartile (Q1) and third quartile (Q3)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    # Get the lower and upper extreme
    lower_extreme = q1 - (1.5 * (q3 - q1))
    upper_extreme = q3 + (1.5 * (q3 - q1))

    # If the outlier is more than the upper extreme, than set the outlier value as the upper extreme
    # Else if the outlier is less than the lower extreme, than set the outlier value as the lower extreme
    out_upper = df[(df[col] > upper_extreme)].values
    out_lower = df[(df[col] < lower_extreme)].values

    df[col].replace(out_upper, upper_extreme, inplace=True)
    df[col].replace(out_lower, lower_extreme, inplace=True)

    return df[col]
