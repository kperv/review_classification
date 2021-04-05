"""This module consists of methods to preprocess the raw dataframe
to be ready for machine learning tasks.
Methods:
    read_data: find the correct csv file in the given folder and deeper,
        read the file and return it with two relevant columns
    get_nclass_df: pull n classes out of the original dataframe
        to a new dataframe
    add_splits: calculate the number of rows to be used
        for training and validation according to attributes and create
        a new dataframe with additional column, holding a string
        with a split name.
 """

__all__ = ["read_data", "get_nclass_df", "add_splits"]


import pandas as pd
import os


def read_data(dataset_path):
    """Find the file in dataset_path or lower in subdirectories,
    read the file and return a dataframe with two relevant
    columns 'description' and 'country'. The 'description' column holds
    text data - (str) reviews, given to a particular wine
    and 'country' column holds class data - (str) the wine origin country.

    Args:
        dataset_path: absolute path to start search from

    Returns:
        a dataframe with two relevant for the ML task columns
    """
    data_path = ""
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('150k.csv'):
                data_path = os.path.join(root, file)
    assert os.path.isfile(data_path), "Data file is absent."
    dataframe = pd.read_csv(data_path)
    return dataframe[['country', 'description']]

def get_nclass_df(data_df, n_classes):
    """Choose the number of countries equal to n_classes argument value
     out of all according to country counts and create a new dataframe
     with chosen top count countries.

    Args:
        data_df: dataframe with reviews
        :param n_classes: should be 2 (for binary classeification)
            or more (for multiclass classification)
    Returns:
        a new smaller dataframe with n_classes unique countries

    >>> df = pd.DataFrame({})
    >>> new_df = get_nclass_df(df, 3)
    DataFrame is empty
    >>> df = pd.DataFrame({
    ...     'review': ['Good wine', 'Bad wine', 'Woody wine'],
    ...     'country': ['US', 'France', 'Italy']
    ...     }
    ... )
    >>> n_classes = 2
    >>> n_class_df = get_nclass_df(df, n_classes)
    >>> len(n_class_df.country.unique()) == n_classes
    True
    >>> n_class_df = get_nclass_df(df, -1)
    Traceback (most recent call last):
    ...
    AssertionError: Number of classes is invalid
    """
    invalid_class_number_msg = "Number of classes is invalid"
    try:
        original_num_classes = len(data_df.country.unique())
    except AttributeError:
        print("DataFrame is empty. None object is returned.")
        return None
    else:
        assert 0 < n_classes <= original_num_classes, invalid_class_number_msg
        if n_classes == original_num_classes:
            return data_df
        top_countries = data_df.country.value_counts()[:n_classes].keys()
        n_country_df = pd.DataFrame(
            columns=['country', 'description']
        )
        for country in top_countries:
            country_reviews = pd.DataFrame(
                data_df[data_df.country == country]
            )
            n_country_df = n_country_df.append(country_reviews)
        return n_country_df

def _write_to_split_column(df, n_train, n_val):
    """Add corresponding string to the dataframe column."""
    df['split'] = None
    df.split.iloc[:n_train] = 'train'
    df.split.iloc[n_train:n_train + n_val] = 'val'
    df.split.iloc[n_train + n_val:] = 'test'
    return df

def _calc_num_rows(df, train_ratio, val_ratio):
    """Calculate number of rows for each split according to ratio."""
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return n_train, n_val

def add_splits(data_df, train_ratio=0.7, val_ratio=0.15):
    """Create a new dataframe with additional column 'split',
    calculate number of rows for each split
    and populate with a corresponding string the new column.

    Args:
        data_df: a dataframe with reviews
        train_ratio: (float) proportion of dataframe
            to be used for training
        val_ratio: the same for validation
    Returns:
        a sorted dataframe with three columns
    """
    wrong_ratio_msg = "Entered train/val/test ratio is incorrect."
    test_ratio = 1. - (train_ratio + val_ratio)
    assert test_ratio >= 0.1, wrong_ratio_msg
    split_reviews = pd.DataFrame(columns=['country', 'description', 'split'])
    for country in data_df.country.unique():
        country_reviews = pd.DataFrame(data_df[data_df.country == country])
        country_num_rows = _calc_num_rows(country_reviews, train_ratio, val_ratio)
        country_reviews = _write_to_split_column(country_reviews, *country_num_rows)
        split_reviews = split_reviews.append(country_reviews)
    split_reviews = split_reviews.sample(frac=1)
    return split_reviews

