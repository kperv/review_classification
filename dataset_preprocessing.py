"""This module consists of methods to preprocess the raw dataframe
to be ready for machine learning tasks.

Methods:
    clean_descriptions: clean strings from numbers and punctuation,
        apply lowercase and tokenize, if attribute not set to False;
    read_data: find the correct csv file in the given folder and deeper,
        read the file and return it with two relevant columns;
    get_nclass_df: pull n classes out of the original dataframe
        to a new dataframe;
    add_splits: calculate the number of rows to be used
        for training and validation according to attributes and create
        a new dataframe with additional column, holding a string
        with a split name.
    get_chunk_for_pretty_print: create a dataframe with n texts
        taken from the end of the original dataframe, return a tuple
        with the original dataframe without without these n texts
        and the new one.
    reviews_for_pretty_print: process a left out chunk of texts
        and return a tuple with three lists: texts, labels
        and original texts.
 """

__all__ = ["clean_descriptions", "read_data", "get_nclass_df", "add_splits",
           "reviews_for_pretty_print", "get_chunk_for_pretty_print"]


import pandas as pd
import os
import math


def read_data(dataset_path=os.getcwd()):
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
            if file.endswith('.csv'):
                data_path = os.path.join(root, file)
    assert os.path.isfile(data_path), "Data file is absent."
    dataframe = pd.read_csv(data_path)
    return dataframe[['country', 'description']]


def clean_descriptions(data_df, split_string=True):
    """Clean all description strings from numbers
     and characters different from alphabet letters,
     turn all words to lowercase to exclude duplicates
     and split string into tokens if attribute split_string.
     """
    data_df.description = data_df.description.str.lower()
    data_df.description = data_df.description.str.replace(
        '[^a-zA-Z\']+', ' ',
        regex=True
    )
    if split_string:
        data_df.description = data_df.description.str.split()
    return data_df


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
    assert (
            test_ratio > 0.1
            or math.isclose(test_ratio, 0.1, rel_tol=0.01, abs_tol=0.01)
    ), wrong_ratio_msg
    split_reviews = pd.DataFrame(columns=[
        'country',
        'description',
        'split'
        ]
    )
    for country in data_df.country.unique():
        country_reviews = pd.DataFrame(data_df[data_df.country == country])
        country_num_rows = _calc_num_rows(
            country_reviews,
            train_ratio,
            val_ratio
        )
        country_reviews = _write_to_split_column(
            country_reviews,
            *country_num_rows
        )
        split_reviews = split_reviews.append(country_reviews)
    split_reviews = split_reviews.sample(frac=1)
    return split_reviews


def get_chunk_for_pretty_print(data_df, n=5):
    """Take away n texs from the tail of original dataframe
     and save them in a new df.

    Args:
        data_df: DataFrame with reviews
        n: a number of reviews to be explicitly printed and predicted
    Returns:
        a tuple with end-cropped original DataFrame
        and a new DataFrame with n unprocessed reviews
    """
    return data_df[:-n], data_df[-n:]


def reviews_for_pretty_print(data_df):
    """Prepare a small test set for detailed printing.

    Args:
        data_df: a list of unprocessed original strings
    Returns:
        a tuple with three elements: text is a list
        of tokenized data, label is a list of classes
        and unprocessed data is a list of unprocessed string reviews
    """
    unprocessed_data = data_df.description.values.tolist()
    data_df.description = clean_descriptions(data_df)
    text = data_df.description.values.tolist()
    label = data_df.country.values.tolist()
    return text, label, unprocessed_data

