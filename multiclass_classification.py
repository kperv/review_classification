"""
Description

"""

import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import dataset_preprocessing as dp
import pretty_print as pp

DATA_DIR = '/Users/ksu/projects/data'
NUM_CLASSES = 3
TRAIN_RATIO = 0.7
VALIDATE_RATIO = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_EPOCHS = 5
NUM_CHANNELS = 64
LEARNING_RATE= 0.005
MIN_TOKEN_COUNT = 25  # minimal count of a token o be put in Vocabulary


class Vocabulary(object):
    """Functionality of vocabulary dictionary.

    Attributes:
        _token_to_idx: a dictionary with tokens as keys
                       and indexes as values
        _idx_to_token: a dictionary with indexes as keys
        _unk_index: a value to be returned
                    if the token is not present in the vocabulary
    Methods:
        get_vocab: return a dictionary of tokens and indexes
        add_token: check the token and if absent, add to the dictionary
        lookup: return the index of token in question or -1
        lookup_index: return the token that corresponds to the index
    """

    def __init__(self, token_to_idx={}):
        """
        Initialize a Vocabulary object from a dictionary.

        :param token_to_idx: a dictionary created and filled by Vectorizer
        :param unk_token: a symbol to represent a missing token
        """
        self._token_to_idx = token_to_idx
        for token, idx in self._token_to_idx.items():
            self._idx_to_token = {idx: token}
        self._unk_index = -1

    def add_token(self, token):
        """Add a token to the dictionaries."""
        if token not in self._token_to_idx:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

    def lookup(self, token):
        """Get index of the token."""
        return self._token_to_idx.get(token, self._unk_index)

    def lookup_index(self, index):
        """Get token that is bound with the index."""
        return self._idx_to_token[index]

    def get_vocab(self):
        """Return the attribute dictionary."""
        return self._token_to_idx

    def __str__(self):
        """Representation view of the class instance"""
        return "<Vocabulary (size={})>".format(len(self))

    def __len__(self):
        """Get the length of the vocabulary"""
        return len(self._token_to_idx)


class ReviewVectorizer(object):
    """Create a matrix representing a review using token-index dictionary."""

    def __init__(self, review_vocab, country_vocab, max_review_length):
        self.review_vocab = review_vocab
        self.country_vocab = country_vocab
        self.max_review_length = max_review_length

    def get_review_vocab(self):
        return self.review_vocab

    def get_country_vocab(self):
        return self.country_vocab

    def get_max_review_length(self):
        return self.max_review_length

    def vectorize(self, review):
        one_hot_matrix_size = (len(self.review_vocab), self.max_review_length)
        one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)
        for index, word in enumerate(review):
            word_index = self.review_vocab.lookup(word)
            one_hot_matrix[word_index][index] = 1
        return one_hot_matrix

    @classmethod
    def from_dataframe(cls, review_df, cutoff):
        word_count = Counter()
        for review in review_df.description:
            word_count.update(review)

        tokens = [
            word for word, count in word_count.items()
            if count > cutoff
            ]
        vocab_token_to_idx = {
            token: index for index, token in enumerate(tokens)
            }
        country_token_to_idx = {
            country: index for index, country
            in enumerate(set(review_df.country))
            }

        review_vocab = Vocabulary(vocab_token_to_idx)
        country_vocab = Vocabulary(country_token_to_idx)

        max_review_length = 0
        for index, row in review_df.iterrows():
            max_review_length = max(max_review_length, len(row.description))

        return cls(review_vocab, country_vocab, max_review_length)


class WineDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split == "train"]
        self.val_df = self.review_df[self.review_df.split == "val"]
        self.test_df = self.review_df[self.review_df.split == "test"]

        self.train_size = len(self.train_df)
        self.val_size = len(self.val_df)
        self.test_size = len(self.test_df)

        self._split_dict = {'train': (self.train_df, self.train_size),
                            'val': (self.val_df, self.val_size),
                            'test': (self.test_df, self.test_size)}
        self.set_split('train')

    def set_split(self, split):
        self._split = split
        self._split_df, self._split_size = self._split_dict[self._split]

    def get_split(self):
        return self._split

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_df, cutoff):
        return cls(
            review_df,
            ReviewVectorizer.from_dataframe(review_df, cutoff=cutoff)
            )

    def get_vectorizer(self):
        return self._vectorizer

    def __len__(self):
        return self._split_size

    def __getitem__(self, index):
        row = self._split_df.iloc[index]
        review_matrix = self._vectorizer.vectorize(row.description)
        country_index = self._vectorizer.country_vocab.lookup(row.country)
        return {'x_review': review_matrix, 'y_country': country_index}


class ReviewClassifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        super(ReviewClassifier, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(
                in_channels=initial_num_channels,
                out_channels=num_channels,
                kernel_size=3
            ),
            nn.ELU())
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x_review, apply_softmax=False):
        features = self.convnet(x_review)
        #features = features.squeeze(dim=2)
        features = torch.sum(features, dim=2)
        prediction_vector = self.fc(features)
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
        return prediction_vector


def generate_batch(batch):
    text = torch.tensor(
        [entry['x_review'] for entry in batch],
        dtype=torch.float32
        )
    label = torch.tensor(
        [entry['y_country'] for entry in batch],
        dtype=torch.long
        )
    return text, label


def set_training_params(wine_dataset_object, num_channels, device, learning_rate):
    vectorizer = wine_dataset_object.get_vectorizer()
    classifier = ReviewClassifier(
        initial_num_channels=len(vectorizer.review_vocab),
        num_classes=len(vectorizer.country_vocab),
        num_channels=num_channels
        )
    classifier = classifier.to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    train_params = classifier, optimizer, loss_func
    predict_params = classifier, vectorizer
    return train_params, predict_params


def train_func(train_df, train_params, batch_size):
    classifier, optimizer, loss_func = *train_params
    train_loss = 0
    train_acc = 0
    data = DataLoader(
        train_df,
        batch_size = batch_size,
        shuffle=True,
        collate_fn=generate_batch
        )
    for i, (text, label) in enumerate(data):
        optimizer.zero_grad()
        text = text.to(DEVICE)
        label = label.to(DEVICE)
        output = classifier.forward(text)
        loss = loss_func(output, label)
        train_loss = loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == label).sum().item()
    return train_loss/len(train_df), train_acc/len(train_df)


def test_func(test_df, train_params):
    classifier, _, loss_func = *train_params
    loss = 0
    acc = 0
    data = DataLoader(
        test_df,
        batch_size=BATCH_SIZE,
        collate_fn=generate_batch
        )
    for text, label in data:
        text = text.to(DEVICE)
        label = label.to(DEVICE)
        with torch.no_grad():
            output = classifier.forward(text)
            loss = loss_func(output, label)
            loss += loss.item()
            acc += (output.argmax(1) == label).sum().item()
    return loss/len(test_df), acc/len(test_df)


def train(data, train_params, batch_size, epochs):
    for epoch in range(epochs):
        start_time = time.time()
        data.set_split("train")
        train_loss, train_acc = train_func(data, train_params, batch_size)
        data.set_split("val")
        val_loss, val_acc = test_func(data, train_params)
        secs = int(time.time() - start_time)
        mins = secs // 60
        secs = secs % 60
        print(('Epoch: {} | time in {} minutes, '
              '{} seconds').format(epoch + 1, mins, secs))
        print(('\t Loss: {:.4f} (train)\t '
               '| \t Acc: {:.1f}%').format(train_loss, train_acc * 100))
        print(('\t Loss: {:.4f} (validate)\t '
               '| \t Acc: {:.1f}%').format(val_loss, val_acc * 100))


def predict_label(data, predict_params):
    classifier, vectorizer = *predict_params
    vectorized_data = vectorizer.vectorize(data)
    vectorized_data = torch.tensor(vectorized_data).unsqueeze(0)
    vectorized_data = vectorized_data.to(DEVICE)
    result = classifier(vectorized_data, apply_softmax=True)
    probability_values, indices = result.max(dim=1)
    index = indices.item()
    predicted_label = vectorizer.country_vocab.lookup_index(index)
    probability_value = probability_values.item()
    return {'country': predicted_label,
            'probability': probability_value}

def pretty_print(data_for_print, predict_params):
    for text, label, unprocessed_text in zip(*data_for_print):
        predict_dict = predict_label(text, predict_params)
        print("Review: ", unprocessed_text)
        print("Country is ", label)
        print(("Prediction for country is {} "
              "with probability {}").format(**predict_dict))

def clean_memory(wine_dataset_object, params):
    del wine_dataset_object
    for param in params:
        del param
    torch.cuda.empty_cache()


def main():
    """Read the dataframe with reviews, preprocess the dataframe,
    instantiate WineDataset instance and set convolutional neural net params,
    train the network and print predictions on saved untokenized reviews.
    """

    reviews = dp.read_data(DATA_DIR)
    print(reviews.head())
    reviews, chunk = pp.get_chunk_for_pretty_print(reviews)
    reviews = dp.get_nclass_df(reviews, NUM_CLASSES)
    reviews = dp.add_splits(reviews)
    wine_dataset = WineDataset.load_dataset_and_make_vectorizer(reviews, MIN_TOKEN_COUNT)
    train_params, predict_params = set_training_params(
        wine_dataset,
        NUM_CHANNELS,
        DEVICE,
        LEARNING_RATE
        )
    train(wine_dataset, train_params, BATCH_SIZE, NUM_EPOCHS)
    reviews_for_print = pp.reviews_for_print(chunk)
    pretty_print(reviews_for_print, predict_params)
    clean_memory(wine_dataset, train_params)


if __name__ == '__main__':
    main()