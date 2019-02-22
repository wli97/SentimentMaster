from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, name):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in name.split()])

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        return [ [self.average_word_length(x)] for x in X]

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class AverageWordLengthExtractor2(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, name):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in name.split()])

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        return [ [self.average_word_length(x)] for x in X]

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self