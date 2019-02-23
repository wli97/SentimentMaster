from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from IO import read_lexicon_afinn, read_lexicon_ole
import random
import itertools

class GetCombinations(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def get_combinations(self, X):
        num_features = range(0, len(X))
        for combo in itertools.combinations(num_features, 2):
            print(combo)


    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        return [ [self.average_word_length(x)] for x in X]

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class PositiveNegativeExtractorAFINN(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        self.word_scores = read_lexicon_afinn('AFINN-111.txt')

    def get_ratio(self, text):
        """Helper code to compute average word length of a name"""
        score = 0
        for word in text.split(' '):
            if word in self.word_scores:
                if self.word_scores[word] < 0:
                    score += self.word_scores[word]
        return score/len(text.split(' '))

    def get_last_ratio(self, text):

        score = 0

        for word in text.split('.')[0].split(' '):
            if word in self.word_scores:
                score += self.word_scores[word]

        # for sentence in text.split('.'):
        #     sentence_score = 0
        #     for word in sentence.split(' '):
        #         if word in self.word_scores:
        #             if self.word_scores[word] < 0:
        #                 sentence_score += 1
        #     score += sentence_score

        return score

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        return [ [random.randint(0, 5)] for x in X]

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class PositiveNegativeExtractorOLE(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        self.word_scores = read_lexicon_ole()

    def get_ratio(self, text):
        """Helper code to compute average word length of a name"""
        score = 0
        for word in text.split(' '):
            if word in self.word_scores:
                score += self.word_scores[word]
        return score

    def get_last_ratio(self, text):

        score = 0

        for word in text.split('.')[len(text.split('.'))-1].split(' '):
            if word in self.word_scores:
                score += self.word_scores[word]

        sentence_scores = []
        for sentence in text.split('.'):
            sentence_score = 0
            for word in sentence.split(' '):
                if word in self.word_scores:
                    sentence_score += self.word_scores[word]
            sentence_scores.append(sentence_score)

        for i in range(len(sentence_scores)):
            sentence_scores[i] =  (i -(len(sentence_scores)/2))**2 * sentence_scores[i]

        return sum(sentence_scores)

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        return [ [self.get_last_ratio(x)] for x in X]

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


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


class AverageSentenceLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_sentence_length(self, text):
        """Helper code to compute average word length of a name"""

        return len(text.split(' '))/len(text.split('.'))

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        return [ [self.average_sentence_length(x)] for x in X]

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self