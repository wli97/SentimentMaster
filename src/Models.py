import math
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import Normalizer

from Transformers import AverageWordLengthExtractor

# BASE MODEL (random guess for each)
def base_model(X_train, y_train, X_test):
    return np.random.randint(2, size=len(X_test))

# PIPELINE MODEL (returns prediction of model defined by pipeline
def pipe_model(X_train, y_train, X_test, pipeline):

    pipeline.fit(X_train, y=y_train)
    y_pred = pipeline.predict(X_test)

    return y_pred

# BERNOUILLI NAIVE BAYES
class BNB():

    def __init__(self, count_vect):
        self.sum = None
        self.probs = None
        self.feat = count_vect.get_feature_names()
        self.vect = count_vect

    def train(self, x, y):

        N = float(len(y))
        self.sum = {0: N - sum(y), 1: 0. + sum(y)}

        # Initialize the count for both good/bad reviews with 1 for Laplace numerator
        self.probs = {sign: {f: 1. for f in self.feat} for sign in self.sum}

        # Step through each document and increment by 1 if feature found in document i
        for review, sign in zip(x, y):
            review = self.to_set(review)
            for word in self.feat:
                if word in review:
                    self.probs[sign][word] += 1.0

        # Compute probs for each feature in feature set with Laplace smoothing
        for sign in self.probs:
            self.probs[sign] = {k: v / (self.sum[sign] + 2) for k, v in self.probs[sign].items()}

    def predict(self, review):
        # Compute the sum of log probability for the given review
        words = self.to_set(review)
        log_sum0 = math.log(self.sum[0] / (self.sum[0] + self.sum[1]))
        log_sum1 = math.log(self.sum[1] / (self.sum[0] + self.sum[1]))
        for f in self.feat:
            prob1 = self.probs[1][f]
            prob0 = self.probs[0][f]
            if f not in words:
                prob1 = 1 - prob1
                prob0 = 1 - prob0
            log_sum1 += math.log(prob1)
            log_sum0 += math.log(prob0)
        # Compare if the likelihood is higher for y=0 or y=1 given x
        prediction = 1
        if log_sum1 < log_sum0: prediction = 0
        return prediction

    def to_set(self, review):
        return set([word.lower() for word in review.split()])

def predict_BNB(X_train, y_train, X_test):
    count_vect = CountVectorizer(max_df=0.5, min_df=5).fit(X_train)
    bnb = BNB(count_vect)
    bnb.train(X_train, y_train)
    return [bnb.predict(x) for x in X_test]


# MULTINOMIAL NAIVE BAYES
def MNB(X_train, y_train, X_test, tdidf=True):
    # This function allows two experiments on MNB model: if tdidf is true, we use tdidf as processor; else bag of words (binary).
    if tdidf:
        pclf = Pipeline([
            ('vects', FeatureUnion([
                ('vect', TfidfVectorizer(encoding='utf-8', strip_accents='unicode', max_df=0.5, min_df=2)),  # can pass in either a pipeline
                ('ave', AverageWordLengthExtractor())  # or a transformer
            ])),
            ('norm', Normalizer()),
            ('clf', MultinomialNB())
        ])
    else:
        vectorizer = CountVectorizer(encoding='utf-8',strip_accents='unicode',max_df=0.5, min_df=2, binary=True)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        pclf = MultinomialNB()

    pclf.fit(X_train, y_train)
    y_pred = pclf.predict(X_test)
    return (y_pred)


# SVM - Best Model
def SVM(X_train, y_train, X_test):
  stop_words=['in','of','at','a','the','this','we','i','/div','/br']
  custom_feature_pipeline = Pipeline([
    ('lem', CountVectorizer(tokenizer=LemmaTokenizer(),
                                strip_accents = 'unicode', stop_words=stop_words, ngram_range=(1, 2))), 
    ('tdif', TfidfTransformer()),
    ('norm', Normalizer()),
    ('clf-svm', SGDClassifier(random_state=69, max_iter=1000, tol=1e-4)),
  ])

  custom_feature_pipeline.fit(X_train, y = y_train)
  y_pred = custom_feature_pipeline.predict(X_test)
  return y_pred
