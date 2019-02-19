from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import os
import codecs
import BNB as BNB
import MNB
import SVM
import numpy as np
import random


# Creates single .txt file with all review text
def read_data(directory, rating):

    X = []

    for filename in os.listdir(directory):
        with open(directory + filename, encoding='utf-8', mode='r') as f:
            data = f.read().lower()
            X.append(data)
    f = codecs.open(str(rating) + '.txt', 'w', 'utf-8')

    for point in X:
        point.replace('\n', '')
        f.write(point + '\n')
    f.close()

# Reads data from .txt file to create X, y vectors
def read_data_txt(filename, rating):
    X = []

    f = codecs.open(filename, 'r', 'utf-8')
    reviews = f.read().split('\n')[:-1] # Remove last element since just a blank

    for review in reviews:
        X.append(review)

    y = [rating] * len(X)
    f.close()
    return (X, y)

def read_test(directory):
    X=[]
    for i in range(25000):
        with open(directory + str(i) + '.txt', encoding='utf-8', mode='r') as f:
            data = f.read().lower()
            X.append(data)
    f = codecs.open('submit' + '.txt', 'w', 'utf-8')
    for point in X:
        point.replace('\n', '')
        f.write(point + '\n')
    f.close()

# Performs k-fold cross validation using the model provided by predict_fct and returns the average accuracy over all folds
def validate_model(folds, X, y, predict_fct):

    """
    folds: Must evenly divide the number of sample points
    X: The X values for the whole data set
    y: The y values for the whole data set
    predict_fct: Model to use, must be a function that takes X_train, y_train, X_test, y_test (in that order) and returns the prediction on the test set
    """

    # Shuffle with seed to get more even distribution but still reproducible
    random.Random(2).shuffle(X)
    random.Random(2).shuffle(y)

    # Number of folds must cleanly divide the number of sample points
    num_per_fold = int(len(X)/folds)

    accuracy = 0

    # Repeat experiment for each set being left out
    for i in range(folds):

        X_train, y_train = [], []
        X_test, y_test = [], []

        # Create train/test split
        for j in range(folds):
            if i == j:
                X_test += X[j * num_per_fold:(j + 1) * num_per_fold]
                y_test += y[j * num_per_fold:(j + 1) * num_per_fold]
            else:
                X_train += X[j * num_per_fold:(j + 1) * num_per_fold]
                y_train += y[j * num_per_fold:(j + 1) * num_per_fold]

        # Get the prediction and calculate the amount of mistakes
        y_pred = predict_fct(X_train, y_train, X_test, y_test)
        mistakes = np.linalg.norm(y_pred-y_test, ord=1)

        accuracy += (len(X)/folds-mistakes)/(len(X)/folds)

    return accuracy/folds

# Performs custom BNB on training data
def BernoulliNB(x, y):
  count_vect = CountVectorizer(max_df=0.5,min_df=5).fit(X_train)
  bnb = BNB.BNB(count_vect)
  bnb.train(X_train, y_train)
  return bnb


# Read training data (only run commented lines once)
# read_data('train/pos/', 1)
# read_data('train/neg/', 0)
positive = read_data_txt('1.txt', 1)
negative = read_data_txt('0.txt', 0)
X = positive[0] + negative[0]
y = positive[1] + negative[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# Validate MNB and SVM
print(validate_model(5, X, y, MNB.MNB))
print(validate_model(5, X, y, SVM.SVM))