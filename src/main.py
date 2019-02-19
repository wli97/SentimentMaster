from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import os
import codecs
import BNB as BNB

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

result = BernoulliNB(X, y)
print(result)
