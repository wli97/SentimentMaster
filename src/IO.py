import os
import codecs
import csv

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


def submitModel(model, X_train, y_train, X_test, filename):
    y_sub = model(X_train, y_train, X_test)
    with open(filename, "w",newline="") as f:
        writer = csv.writer(f)
        writer.writerows([[i] for i in y_sub])


def read_lexicon_afinn(filename):
    with open(filename, 'r') as f:
        word_scores = f.readlines()

    words = {}
    for word in word_scores:

        list = word.split('\t')

        words[list[0]] = int(list[1])

    return words

def read_lexicon_ole():
    words = {}

    with open('negative-words.txt', 'r') as f:
        word_scores = f.read().splitlines()
        for word in word_scores:
            words[word] = -1

    with open('positive-words.txt', 'r') as f:
        word_scores = f.read().splitlines()
        for word in word_scores:
            words[word] = 1

    return words

def print_test(name, results):
    print('Testing ' + name + ':')
    print('----------------------------')
    print('Accuracy: ' + str(results[0]))
    print('f1: ' + str(results[1]))
    print()
