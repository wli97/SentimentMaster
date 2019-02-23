from sklearn import linear_model, discriminant_analysis, tree, metrics, svm, neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from IO import read_data, read_data_txt, read_test, submitModel, print_test, read_lexicon_ole
from CrossV import validate_model
from Models import base_model, predict_BNB, MNB, SVM, pipe_model, stacking_model
from Transformers import AverageWordLengthExtractor, PositiveNegativeExtractorAFINN, PositiveNegativeExtractorOLE, \
    AverageSentenceLengthExtractor

import matplotlib.pyplot as plt

# Read training data (only run commented lines once)
# read_data('train/pos/', 1)
# read_data('train/neg/', 0)
# read_test('test/')

# Construct X, y vectors for training data
positive = read_data_txt('1.txt', 1)
negative = read_data_txt('0.txt', 0)
X = positive[0] + negative[0]
y = positive[1] + negative[1]


"""" BASIC CLASSIFIERS """
# Test base models (all random and Naive Bayes from scratch)
# print_test('Random guess model', validate_model(5, X, y, base_model))
# print_test('Naive Bayes implemented from scratch', validate_model(5, X, y, predict_BNB))

Logistic_Regression_Basic = Pipeline([
    ('count_vect', CountVectorizer()),
    ('norm', Normalizer()),
    ('log_reg', linear_model.LogisticRegression(solver = 'sag', max_iter = 1000, n_jobs = 4)),
])
# print_test('Logistic Regression (basic)', validate_model(5, X, y, pipe_model(Logistic_Regression_Basic).predict))

Decision_Tree_Basic = Pipeline([
    ('count_vect', CountVectorizer()),
    ('norm', Normalizer()),
    ('dec_tree', tree.DecisionTreeClassifier(min_samples_leaf = 10)),
])
# print_test('Decision Tree (basic)', validate_model(5, X, y, pipe_model(Decision_Tree_Basic).predict))

Linear_SVC_Basic = Pipeline([
    ('count_vect', CountVectorizer()),
    ('norm', Normalizer()),
    ('lin_svc', LinearSVC() ),
])
# print_test('Linear SVC (basic)', validate_model(5, X, y, pipe_model(Linear_SVC_Basic).predict))


""" FEATURE EXPERIMENTS """
Linear_SVC_Binary = Pipeline([
    ('count_vect', CountVectorizer(binary = True)),
    ('norm', Normalizer()),
    ('lin_svc', LinearSVC() ),
])
# print_test('Linear SVC (binary counts)', validate_model(5, X, y, pipe_model(Linear_SVC_Binary).predict))

Linear_SVC_Vocab = Pipeline([
    ('count_vect', CountVectorizer(vocabulary = read_lexicon_ole().items())),
    ('norm', Normalizer()),
    ('lin_svc', LinearSVC() ),
])
# print_test('Linear SVC (pos/neg word lexicon)', validate_model(1, X, y, pipe_model(Linear_SVC_Binary).predict))

Linear_SVC_NGRAM = Pipeline([
    ('count_vect', CountVectorizer(ngram_range = (1,2))),
    ('norm', Normalizer()),
    ('lin_svc', LinearSVC() ),
])
# print_test('Linear SVC (1-2grams)', validate_model(5, X, y, pipe_model(Linear_SVC_NGRAM).predict))

Linear_SVC_NGRAM_Binary = Pipeline([
    ('count_vect', CountVectorizer(ngram_range = (1,2), binary = True)),
    ('norm', Normalizer()),
    ('lin_svc', LinearSVC() ),
])
# print_test('Linear SVC (1-2grams, binary)', validate_model(5, X, y, pipe_model(Linear_SVC_NGRAM_Binary).predict))

Linear_SVC_TFIDF = Pipeline([
    ('count_vect', TfidfVectorizer()),
    ('norm', Normalizer()),
    ('lin_svc', LinearSVC() ),
])
# print_test('Linear SVC (TFIDF)', validate_model(5, X, y, pipe_model(Linear_SVC_TFIDF).predict))

Linear_SVC_TFIDF_NGRAM = Pipeline([
    ('count_vect', TfidfVectorizer(ngram_range = (1,2))),
    ('norm', Normalizer()),
    ('lin_svc', LinearSVC() ),
])
# print_test('Linear SVC (TFIDF, 1-2grams)', validate_model(5, X, y, pipe_model(Linear_SVC_TFIDF_NGRAM).predict))


# """" NUMBER OF FEATURES EXPERIMENT """
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
# classifiers = {
#     'Logistic Regression': linear_model.LogisticRegression(max_iter = 250, tol= 1e-3),
#     'Linear SVC': LinearSVC(),
#     'Multinomial Naive Bayes': MultinomialNB(),
#     'Ridge Regression': linear_model.RidgeClassifier()
# }
#
# results = {classifier: [] for classifier in classifiers}
# for i in range (1, 25):
#
#     TFIDF_NGRAM = Pipeline([
#         ('vect', TfidfVectorizer(ngram_range=(1,3), max_features=i*10000)),
#         ('norm', Normalizer()),
#     ])
#
#     tf_vect = TFIDF_NGRAM.fit(X_train)
#     tf_vect_train = tf_vect.transform(X_train)
#     tf_vect_test = TFIDF_NGRAM.transform(X_test)
#
#     for name, classifier in classifiers.items():
#         clf = classifier.fit(tf_vect_train,y_train)
#         y_pred = clf.predict(tf_vect_test)
#         print(name)
#         print(metrics.accuracy_score(y_test, y_pred))
#         results[name].append(metrics.accuracy_score(y_test, y_pred))
#
# print(results)
results = {'Logistic Regression': [0.8828, 0.8902, 0.8888, 0.8902, 0.888, 0.8884, 0.8884, 0.8892, 0.8898, 0.8888, 0.8894, 0.8894, 0.8888, 0.8888, 0.889, 0.8892, 0.889, 0.8886, 0.8888, 0.8882, 0.8886, 0.8892, 0.8888, 0.8886, 0.8886, 0.888, 0.8876, 0.8868, 0.8866, 0.8868, 0.8872, 0.887, 0.8872, 0.887, 0.887, 0.887, 0.8872, 0.887, 0.887, 0.887, 0.887, 0.887, 0.887, 0.887, 0.887, 0.887, 0.887, 0.8868, 0.887], 'Linear SVC': [0.8846, 0.8892, 0.8916, 0.8944, 0.8958, 0.8958, 0.8952, 0.897, 0.8986, 0.9, 0.8994, 0.8996, 0.8992, 0.8994, 0.8982, 0.899, 0.9, 0.8998, 0.8992, 0.8986, 0.8984, 0.899, 0.8992, 0.8988, 0.8992, 0.8992, 0.8996, 0.8988, 0.8984, 0.899, 0.8988, 0.8986, 0.8992, 0.899, 0.898, 0.8978, 0.898, 0.8976, 0.8976, 0.8978, 0.898, 0.898, 0.8982, 0.8976, 0.898, 0.8982, 0.8984, 0.8988, 0.8988], 'Multinomial Naive Bayes': [0.8624, 0.865, 0.8698, 0.8698, 0.8718, 0.874, 0.8734, 0.8748, 0.8754, 0.8768, 0.8768, 0.877, 0.8768, 0.8764, 0.8772, 0.8758, 0.8758, 0.876, 0.876, 0.8762, 0.8756, 0.8758, 0.8752, 0.8756, 0.876, 0.8756, 0.8756, 0.875, 0.8756, 0.8754, 0.8756, 0.8754, 0.8752, 0.8754, 0.8758, 0.8754, 0.876, 0.8764, 0.8764, 0.8764, 0.8762, 0.8766, 0.8762, 0.8762, 0.876, 0.876, 0.876, 0.876, 0.8758], 'Ridge Regression': [0.8842, 0.8886, 0.8916, 0.8948, 0.8954, 0.8968, 0.8968, 0.8988, 0.8988, 0.901, 0.9022, 0.9018, 0.901, 0.9014, 0.9004, 0.9006, 0.901, 0.8996, 0.8998, 0.8994, 0.8992, 0.9004, 0.8994, 0.8998, 0.9, 0.8996, 0.9006, 0.9002, 0.8998, 0.9008, 0.9012, 0.9008, 0.9004, 0.9004, 0.9006, 0.9, 0.9004, 0.9, 0.9002, 0.9, 0.8996, 0.8998, 0.8998, 0.8994, 0.8994, 0.8996, 0.8994, 0.8994, 0.8996]}

# x = [10000 * i for i in range(1,50)]
#
# plt.figure()
# plt.title('Effect of variation of number of TFIDF features on accuracy of 4 models\n(1 and 2-grams)')
# plt.xlabel('Number of features')
# plt.ylabel('Accuracy')
#
# for result in results.values():
#     plt.plot(x, result)
#
# plt.legend(['Logistic Regression', 'Linear SVC', 'Multinomial Naive Bayes', 'Ridge Regression'], loc='lower right')
# plt.show()
#
# results2 = {'Logistic Regression': [0.8832, 0.8864, 0.8888, 0.8906, 0.889, 0.8896, 0.8892, 0.8888, 0.889, 0.8882, 0.8876, 0.8872, 0.8876, 0.888, 0.8876, 0.8884, 0.8882, 0.8884, 0.888, 0.888, 0.8882, 0.8876, 0.8874, 0.8874], 'Linear SVC': [0.884, 0.8894, 0.8924, 0.8924, 0.8942, 0.8944, 0.8936, 0.8946, 0.8948, 0.8954, 0.8956, 0.8982, 0.8976, 0.8968, 0.8972, 0.8976, 0.899, 0.8988, 0.8986, 0.8984, 0.8984, 0.8996, 0.8986, 0.8988], 'Multinomial Naive Bayes': [0.859, 0.8648, 0.869, 0.873, 0.875, 0.8746, 0.8754, 0.877, 0.8784, 0.8774, 0.8794, 0.8796, 0.8804, 0.8792, 0.8784, 0.8798, 0.88, 0.8804, 0.88, 0.8796, 0.8794, 0.8804, 0.8804, 0.8796], 'Ridge Regression': [0.8818, 0.8872, 0.8904, 0.894, 0.893, 0.8934, 0.8956, 0.8972, 0.8976, 0.8972, 0.8974, 0.8974, 0.8982, 0.8988, 0.8996, 0.8996, 0.8994, 0.8986, 0.8982, 0.8976, 0.8986, 0.9, 0.8996, 0.8996]}
#
# x = [10000 * i for i in range(1,25)]
#
# plt.figure()
# plt.title('Effect of variation of number of TFIDF features on accuracy of 4 models\n(1, 2 and 3-grams)')
# plt.xlabel('Number of features')
# plt.ylabel('Accuracy')
#
# for result in results2.values():
#     plt.plot(x, result)
#
# plt.legend(['Logistic Regression', 'Linear SVC', 'Multinomial Naive Bayes', 'Ridge Regression'], loc='lower right')
# plt.show()


""" STACKING EXPERIMENTS """

clf1 = linear_model.RidgeClassifier()
clf2 = LinearSVC()
clf3 = ExtraTreesClassifier(n_estimators= 100, bootstrap = True, min_samples_split = 10 )
clf4 = MultinomialNB()
# clf5 = R
clf6 = KNeighborsClassifier(n_neighbors=100, weights='distance')
clf7 = AdaBoostClassifier(n_estimators=100)




Stack = Pipeline([
    ('count_vect', TfidfVectorizer(ngram_range = (1,2), max_features=120000)),
    ('norm', Normalizer()),
    ('vote_clf', VotingClassifier(estimators=[
        ('clf1', clf1),
        ('clf2', clf2),
        ('clf3', clf3),
        ('clf4', clf4),
        # ('clf6', clf6),
        # ('clf7', clf7)
    ])),
])

print_test('7 Stack (TFIDF, 1-2grams)', validate_model(1, X, y, pipe_model(Stack).predict))


# Submit model
submit = read_data_txt('submit.txt', 0)
submitModel(MNB, X, y, submit[0], 'output1.csv')