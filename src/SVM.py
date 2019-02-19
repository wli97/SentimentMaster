from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics

def SVM(X_train, y_train, X_test, y_test):
  custom_feature_pipeline = Pipeline([
    ('vect', TfidfVectorizer(encoding='utf-8',strip_accents='unicode',stop_words='english')),
    ('norm', Normalizer()),
    ('clf-svm', SGDClassifier(random_state=69, max_iter=1000, tol=1e-3)),
  ])

  custom_feature_pipeline.fit(X_train, y = y_train)
  y_pred = custom_feature_pipeline.predict(X_test)
  print(metrics.classification_report(y_pred, y_test))
  return y_pred
