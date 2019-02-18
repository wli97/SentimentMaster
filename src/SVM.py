from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics

def SVM(X_train, X_test, y_train, y_test):
  custom_feature_pipeline = Pipeline([
      ('vect', TfidfVectorizer(encoding='utf-8',strip_accents='unicode',max_df=0.5, min_df=2)),
      ('norm', Normalizer()),
      ('svc', SVC(kernel='linear', gamma='scale'))
  ])

  custom_feature_pipeline.fit(X_train, y = y_train)
  y_pred = custom_feature_pipeline.predict(X_test)
  print(metrics.classification_report(y_pred, y_test))
  return custom_feature_pipeline
