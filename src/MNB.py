from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def MNB(X_train, y_train, X_test, y_test):
    # Has 0.86 accuracy, which is higher than 0.83 BNB on validation set.
    pclf = Pipeline([
        ('vect', TfidfVectorizer(encoding='utf-8',strip_accents='unicode',analyzer='word',max_df=0.5, min_df=2)),
        ('norm', Normalizer()),
        ('clf', MultinomialNB()),
    ])

    pclf.fit(X_train, y_train)
    y_pred = pclf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    return pclf
