from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def MNB(X_train, y_train, X_test, y_test, tdidf=True):
    # This function allows two experiments on MNB model: if tdidf is true, we use tdidf as processor; else bag of words (binary).
    if tdidf:
        pclf = Pipeline([
            ('vect', TfidfVectorizer(encoding='utf-8',strip_accents='unicode',max_df=0.5, min_df=2)),
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
    print(metrics.classification_report(y_test, y_pred))
    return pclf
