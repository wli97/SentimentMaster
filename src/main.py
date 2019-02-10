from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(max_df=0.5,min_df=2).fit(X_train)
