from sklearn.feature_extraction.text import CountVectorizer

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

# Marco fix the organization if you want :p
def BernoulliNB(self, x, y):
  count_vect = CountVectorizer(max_df=0.5,min_df=5).fit(X_train)
  bnb = BNB(count_vect)
  bnb.train(X_train, y_train)
  return bnb
