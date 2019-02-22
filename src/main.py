from IO import read_data, read_data_txt, read_test, submitModel
from CrossV import validate_model
from Models import base_model, predict_BNB, MNB, SVM

# Read training data (only run commented lines once)
# read_data('train/pos/', 1)
# read_data('train/neg/', 0)
# read_test('test/')

# Construct X, y vectors for training data
positive = read_data_txt('1.txt', 1)
negative = read_data_txt('0.txt', 0)
X = positive[0] + negative[0]
y = positive[1] + negative[1]

# Validate MNB and SVM
print(validate_model(5, X, y, base_model))
print(validate_model(5, X, y, predict_BNB))
print(validate_model(5, X, y, MNB, tfidf = True))
print(validate_model(5, X, y, SVM))

# Submit model
submit = read_data_txt('submit.txt', 0)
submitModel(MNB, X, y, submit[0], 'output1.csv')