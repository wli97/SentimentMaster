import numpy as np
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Performs k-fold cross validation using the model provided by predict_fct and returns the average accuracy/f1 score over all folds
def validate_model(folds, X, y, predict_fct, **kwargs):

    """
    folds: Must evenly divide the number of sample points. If 1, does 80/20 split and regular validation
    X: The X values for the whole data set
    y: The y values for the whole data set
    predict_fct: Model to use, must be a function that takes X_train, y_train, X_test, y_test (in that order) and returns the prediction on the test set
    """

    accuracy = 0
    f1 = 0

    # When 1, just perform 80/20 split
    if folds == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

        y_test = np.array(y_test)
        y_pred = predict_fct(X_train, y_train, X_test)

        accuracy = round(metrics.accuracy_score(y_test, y_pred), 3)
        f1 = round(metrics.f1_score(y_test, y_pred), 3)

        return(accuracy, f1)

    else:

        # Shuffle with seed to get more even distribution but still reproducible
        random.Random(2).shuffle(X)
        random.Random(2).shuffle(y)

        # Number of folds must cleanly divide the number of sample points
        num_per_fold = int(len(X)/folds)

        # Repeat experiment for each set being left out
        for i in range(folds):

            X_train, y_train = [], []
            X_test, y_test = [], []

            # Create train/test split
            for j in range(folds):
                if i == j:
                    X_test += X[j * num_per_fold:(j + 1) * num_per_fold]
                    y_test += y[j * num_per_fold:(j + 1) * num_per_fold]
                else:
                    X_train += X[j * num_per_fold:(j + 1) * num_per_fold]
                    y_train += y[j * num_per_fold:(j + 1) * num_per_fold]

            # Get the prediction and calculate metrics
            y_test = np.array(y_test)
            y_pred = predict_fct(X_train, y_train, X_test)

            curr_accuracy = round(metrics.accuracy_score(y_test, y_pred), 3)
            curr_f1 = round(metrics.f1_score(y_test, y_pred), 3)

            # print("Fold " + str(i) + ":")
            # print("Accuracy: " + str(accuracy))
            # print("f1: " + str(f1))

            accuracy += curr_accuracy
            f1 += curr_f1

        return (round(accuracy/folds, 3), round(f1/folds, 3))