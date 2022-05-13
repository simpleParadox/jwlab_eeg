from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def cross_validaton_nested_svm(X_train, y_train, X_test, y_test):
    results = []
    preds = []
    animacy_results = []
    tgm_matrix_temp = np.zeros((120, 120))
    scoring = 'accuracy'

    ## Define the hyperparameters.
    # ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    ridge_params = {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    best_alphas = []
    all_word_pairs_2v2 = {}
    for i in range(len(X_train)):
        temp_results = {}
        for j in range(len(X_train[i])):

            # model = LinearSVC()
            model = LogisticRegression(max_iter=500, n_jobs=-1)

            # Change the labels to 0 or 1 for animate and inanimate respectively.
            y_train_w = np.array(y_train[i][j])
            y_test_w = np.array(y_test[i][j])

            y_train_w[y_train_w < 8] = 0
            y_train_w[y_train_w >= 8] = 1

            y_test_w[y_test_w < 8] = 0
            y_test_w[y_test_w >= 8] = 1

            clf = GridSearchCV(model, ridge_params, scoring=scoring, n_jobs=-1, cv=5)
            clf.fit(X_train[i][j], y_train_w)
            best_alphas.append(clf.best_params_)
            y_pred = clf.predict(X_test[i][j])

            # Typecasting to int just in case.
            y_pred = y_pred.astype(int)

            testScore = accuracy_score(y_test_w, y_pred)

            if j in temp_results.keys():
                temp_results[j] += [testScore]
            else:
                temp_results[j] = [testScore]

        results.append(temp_results)


    return results, animacy_results, preds, tgm_matrix_temp, all_word_pairs_2v2
