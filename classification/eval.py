from sklearn.model_selection import train_test_split
import sys
import numpy as np


def eval_normal(model, X, y, num_trials, test_size=0.2, random_state=0):
    errs = np.zeros(num_trials)
    for i in range(num_trials):
        X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=test_size, random_state=random_state+i)
        model.fit(X_train, y_train)
        
        errs[i] = np.mean(model.predict(X_test) != y_test)
        sys.stdout.write('\r')
        percent = (i + 1) / num_trials
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*percent), percent*100))
    return errs

def eval_across_participants(model, X, y, p, num_trials, test_size=0.2, random_state=0):
    num_participants = int(np.max(p)) + 1
    errs = np.zeros((num_participants, num_trials))
    for i in range(num_trials):
        X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(X, y, p, test_size=test_size, random_state=i+random_state)
        model.fit(X_train, y_train)
        
        for j in range(num_participants):
            if len(y_test[p_test == j]) == 0:
                errs[j, i] = 0.5
            else:
                errs[j, i] = np.mean(model.predict(X_test[p_test == j]) != y_test[p_test == j])
        
        sys.stdout.write('\r')
        percent = (i + 1) / num_trials
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*percent), percent*100))
    return errs