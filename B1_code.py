import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, base_learner):
        self.base_learner = base_learner
        self.learner_list=[]
        self.scalar=StandardScaler()
        self.test_errors={}
        self.alpha_values={}
        self.train_error = {}

    def train(self, X_train, y_train, T):
        X_train_scaled = self.scalar.fit_transform(X_train)
        kf = KFold(n_splits=10, random_state=42, shuffle=True)

        t_values = list(range(1,T + 1))

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

            w = np.ones(len(y_train_cv)) / len(y_train_cv)
            base_classifiers = []

            for t in t_values:
                if len(self.learner_list) <= t:
                    base = self.base_learner(max_depth=1)
                    self.learner_list.append(base)
                else:
                    base = self.learner_list[t]

                base.fit(X_train_cv, y_train_cv, sample_weight=w)

                y_pred_train = base.predict(X_train_cv)
                err_t = max(np.sum(w * (y_pred_train != y_train_cv)) / np.sum(w), 1e-16)

                alpha_t = 0.5 * np.log((1 - err_t) / err_t)

                w = w * np.exp(-alpha_t * y_train_cv * y_pred_train)
                w = w / np.sum(w)

                base_classifiers.append(base)
                self.alpha_values[t]=alpha_t

                y_pred_val = np.zeros(len(y_val_cv))
                for i in range(1,t + 1):
                    clf = base_classifiers[i-1]
                    alpha = self.alpha_values[i]
                    y_pred_val += alpha * clf.predict(X_val_cv)
                y_pred_val = np.sign(y_pred_val)

                val_error_t = np.mean(y_pred_val != y_val_cv)
                if val_error_t == 0:
                    break

                self.train_error[t]=val_error_t

        mean_cv_error=0
        for i in range(1,T+1):
            mean_cv_error += self.train_error[i]
        return mean_cv_error/T

    def test(self, X_test, Y_test, T):
        t_values = list(range(1,T+1))

        for t in t_values:
            base_classifiers = self.learner_list[:t + 1]

            y_pred_test = np.zeros(len(X_test))
            for i, clf in enumerate(base_classifiers):
                y_pred_test += self.alpha_values[i+1] * clf.predict(X_test)
            y_pred_test = np.sign(y_pred_test)

            test_error_t = np.mean(y_pred_test != Y_test)
            self.test_errors[T]=test_error_t


    def plot_train_errors_t(self, T_values):
        from operator import itemgetter
        y=itemgetter(*T_values)(self.train_error)
        plt.plot(T_values,y, label='Training Error')
        plt.xlabel('Number of Boosting Rounds (t)')
        plt.ylabel('Error Rate')
        plt.title('Training Errors vs. Number of Boosting Rounds (t)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_train_test_errors_vs_t(self,T):
        from operator import itemgetter
        t_values = list(range(1,T + 1))
        train_error = itemgetter(*t_values)(self.train_error)
        plt.plot(t_values, train_error, label='Training Error')
        test_error = itemgetter(*t_values)(self.test_errors)
        plt.plot(t_values, test_error, label='Test Error')
        plt.xlabel('Number of Boosting Rounds (t)')
        plt.ylabel('Error Rate')
        plt.title('Training and Test Errors vs. Number of Boosting Rounds (t)')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv('spambase.data.shuffled', header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=3450, test_size=1151, random_state=42)

    base_learner = DecisionTreeClassifier

    T_values = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    train_error={}
    adaBoost_classifier = AdaBoostClassifier(base_learner)

    for T in T_values:
        #Train
        err=adaBoost_classifier.train(X_train, y_train, T)
        train_error[T]=err


    # Best T
    best_T = min(train_error, key=lambda x: adaBoost_classifier.train_error[x])
    adaBoost_classifier.plot_train_errors_t(T_values)

    for t in list(range(1,best_T+1)):
        adaBoost_classifier.test(X_test,y_test,t)

    adaBoost_classifier.plot_train_errors_t(T_values)
    adaBoost_classifier.plot_train_test_errors_vs_t(best_T)