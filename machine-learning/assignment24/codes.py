import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

np.random.seed(20180631)

data_df = pd.read_csv('./origin_data.csv')

data_df = data_df.fillna(data_df.mean())

target = data_df.columns[-1]
predictor = [x for x in data_df.columns if x not in [target]]

X_train = data_df[predictor]
y_train = data_df[target]


import seaborn as sns
# sns.heatmap(data_df.corr())
from sklearn import preprocessing
from sklearn import linear_model
clf = linear_model.LinearRegression()

clf.fit(X_train, y_train)
# Make predictions using the testing set
y_pred = clf.predict(X_train)


def plot():
    label = predictor[3]

    X_plot = X_train[label]

    plt.plot(X_plot, y_pred, color='blue', linewidth=3, alpha=.5)

    plt.scatter(X_plot, y_train,  color='black', alpha=.5)

    plt.scatter(X_plot, y_pred, color='red', alpha=.5)
    #plt.plot([X_plot.max(), X_plot.min()], [y_pred.max(), y_pred.min()], 'k--')
    plt.xticks(())
    plt.yticks(())
    plt.savefig('number vs %s' % label, dpi=500)
    plt.show()


print('stage2')
from sklearn import cross_validation
from sklearn import metrics


def split_size_effect(X_df, y_df, clf, score):
    ans = []
    for i in np.linspace(0, 1, 11)[1:-1]:
        X_trains, X_test, y_trains, y_test = cross_validation.train_test_split(
            X_df, y_df, test_size=i)
        clf.fit(X_trains, y_trains)
        train_score = score(y_trains, clf.predict(X_trains))
        test_score = score(y_test, clf.predict(X_test))
        ans.append([len(X_trains), train_score, test_score])
    return ans


x = split_size_effect(
    X_train, y_train, linear_model.LinearRegression(), metrics.mean_squared_error)
pd.DataFrame(x,columns=['train size', 'train score', 'test score'])