import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def LocalOutlierFactorAD(args):
    data = pd.read_csv(args.data_path + args.data_type)
    data = data.sort_values(by=['y'])
    data.reset_index(inplace=True)
    data.drop(['index'], axis=1, inplace=True)

    for i in range(len(data)):
        if data.iloc[:, -1][i] == 1:
            data.iloc[:, -1][i] = -1

    for i in range(len(data)):
        if data.iloc[:, -1][i] == 0:
            data.iloc[:, -1][i] = 1

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    n_outliers = len(data[data.iloc[:, -1]==-1])
    ground_truth = np.ones(len(X_data), dtype=int)
    ground_truth[-n_outliers:] = -1

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    lof = LocalOutlierFactor(n_neighbors = args.neighbors, contamination=args.anomaly_ratio)
    y_pred = lof.fit_predict(X_scaled)
    n_errors = (y_pred != ground_truth).sum()
    accuracy = accuracy_score(y_pred, y_data)
    precision = precision_score(y_pred, y_data)
    recall = recall_score(y_pred, y_data)
    f1score = f1_score(y_pred, y_data)
    
    print('LOF neighbors =', args.neighbors)
    print('Accuracy :', accuracy, " Precision :", precision)
    print('Recall :', recall, 'F1-Score :', f1score)

    X_scores = lof.negative_outlier_factor_

    random_var1 = random.choice(X_data.columns.to_list())
    random_var2 = random.choice(X_data.columns.to_list())

    while random_var2 == random_var1:
        random_var2 = random.choice(X_data.columns.to_list())

    plt.title("Local Outlier Factor (LOF)")
    plt.scatter(X_data[random_var1], X_data[random_var2], color="k", s=3.0, label="Data points")
    # plot circles with radius proportional to the outlier scores
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    plt.scatter(
        X_data[random_var1],
        X_data[random_var2],
        s=1000 * radius,
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )
    plt.axis("tight")
    plt.xlabel("prediction errors: %d" % (n_errors))
    legend = plt.legend(loc="upper left")
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show()