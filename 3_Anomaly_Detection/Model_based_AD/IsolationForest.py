import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def IsolationForestAD(args):
    data = pd.read_csv(args.data_path + args.data_type)
    data = data.sort_values(by=['y'])
    data.reset_index(inplace=True)
    data.drop(['index'], axis=1, inplace=True)

    scaler = MinMaxScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

    for i in range(len(data)):
        if data.iloc[:, -1][i] == 1:
            data.iloc[:, -1][i] = -1

    for i in range(len(data)):
        if data.iloc[:, -1][i] == 0:
            data.iloc[:, -1][i] = 1

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if_model = IsolationForest(n_estimators = args.n_estimators,
                                max_samples = args.max_samples,
                                contamination = args.anomaly_ratio,
                                random_state = args.seed)

    if_model.fit(X)
    y_pred = if_model.predict(X)

    accuracy = accuracy_score(y_pred, y)
    precision = precision_score(y_pred, y)
    recall = recall_score(y_pred, y)
    f1score = f1_score(y_pred, y)
    
    print('Isolation Forest Performance')
    print('Accuracy :', accuracy, " Precision :", precision)
    print('Recall :', recall, 'F1-Score :', f1score)
    
    
    pred_outlier_idx = np.where(y_pred == -1)[0]
    pred_normal_idx = np.where(y_pred == 1)[0]
    real_outlier_idx = y[y==-1].index
    real_normal_idx = y[y==1].index

    random_var1 = random.choice(X.columns.to_list())
    random_var2 = random.choice(X.columns.to_list())

    while random_var2 == random_var1:
        random_var2 = random.choice(X.columns.to_list())

    plt.figure(figsize=(10, 10))
    plt.title("Isolation Forest Anomaly Detection")
    plt.scatter(X[random_var1][pred_outlier_idx], X[random_var2][pred_outlier_idx], c='red', s = 80, edgecolor='k', label = 'outlier(prediction) instances')
    plt.scatter(X[random_var1][pred_normal_idx], X[random_var2][pred_normal_idx], c='white', s = 80, edgecolor='k', label = 'normal(prediction) instances')
    plt.legend()
    plt.show()
    plt.clf()

    plt.title("Isolation Forest Anomaly Detection")
    plt.scatter(X[random_var1][real_outlier_idx], X[random_var2][real_outlier_idx], c='red', s = 80, edgecolor='k', label = 'outlier(real) instances')
    plt.scatter(X[random_var1][real_normal_idx], X[random_var2][real_normal_idx], c='white', s = 80, edgecolor='k', label = 'normal(real) instances')
    plt.legend()
    plt.show()