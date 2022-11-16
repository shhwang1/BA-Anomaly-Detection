import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def KNearestNeighborAD(args):
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

    real_abnormal_data = data[data['y'] == -1].index
    real_normal_data = data[data['y'] == 1].index

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    knnbrs = NearestNeighbors(n_neighbors = args.neighbors)
    knnbrs.fit(X_scaled)
    distances, _ = knnbrs.kneighbors(X_scaled)

    outlier_idx = np.where(distances.mean(axis=1) > args.threshold)
    normal_idx = list(set(range(len(X_data))) - set(outlier_idx[0]))
    y_pred = np.ones(len(X_data), dtype=int)
    y_pred[outlier_idx] = -1

    accuracy = accuracy_score(y_pred, y_data)
    precision = precision_score(y_pred, y_data)
    recall = recall_score(y_pred, y_data)
    f1score = f1_score(y_pred, y_data)
    

    print('K-NN Anomaly Detection') 
    print('neighbors :', args.neighbors, ', threshold :', args.threshold)
    print('Accuracy :', accuracy, " Precision :", precision)
    print('Recall :', recall, 'F1-Score :', f1score)

    random_var1 = random.choice(X_data.columns.to_list())
    random_var2 = random.choice(X_data.columns.to_list())

    while random_var2 == random_var1:
        random_var2 = random.choice(X_data.columns.to_list())

    plt.figure(figsize=(10, 10))
    plt.title("K-NN Anomaly Detection")
    plt.scatter(X_data[random_var1][outlier_idx[0]], X_data[random_var2][outlier_idx[0]], c='red', s = 80, edgecolor='k', label = 'outlier(prediction) instances')
    plt.scatter(X_data[random_var1][normal_idx], X_data[random_var2][normal_idx], c='white', s = 80, edgecolor='k', label = 'normal(prediction) instances')
    plt.legend()
    plt.show()
    plt.clf()

    plt.figure(figsize=(10, 10))
    plt.title("K-NN Anomaly Detection")
    plt.scatter(X_data[random_var1][real_abnormal_data], X_data[random_var2][real_abnormal_data], c='red', s = 80, edgecolor='k', label = 'outlier(real) instances')
    plt.scatter(X_data[random_var1][real_normal_data], X_data[random_var2][real_normal_data], c='white', s = 80, edgecolor='k', label = 'normal(real) instances')
    plt.legend()
    plt.show()
    