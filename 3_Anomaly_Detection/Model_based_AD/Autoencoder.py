import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def mad_score(points):
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)

    return 0.6745 * ad / mad

def AutoencoderAD(args):
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

    outlier = data[data.iloc[:, -1] == -1]
    normal = data[data.iloc[:, -1] == 1]

    X_train = normal.iloc[:int(len(normal) * (1 - args.split))]
    X_test = normal.iloc[int(len(normal) * (1 - args.split)):].append(outlier)
    
    X_train, X_valid = train_test_split(X_train.iloc[:, :-1],
                                        test_size = args.split,
                                        random_state = args.seed)

    if args.masking:
        sample_idx_list = list(range(X_train.shape[0]))
        for i in range(len(X_train.iloc[:, :-1].columns)):
            random_idx = random.sample(sample_idx_list, int(args.masking_ratio * len(X_train)))
            for idx in random_idx:
                X_train.iloc[:, i][idx] = 0

    X_test, y_test = X_test.iloc[:, :-1], X_test.iloc[:, -1]


    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(2, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(X_test.shape[-1], activation='relu'),
        ])

    model.compile(optimizer="adam", 
                    loss="mse",
                    metrics=["mse"])
    # model.build(X_train.shape)
    # model.summary()
    
    save_model = tf.keras.callbacks.ModelCheckpoint(
    filepath='autoencoder_best_weights.hdf5',
    save_best_only=True,
    monitor='val_loss',
    verbose=0,
    mode='min'
)
    cb = [save_model]

    history = model.fit(
    X_train, X_train,
    shuffle=True,
    epochs=args.epoch,
    batch_size=args.batch_size,
    callbacks=cb,
    validation_data = (X_valid, X_valid)
    )

    reconstructions = model.predict(X_test)

    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
    z_scores = mad_score(mse)
    outliers = z_scores > args.threshold     
    outliers = outliers * 1
    outliers[outliers == 1] = -1
    outliers[outliers == 0] = 1

    real_outliers_idx = y_test[y_test == -1].index
    real_normal_idx = y_test[y_test == 1].index
    pred_outlier_idx = outliers[outliers == -1].index
    pred_normal_idx = outliers[outliers == 1].index

    accuracy = accuracy_score(outliers.to_numpy(), y_test.to_numpy())
    precision = precision_score(outliers.to_numpy(), y_test.to_numpy())
    recall = recall_score(outliers.to_numpy(), y_test.to_numpy())
    f1score = f1_score(outliers.to_numpy(), y_test.to_numpy())

    print(f"Detected {np.sum(outliers==-1):,} outliers in a total of {np.size(z_scores):,} transactions [{np.sum(outliers==-1)/np.size(z_scores):.2%}].")
    print('Accuracy :', accuracy, " Precision :", precision)
    print('Recall :', recall, 'F1-Score :', f1score)
    
    random_var1 = random.choice(X_test.columns.to_list())
    random_var2 = random.choice(X_test.columns.to_list())

    while random_var2 == random_var1:
        random_var2 = random.choice(X_test.columns.to_list())

    plt.figure(figsize=(10, 10))
    plt.title("Autoencoder Anomaly Detection")
    plt.scatter(X_test[random_var1][pred_outlier_idx], X_test[random_var2][pred_outlier_idx], c='red', s = 80, edgecolor='k', label = 'outlier(prediction) instances')
    plt.scatter(X_test[random_var1][pred_normal_idx], X_test[random_var2][pred_normal_idx], c='white', s = 80, edgecolor='k', label = 'normal(prediction) instances')
    plt.legend()
    plt.show()
    plt.clf()

    plt.title("Autoencoder Anomaly Detection")
    plt.scatter(X_test[random_var1][real_outliers_idx], X_test[random_var2][real_outliers_idx], c='red', s = 80, edgecolor='k', label = 'outlier(real) instances')
    plt.scatter(X_test[random_var1][real_normal_idx], X_test[random_var2][real_normal_idx], c='white', s = 80, edgecolor='k', label = 'normal(real) instances')
    plt.legend()
    plt.show()