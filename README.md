# Anomaly Detection Tutorial

## Table of Contents

#### 0. Overview of Anomaly Detection
___
### Density-based Anomaly Detection
#### 1. Local Outlier Factor (LOF) 
***
### Distance-based Anomaly Detection
#### 2. K-Neareast Neighbor Anomaly Detection
*****
### Model-based Anomaly Detection
#### 3. Auto-encoder
#### 4. Isolation Forest (IF)
___

## 0. Overview of Anomaly Detection
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/201886259-e8dafab7-55fe-480a-8428-e131f93ee1cc.png" width="600" height="400"></p>   

### - What is "Anomaly?"
Let's look at the beer bottles above. Beer bottles seem to be neatly placed, but you can see an unusual bottle of beer. Like this picture,  small number of unusual patterns of data present in this uniform pattern are called 'Anomaly'. This discovery of Anomaly is called Anomaly Detection, and there are some very important moments to utilize this task. Think about the manufacturing process of making cheap screws. The 'Anomaly' screws that occur during this process do not appear to be very fatal because the value per piece is very small. However, 'Anomaly' semiconductors that occur in expensive semiconductor processes are very expensive per unit, so even a small number of Anomaly products are very damaging to the process. That's why it's important to find the anomaly well.

### - What is difference between "Classification" & "Anomaly Detection"?
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/201891495-1dc08074-9e6d-4132-90fe-cf012bc63c39.png" width="600" height="300"></p> 

If so, you may be curious about the difference between Classification and Anomaly Detection. Classification and Anomaly Detection problems differ by the assumptions below.

### 1. Anomaly Detection utilizes very unbalanced data.
The classification problem aims to literally classify data with an appropriate number of different classes. However, in explaining the definition of Anomaly, we mentioned the purpose of Anomaly Detection. Anomaly Detection aims to detect a small number of anomaly data embedded between the majority of data that are even. Therefore, we also use dataset consisting of very few anomaly data and most normal data.

### 2. Anomaly Detection trains the model only with normal data.
The classification problem uses both normal and abnormal data for training. However, Anomaly Detection trains the model only with normal data, and designs to discover anomaly data for datasets containing abnormal data in the test phase. After the model learns the feature of normal data well, then when abnormal data that has not been learned enters as input, it is judged as anomaly data.

Next, the methodology of 1) densitiy-based anomaly detection, 2) distance-based anomaly detection, 3) model-based anomaly detection will be described.

## Dataset

We use 7 unbalanced datasets for Anomaly Detection (Cardiotocogrpahy, Glass, Lympho, Seismic, Shuttle, Annthyroid, Mammography)   

Cardiotocogrpahy dataset : <https://archive.ics.uci.edu/ml/datasets/cardiotocography>     
Glass dataset : <http://odds.cs.stonybrook.edu/glass-data/>   
Lympho dataset : <https://archive.ics.uci.edu/ml/datasets/Lymphography>   
Seismic datset : <http://odds.cs.stonybrook.edu/seismic-dataset/>      
Shuttle dataset : <http://odds.cs.stonybrook.edu/shuttle-dataset/>   
Annthyroid dataset : <http://odds.cs.stonybrook.edu/annthyroid-dataset/>   
Mammography dataset : <http://odds.cs.stonybrook.edu/mammography-dataset/>   

In all methods, data is used in the following form.
``` C
import argparse

def Parser1():
    parser = argparse.ArgumentParser(description='3_Anomaly Detection')

    # data type
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--data-type', type=str, default='Cardiotocogrpahy.csv',
                        choices = ['Cardiotocogrpahy.csv', 'Glass.csv', 'Lympho.csv', 'Seismic.csv', 'Shuttle.csv', 'Annthyroid.csv', 'Mammography.csv'])            
                        
data = pd.read_csv(args.data_path + args.data_type)

X_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
```
___

## Density-based Anomaly Detection

### 1. Local Outlier Factor (LOF)

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/201899094-fd568fa0-1f49-44b0-b249-d4597427620f.png" width="600" height="300"></p> 

Local Outlier Factor is an Anomaly Detection method that considers the relative density of data around an instance. There are two key elements of LOF, 1. k-distance and 2. reachability distance.

![image](https://user-images.githubusercontent.com/115224653/201900052-296dbd21-1c64-4044-b44f-3e8abd597ae2.png)

#### 1. K-distance
The K-distance has the largest distance value among the distances from the K instances arbitrarily determined around the corresponding instance. $N_k(p)$ means a set of all instances that are closer than k-distance from a particular instance.

#### 2. Reachability distance
Reachability distance has a larger value between the distance from a particular instance to the distance of another instance and the k-distance value.

The formula for calculating the LOF score using k-distance and reachability distance is as follows. The outlier is judged based on the score.
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/201903345-a26f7109-b90b-4915-afc6-a488d8ce1304.png" width="750" height="300"></p> 

#### Python Code
``` py
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
```
In all four data sets used, normal label is set to 0 and abnormal label is set to 1. Subsequently, for compatibility with the package used as an evaluation index, it went through the process of changing the normal to label '1' and the abnormal to label '-1'.

```py
    n_outliers = len(data[data.iloc[:, -1]==-1])
    ground_truth = np.ones(len(X_data), dtype=int)
    ground_truth[-n_outliers:] = -1

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    lof = LocalOutlierFactor(n_neighbors = args.neighbors, contamination=.03)
    y_pred = lof.fit_predict(X_scaled)
    n_errors = (y_pred != ground_truth).sum()
    accuracy = accuracy_score(y_pred, y_data)
    precision = precision_score(y_pred, y_data)
    recall = recall_score(y_pred, y_data)
    f1score = f1_score(y_pred, y_data)
    
    print('LOF neighbors =', args.neighbors)
    print('Accuracy :', accuracy, " Precision :", precision)
    print('Recall :', recall, 'F1-Score :', f1score)
```
In Local Outlier Factor algorithm, there's one hyperparameter, 'n_neighbors'. This represents the value of k for obtaining k-distance in the above description. The experimental results according to the change in the neighbors value will be examined in the analysis chapter later.
___
## Density-based Anomaly Detection

### 2. K-Neareast Neighbor Anomaly Detection
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/202087248-f83e7dd4-0b81-4060-9df2-015a9eb24696.png" width="400" height="300"></p>
The k-nearest neighbor-based anomaly detection problem is a problem of determining whether or not this is an outlier by calculating the distance between k neighbors arbitrarily determined from a single instance. Therefore, setting the number of neighbors k, which is a hyperparameter, is directly related to performance. In the case of outlier data, it is determined that it is an outlier because the distance value from k-neighbors is large.

#### Python Code
``` py
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
```
In all four data sets used, normal label is set to 0 and abnormal label is set to 1. Subsequently, for compatibility with the package used as an evaluation index, it went through the process of changing the normal to label '1' and the abnormal to label '-1'.

```py
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
```
'knnbrs' refers to the Nearest neighbor model according to the number of neighbors, which is a hyperparameter. Then, the distance from k neighbors is calculated, and the average of the corresponding distance values is calculated. args.threshold represents a threshold to be determined as an outlier, which is also set as a hyperparameter. As will be seen in the experiment, the setting of threshold is directly related to performance.
___
## Model-based Anomaly Detection

### 3. Auto-encoder
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/202088547-79e0ba8f-9cc7-41ca-b632-0eeeb106fc1e.png" width="600" height="300"></p>
Auto-encoder refers to a model that compresses input data into a late vector through an encoder and then restores it to the same shape as input data through a decoder. One might wonder how this model, which simply compresses and reconstructs input data, can be used for the anomaly detection problem. The important part is the second content mentioned earlier that anomaly detection is different from the classification problem. Since the model is trained to reconstruct input data using only normal data in the train phase, it is impossible to properly reconstruct outlier data when it enters input data in test phase, and the data is judged as outlier data. 

#### Python Code
``` py
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
```
'mad score' is used as a threshold for the anonymous detection problem using Auto-encoder. The reason why I use mad score is because the mean and standard deviation are themselves susceptible to the presence of anomalies. As a criterion for determining an outlier, MSE value between the reconstructed data and the existing data is obtained and converted into MAD_Score, and if it is higher than the threshold value, it is determined as an outlier. A brief formula for mad score is as follows.
<p align="center"><img src="https://user-images.githubusercontent.com/115224653/202091036-5a3155bc-8c45-4ab5-976c-6f56703983ab.png" width="600" height="150"></p>

``` py
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
```
``` py
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
```
We used tensorflow to build a deep learning model. The model structure is very simple. The existing feature is compressed into 16-dim -> 8-dim -> 4dim -> 2dim and reconstructed with the existing shape. We conducted an additional experiment applying masking, which replaces random information among input data with a value of 0. The following is a architecture of the model structure, and a comparative experiment according to masking will be confirmed later in analysis. 

### Masking

<p align="center"><img src="https://user-images.githubusercontent.com/115224653/202111809-441be92d-89b2-491c-b744-29887a9758c0.png" width="600" height="300"></p>

Masking is one of the data augmentation techniques and it is usually used a lot when dealing with image data.. It can be thought that distorting the data value can rather interfere with training. However, it has been experimentally demonstrated that by omitting some values, the encoder that compresses the data and the decoder that reconstruct it can better understand the features of the data. In the experiment, how much of the data to mask will be set as a hyperparameter and the experimental performance will be compared. 

![image](https://user-images.githubusercontent.com/115224653/202095568-700242b1-22da-4f5b-bfd0-6dab955953f1.png)

___
### 4. Isolation Forest
![image](https://user-images.githubusercontent.com/115224653/202096384-0b20d985-336b-4b3a-8da9-59eda25b3f41.png)
Isolation Forest is a model that literally uses straight lines to separate data, determining whether it is outlier data based on the number of straight lines required to isolate. The picture above shows that the anomaly data needs a small number of straight lines to isolate, and the normal data needs a large number of straight lines to isolate. It is a model that determines whether the data is abnormal based on the number of straight lines required to isolate the data.

#### Python Code
``` py
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
```
``` py
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
```
In 'IsolationForest' model, we used four arguments, 'n_estimators', 'max_samples', 'contamination', 'random_state'. 'n_estimators' represent the number of trees and the default value is 100. 'max_samples' represents the number of data sampled for each tree by the specified number of integers. If it is a real number between 0 and 1, the total number of data is multiplied by the real number, and if not specified, the total number of data is used, and the smaller number of 256 is used. "contamination" represents the ratio of outliers in all data, and according to this ratio, the threshold of the score to be determined as outliers is defined. 'random_state' represents the seed value, as you know.
___

## Analysis
Since the outlier threshold determined according to the methodology to be tested is different, the experiment was first conducted by methodology. The threshold is a point value corresponding to 95% of the outer score's max value, and has a different value for each dataset. Therefore, since the outer score calculated by the model is different, the comparison by model is meaningless. Therefore, the main content is not comparison by model, but performance comparison through hyperparameter tuning for one model.

### [Experiment 1.] Comparison of Local Outlier Factor performance by neighbors hyperparameter changes

In the Local Outlier Factor methodology, 'k' as defined by the hyperparameter is very important because it directly affects the calculation of k-distance and reachability, which are important elements of the methodology. In the existing code, the default value of k is 20. We compared the performance for a total of six cases: 5, 10, 15, 20, 30, 50. The evaluation metrics were set to accuracy and F1 score.

|  Accuracy  | Dataset              |  K = 5 |   K = 10 |   K = 15 |   K = 20 |   K = 30 |   K = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.9001 |        **0.9022** |         0.8990|        0.8990 |        0.9000|        0.9001|
|          2 | Glass                 |    0.9439 |        0.9439|        **0.9533** |        0.9439 |        0.9420|        0.9439 |
|          3 | Lympho                 |    0.9595 |        0.9595 |        **0.9730** |        0.9730 |        0.9730 |        0.9730 |
|          4 | Seismic                 |    0.9249 |        0.9249 |        0.9249 |        0.9249 |        0.9249 |        0.9241 |
|          5 | Shuttle                 |    0.9220 |        0.9226 |        **0.9233** |        0.9231 |        0.9230 |        0.9221 |
|          6 | Annthyroid                 |    0.9167 |        0.9214 |        0.9202 |        0.9208 |        0.9219 |        **0.9228** |
|          7 | Mammography                 |    0.9700 |        0.9691 |        0.9701 |        0.9705 |        0.9708 |        **0.9717** |

|  F1-Score  | Dataset              |  K = 5 |   K = 10 |   K = 15 |   K = 20 |   K = 30 |   K = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.9484 |        **0.9484** |         0.9466|        0.9466 |        0.9472|        0.9472|
|          2 | Glass                 |    0.9712 |        0.9712|        **0.9760** |        0.9712 |        0.9712|        0.9712 |
|          3 | Lympho                 |    0.9792 |        0.9792 |        **0.9861** |        0.9861 |        0.9861 |        0.9861 |
|          4 | Seismic                 |    0.9610 |        0.9610 |        0.9610 |        0.9610 |        0.9610 |        0.9606 |
|          5 | Shuttle                 |    0.9593 |        0.9597 |        **0.9600** |        0.9599 |        0.9599 |        0.9594 |
|          6 | Annthyroid                 |    0.9565 |        0.9590 |        0.9584 |        0.9587 |        0.9593 |        **0.9597** |
|          7 | Mammography                 |    0.9847 |        0.9843 |        0.9848 |        0.9850 |        0.9852 |       **0.9856** |

Analyzing the experimental results can be summarized as follows.

#### 1. Datasets (Lympho, Seismic) with a small number of samples are insensitive to performance changes despite changes in K values.
#### 2. As the K value increases, the performance of datasets with a large number of samples (Anthyroid, Mammography) increases slightly.
#### 3. The high value of K did not necessarily mean that the performance was good.
#### 4. It did not seem to be significantly affected by the change in the K value.
___

### [Experiment 2.] Comparison of k-NN anomaly detection performance by neighbors hyperparameter changes
Like the Local Outlier Factor, the role of k-NN anonymous detection is also important for hyperparameter K. Experimentally check whether the role of K, which was insignificant in the Local Outlier Factor, is different in k-NN anonymous detection. The change pattern of K was configured in the same way as the Local Outlier Factor.

|  Accuracy  | Dataset              |  K = 5 |   K = 10 |   K = 15 |   K = 20 |   K = 30 |   K = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.9044 |        0.9049 |         0.9066|        0.9082 |        **0.9131**|        0.9121|
|          2 | Glass                 |    **0.9533** |        0.9439|       0.9393 |        0.9392 |        0.9392|        0.9252|
|          3 | Lympho                 |    0.9594 |        0.9594 |        0.9662 |        0.9662 |        0.9797 |        **0.9865** |
|          4 | Seismic                 |    0.9342 |        0.9342 |       **0.9342** |        0.9299 |        0.9276 |        0.9249 |
|          5 | Shuttle                 |    0.9284 |        0.9284 |        0.9284 |        0.9285 |        0.9285 |        0.9285 |
|          6 | Annthyroid                 |    0.9258 |        0.9257 |        0.9256 |        0.9257 |        0.9254 |        **0.9259** |
|          7 | Mammography                 |    0.9767 |        0.9767 |        0.9767 |        **0.9768** |        0.9766 |        0.9767 |

|  F1-Score  | Dataset              |  K = 5 |   K = 10 |   K = 15 |   K = 20 |   K = 30 |   K = 50 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.9498 |        0.9500 |         0.9508|        0.9517 |        **0.9541**|        0.9533|
|          2 | Glass                 |    **0.9761** |        0.9712|        0.9687 |        0.9685 |        0.9685|        0.9610 |
|          3 | Lympho                 |    0.9793 |        0.9793 |        0.9827 |        0.9827 |        0.9895 |        **0.9930** |
|          4 | Seismic                 |    0.9659 |        0.9659 |        **0.9660** |        0.9637 |        0.9624 |        0.9609 |
|          5 | Shuttle                 |    0.9629 |        0.9629 |        0.9629 |        0.9629 |        0.9629 |        0.9629 |
|          6 | Annthyroid                 |    0.9614 |        0.9614 |        0.9614 |        0.9614 |        0.9612 |        **0.9615** |
|          7 | Mammography                 |    0.9882 |        0.9882 |        0.9882 |        0.9882 |        0.9881 |       **0.9882** |

Analyzing the experimental results can be summarized as follows.

#### 1. It seems that the effect is weaker in k-NN than the weak effect of K in the Local Outlier Factor.
#### 2. Cardiotography datasets have lower performance than other datasets. This seems to be a characteristic of a dataset that should not be approached based on distance (K).
#### 3. K's role seems to be meaningful only when the threshold for determining the outsider is clearly established.

___
### [Experiment 3.] Effects of Masking on Auto-encoder
In Auto-encoder, the MAD_Score described above is set to threshold. In this experiment, the performance when masking is applied and when not applied is compared. If masking is applied, the ratio of masking, args.Compare the masking_ratio in 0.1, 0.2, 0.3, and 0.4 cases. Epoch is 300, batch size is 128.

#### 1) No Masking Case
|  Accuracy  | Dataset              |  Base |
|:----------:|:--------------------:|:------:|
|          1 | Cardiotocogrpahy            |    0.7554 |
|          2 | Glass                 |    0.7600 |
|          3 | Lympho                 |    0.9429 |
|          4 | Seismic                 |    0.7320 |
|          5 | Shuttle                 |    0.9865 |
|          6 | Annthyroid                 |    0.7479 |
|          7 | Mammography                 |    0.9100 |

|  F1-Score  | Dataset              |  Base |
|:----------:|:--------------------:|:------:|
|          1 | Cardiotocogrpahy            |    0.8211 |
|          2 | Glass                 |    0.8500 |
|          3 | Lympho                 |    0.9666 |
|          4 | Seismic                 |    0.8262 |
|          5 | Shuttle                 |    0.9907 |
|          6 | Annthyroid                 |    0.8448 |
|          7 | Mammography                 |    0.9519 |

And the following is the result table of applying masking.

|  Accuracy  | Dataset              |  Base |   Masking ratio = 0.1 |   Masking ratio = 0.2 |   Masking ratio = 0.3 |   Masking ratio = 0.4 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.7554 |        0.7377 |         **0.8560**|        0.7949 |        0.7929|
|          2 | Glass                 |    0.7600 |        0.8000|       0.7600 |        **0.8600** |        0.7400|
|          3 | Lympho                 |    **0.9429** |        0.8857 |        0.8000 |        0.8857 |        0.9143 |
|          4 | Seismic                 |    0.7320 |       **0.7810**|       0.6554|        0.7795|        0.7688|
|          5 | Shuttle                 |    0.9865|        0.9854|        0.9842|        0.8383|        **0.9868**|
|          6 | Annthyroid                 |    0.7479 |        **0.7602**|        0.7190|        0.7126|        0.7093|
|          7 | Mammography                 |    0.9100 |        **0.9104**|        0.8990|        0.8810|        0.8936|

|  F1-Score  | Dataset              |  Base |   Masking ratio = 0.1 |   Masking ratio = 0.2 |   Masking ratio = 0.3 |   Masking ratio = 0.4 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.8211 |        0.8261 |         **0.9004**|        0.8628 |        0.8575|
|          2 | Glass                 |    0.8500 |        0.8863|       0.8537 |        **0.9195**|        0.8395|
|          3 | Lympho                 |    **0.9666** |        0.9333 |        0.8814 |        0.9333 |        0.9474 |
|          4 | Seismic                 |    0.8262 |        **0.8563**|       0.7508|        0.8569|        0.8527|
|          5 | Shuttle                 |    0.9907|        0.9900|        0.9891|        0.8993|        **0.9916**|
|          6 | Annthyroid                 |    0.8448 |        **0.8497**|        0.8289|        0.8275|        0.8283|
|          7 | Mammography                 |    **0.9519** |        0.9497|        0.9428|        0.9318|        0.9438|

Analyzing the experimental results can be summarized as follows.

#### 1. In the dataset (Cardio, Glass), where the accuracy was relatively low, the masking effect appeared quite a bit.
#### 2. The effect was insignificant on the highly accurate datasets (Shuttle, Lympho, and Mammography).
#### 3. It was judged that it was worth trying masking on a dataset where the threshold was not accurately built.
___
### [Experiment 4.] Comparison of Isolation Forest anomaly detection performance by n_esitimators hyperparameter changes
The most important hyperparameter in the code of the Isolation Forest methodology is n_estimators, which represent the number of trees. In the case of max_sample hyperparameter, it can be set to 'auto', so it is utilized. The experimental table below shows the results according to the change in the number of n_estimators.

|  Accuracy  | Dataset              |  n_estimators = 10 |   n_estimators = 50 |   n_estimators = 100 |   n_estimators = 200 |   n_estimators = 400 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.9093|        0.9132|         **0.9143**|        0.9143|        0.9143|
|          2 | Glass                 |    0.9532|        **0.9533**|       0.9533|        0.9533|        0.9533|
|          3 | Lympho                 |    0.9730|        **0.9730**|        0.9730|        0.9730|        0.9730|
|          4 | Seismic                 |    0.9257|       **0.9280**|       0.9280|        0.9280|        0.9280|
|          5 | Shuttle                 |    0.9382|        **0.9384**|        0.9384|        **0.9419**|        0.9419|
|          6 | Annthyroid                 |    0.9328|        0.9283|        **0.9286**|        0.9283|        0.9283|
|          7 | Mammography                 |    0.9747 |        **0.9784**|        0.9757|        0.9747|        0.9751|

|  F1-Score  | Dataset              |  n_estimators = 10 |   n_estimators = 50 |   n_estimators = 100 |   n_estimators = 200 |   n_estimators = 400 |
|:----------:|:--------------------:|:------:|:--------:|:--------:|:--------:|:--------:|
|          1 | Cardiotocogrpahy            |    0.9521|        0.9541|         **0.9547**|        0.9547|        0.9547|
|          2 | Glass                 |    **0.9760**|        0.9760|       0.9760|        0.9760|        0.9760|
|          3 | Lympho                 |    **0.9861**|        0.9861|        0.9861|        0.9861|        0.9861|
|          4 | Seismic                 |    0.9614|       0.9626|       0.9626|        **0.9280**|        0.9626|
|          5 | Shuttle                 |    0.9678|        0.9679|        0.9679|        **0.9688**|        0.9688|
|          6 | Annthyroid                 |    **0.9649**|        0.9626|        0.9627|        0.9626|        0.9626|
|          7 | Mammography                 |    0.9871 |        **0.9890**|        0.9876|        0.9871|        0.9874|

Analyzing the experimental results can be summarized as follows.

#### 1. If the number of samples in the dataset(Glass, Lympho) is small, it is useless to increase n_estimators.
#### 2. In the case of Shuttle dataset with the largest number of samples, there was a change in performance until the n_estimators reached 200.
#### 3. Even if the number of samples was enough, increasing n_estimators did not mean improving performance(Annthyroid, Mammography).

___
## Conclusion

#### 1) For anomaly detection, it is very important to determine the threshold that determines whether it is an outlier.
#### 2) In the case of Isolation Forest, the percentage of the outlier is received as an argument and the threshold is determined by itself. 
#### 3) The IF's overall performance was good. However, in some cases, the Auto-encoder, which arbitrarily set the threshold value, showed better performance than Isolation Forest.
#### 4) I think that research on thresholds suitable for datasets and methodologies used is an essential field.
___

### Reference

- Business Analytics, Korea university (IME-654) https://www.youtube.com/watch?v=ODNAyt1h6Eg
- https://towardsdatascience.com/anomaly-detection-in-time-series-sensor-data-86fd52e62538
- https://www.researchgate.net/figure/The-anomaly-detection-and-the-classification-learning-schemas_fig1_282309055
- https://yjjo.tistory.com/45
- https://medium.com/dataman-in-ai/anomaly-detection-with-pyod-b523fc47db9
- https://www.assemblyai.com/blog/introduction-to-variational-autoencoders-using-keras/
- https://sh-tsang.medium.com/review-context-encoders-feature-learning-by-inpainting-bd181e48997
- https://towardsdatascience.com/how-to-perform-anomaly-detection-with-the-isolation-forest-algorithm-e8c8372520bc
