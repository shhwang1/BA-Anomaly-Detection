import argparse

def Parser1():
    parser = argparse.ArgumentParser(description='3_Anomaly-Detection')

    # data type
    parser.add_argument('--data-path', type=str, default='./data/')
    parser.add_argument('--data-type', type=str, default='Seismic.csv',
                        choices = ['Cardiotocogrpahy.csv', 'Glass.csv', 'Lympho.csv', 'Seismic.csv', 'Shuttle.csv', 'Annthyroid.csv', 'Mammography.csv'])           
    parser.add_argument('--seed', type=int, default=7760)              

    # Choose methods
    parser.add_argument('--method', type=str, default='IsolationForest',
                        choices = ['LOF', 'KNN', 'Autoencoder', 'IsolationForest'])

    # Hyperparameters for Anomaly Detection
    parser.add_argument('--neighbors', type=int, default=5) 
    parser.add_argument('--threshold', type=float, default=4)
    parser.add_argument('--masking', type=bool, default=True)
    parser.add_argument('--masking-ratio', type=float, default=0.4)
    parser.add_argument('--split', type=int, default=0.2)   
    parser.add_argument('--epoch', type=int, default=300)  
    parser.add_argument('--anomaly-ratio', type=float, default=0.01)  
    parser.add_argument('--batch-size', type=int, default=128) 
    parser.add_argument('--n-estimators', type=int, default=400) 
    parser.add_argument('--max-samples', type=str, default='auto')

    return parser