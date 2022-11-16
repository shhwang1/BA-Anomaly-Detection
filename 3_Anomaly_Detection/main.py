from args import Parser1
from Densitiy_based_AD.LOF import LocalOutlierFactorAD
from Distance_based_AD.KNN_AD import KNearestNeighborAD
from Model_based_AD.Autoencoder import AutoencoderAD
from Model_based_AD.IsolationForest import IsolationForestAD

def build_model():
    parser = Parser1()
    args = parser.parse_args()

    if args.method == 'LOF':
        model = LocalOutlierFactorAD(args)
    elif args.method == 'KNN':
        model = KNearestNeighborAD(args)
    elif args.method == 'Autoencoder':
        model = AutoencoderAD(args)
    else:
        model = IsolationForestAD(args)

    return model

def main():
    build_model()

if __name__ == '__main__':
    main()


# data = io.loadmat('./data/mammography.mat')
# df = pd.DataFrame(data['X'])
# df['y'] = data['y']
# df.to_csv('./data/Mammography.csv')