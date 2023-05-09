import os
import argparse
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

sys.path.append('../')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Datasets
from apnet.datasets import MedleySolosDb, GoogleSpeechCommands, ICBHIDataset
from dcase_models.data.datasets import UrbanSound8k

# Models
from dcase_models.model.models import SB_CNN, MLP
from apnet.model import APNet, AttRNNSpeechModel

# Features
from dcase_models.data.features import MelSpectrogram, Openl3

from apnet.layers import PrototypeLayer, WeightedSum

from dcase_models.data.data_generator import DataGenerator
from dcase_models.data.scaler import Scaler
from dcase_models.util.files import load_json
from dcase_models.util.files import mkdir_if_not_exists, save_pickle
from dcase_models.util.data import evaluation_setup


available_models = {
    'APNet' :  APNet,
    'SB_CNN' : SB_CNN,
    'MLP' : MLP,
    'AttRNNSpeechModel' : AttRNNSpeechModel
}

available_features = {
    'MelSpectrogram' :  MelSpectrogram,
    'Openl3' : Openl3
}

available_datasets = {
    'UrbanSound8k' :  UrbanSound8k,
    'MedleySolosDb' : MedleySolosDb,
    'GoogleSpeechCommands' : GoogleSpeechCommands,
    'ICBHIDataset': ICBHIDataset
}

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-d', '--dataset', type=str,
        help='dataset name (e.g. UrbanSound8k, ESC50, URBAN_SED, SONYC_UST)',
        default='ICBHIDataset'
    )
    parser.add_argument(
        '-f', '--features', type=str,
        help='features name (e.g. Spectrogram, MelSpectrogram, Openl3)',
        default='MelSpectrogram'
    )

    parser.add_argument(
        '-m', '--model', type=str,
        help='model name (e.g. MLP, SB_CNN, SB_CNN_SED, A_CRNN, VGGish)', default='APNet')

    parser.add_argument('-fold', '--fold_name', type=str, help='fold name',
                        default='train')
    parser.add_argument('-exp_name', '--exp_name', type=str, help='experiment name',
                        default='exp')

    parser.add_argument(
        '-mp', '--models_path', type=str,
        help='path to load the trained model',
        default='./'
    )
    parser.add_argument(
        '-dp', '--dataset_path', type=str,
        help='path to load the dataset',
        default='./'
        # default='/athena/rameaulab/store/mdo4009/ICBHI_final_database/features/FeatureExtractor/original/'
    )

    parser.add_argument('--c', dest='continue_training', action='store_true')
    parser.set_defaults(continue_training=False)

    parser.add_argument('-gpu', '--gpu_visible', type=str, help='gpu_visible',
                        default='0')

    args = parser.parse_args()

    # only use one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible

    print(__doc__)

    if args.dataset not in available_datasets:
        raise AttributeError('Dataset not available')

    if args.features not in available_features:
        raise AttributeError('Features not available')

    model_name = args.model
    if model_name not in available_models:
        model_name = args.model.split('/')[0]
        if model_name not in available_models:
            raise AttributeError('Model not available')

    # Model paths
    model_folder = os.path.join(args.models_path, args.dataset, args.model)

    # Get parameters
    parameters_file = os.path.join(model_folder, 'config.json')
    params = load_json(parameters_file)

    params_features = params['features'][args.features]
    params_dataset = params['datasets'][args.dataset]
    params_model = params['models'][model_name]

    # Get and init dataset class
    dataset_class = available_datasets[args.dataset]
    dataset_path = os.path.join(args.dataset_path, params_dataset['dataset_path'])
    dataset = dataset_class(dataset_path)

    if args.fold_name not in dataset.fold_list:
        raise AttributeError('Fold not available')

    # Get and init feature class
    features_class = available_features[args.features]
    features = features_class(**params_features)

    print('Features shape: ', features.get_shape(4.0))

    if not features.check_if_extracted(dataset):
        print('Extracting features ...')
        features.extract(dataset)
        print('Done!')


    folds_train, folds_val, folds_test = evaluation_setup(
        args.fold_name, dataset.fold_list,
        params_dataset['evaluation_mode'],
        use_validate_set=True
    )

    if model_name == 'APNet':
        outputs = ['annotations', features, 'zeros']
    else:
        outputs = 'annotations'


    data_gen_train = DataGenerator(
        dataset, features, folds=folds_train,
        batch_size=params['train']['batch_size'],
        shuffle=True, train=True, scaler=None,
        outputs=outputs
    )


    scaler = Scaler('standard') #normalizer=params_model['normalizer'])

    scaler.fit(data_gen_train)
    data_gen_train.set_scaler(scaler)
    print('Done!')

    # Pass scaler to data_gen_train to be used when data loading

    data_gen_val = DataGenerator(
        dataset, features, folds=folds_val,
        batch_size=params['train']['batch_size'],
        shuffle=False, train=False, scaler=scaler
    )

    # Define model
    features_shape = features.get_shape()
    if len(features_shape) > 2:
        n_frames_cnn = features_shape[1]
        n_freq_cnn = features_shape[2]
    else:
        n_freq_cnn = features_shape[1]
    n_classes = len(dataset.label_list)

    model_class = available_models[model_name]

    metrics = ['classification']

    # Set paths
    exp_folder = os.path.join(model_folder, args.exp_name)
    mkdir_if_not_exists(exp_folder, parents=True)

    if args.continue_training:
        model_container = model_class(
            model=None, model_path=exp_folder,
            custom_objects={
                'PrototypeLayer': PrototypeLayer,
                'WeightedSum': WeightedSum
            }
        )
        model_container.load_model_weights(exp_folder)
        params_model['train_arguments']['init_last_layer'] = 0
    else:
        if args.model == 'MLP':
            model_container = model_class(
                model=None, model_path=None, n_classes=n_classes,
                n_frames=None, n_freqs=n_freq_cnn,
                metrics=metrics,
                **params_model['model_arguments']
            )
        else:
            model_container = model_class(
                model=None, model_path=None, n_classes=n_classes,
                n_frames_cnn=n_frames_cnn, n_freq_cnn=n_freq_cnn,
                metrics=metrics,
                **params_model['model_arguments']
            )
        model_container.save_model_json(exp_folder)

    exp_folder_PATH = '/home/mdo4009/APNet/experiments/ICBHIDataset/APNet/binary_evaluation'
    model_container.load_model_weights(exp_folder_PATH)

    annotations = []
    predictions =[]
    for batch_index in range(0, len(data_gen_val)):
        X_val, Y_val = data_gen_val.get_data_batch(batch_index)
        n_files = len(X_val)
        for i in range(n_files):
            X = X_val[i]
            Y_predicted = model_container.model.predict(X)
            if type(Y_predicted) == list:
                Y_predicted = Y_predicted[0]
            predictions.append(np.argmax(Y_predicted))
            annotations.append(np.argmax(Y_val[i]))

    print(len(annotations), sum(annotations), len(predictions), sum(predictions))
    print(confusion_matrix(annotations, predictions))
    print('F1 score:', f1_score(annotations, predictions))


if __name__ == "__main__":
    main()
