import sys
import os
import numpy as np
import json
import argparse

sys.path.append('../../DCASE-models')
from dcase_models.data.feature_extractor import FeatureExtractor
from dcase_models.util.files import load_json

sys.path.append('../')
from apnet.datasets import MedleySolosDb, GoogleSpeechCommands, ICBHIDataset


parser = argparse.ArgumentParser(description='Extract features')
parser.add_argument('-d', '--dataset', type=str, default='ICBHIDataset', help='dataset to extract the features')
parser.add_argument('-a','--augmentation', type=str, help='data augmentation type')
parser.add_argument('-p','--parameter', type=float, help='parameter of data augmentation')
args = parser.parse_args()

params = load_json(os.path.join('ICBHIDataset/APNet', 'config.json'))
params_dataset = params["datasets"][args.dataset]

if args.augmentation is not None:
    params['features']['augmentation'] = {args.augmentation: args.parameter}

# extract features and save files
feature_extractor = FeatureExtractor(**params['features'])

audio_folder = '/athena/rameaulab/store/mdo4009/ICBHI_final_database/'
feature_folder = '/athena/rameaulab/store/mdo4009/ICBHI_final_database/audio/features/MelSpectrogram/'

fold_list = ["train", "validate", "test"]
n_folds = len(fold_list)
if args.dataset == 'ESC50':
    feature_extractor.extract(audio_folder, feature_folder)
else:
    for fold in fold_list:
        print(fold)
        dataset = ICBHIDataset(audio_folder)

        audio_folder_fold = os.path.join(audio_folder, fold)
        feature_folder_fold = os.path.join(feature_folder, fold)
        # feature_extractor.extract(audio_folder_fold, feature_folder_fold)
        feature_extractor.extract(dataset)


# feature_extractor.save_mel_basis(os.path.join(feature_folder,'mel_basis.npy'))
# feature_extractor.save_parameters_json(os.path.join(feature_folder,'parameters.json'))