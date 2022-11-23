from dataclasses import dataclass
from sklearn_pandas import DataFrameMapper
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from pycox.models import LogisticHazard
import numpy as np

@dataclass
class Config:
    # Hyper-parameters
    test_size = 0.15
    val_size = 0.15
    train_size = 0.7
    seed = 1234
    train_dataset_fp = 'finaldata-June26.csv'

    cols_standardize = ['diagnosic-age',
                        'edjucation',
                        'G',
                        'BF',
                        'FH',
                        'T',
                        'N',
                        'stage',
                        'Grade',
                        'CT',
                        'RT',
                        'hormone subgroup']
    cols_leave = ['marital-status', 'AB', 'Path', 'LVI', 'Surgery', 'HT']
    study_feature = 'hormone subgroup'

    num_durations = 10  # discretization

    params = {  # search params
        'learning_rate': [0.01],
        'num_nodes': [[32, 64, 128, 256]],
        'dropout': [0.3],
        'batch_size': [256],
    }

    cv = 5


def get_feature_distribution(feature_index, dataset, standardize):
    filtered_individuals = defaultdict(list)
    dataset_reverse = standardize[feature_index][1].inverse_transform(dataset)

    for i in range(len(dataset)):
        individual = dataset_reverse[i]
        filtered_individuals[int(individual[feature_index])].append(i)
    print(sorted([(k, len(v)) for k, v in filtered_individuals.items()]))
    return filtered_individuals


def normal_standard_scalar(df_train, df_val, df_test):
    standardize = [([col], StandardScaler()) for col in Config.cols_standardize]
    leave = [(col, None) for col in Config.cols_leave]
    x_mapper = DataFrameMapper(standardize + leave)
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')
    return x_train, x_val, x_test


def label_transform(num_durations, df_train, df_val, df_test):
    label_translator = LogisticHazard.label_transform(num_durations, dtype=np.float64)

    get_features = lambda df: (df['time'].values, df['event'].values)

    y_train = label_translator.fit_transform(*get_features(df_train))
    y_val = label_translator.transform(*get_features(df_val))

    # We don't need to transform the test labels, why not?
    durations_test, events_test = get_features(df_test)
    return y_train, y_val, durations_test, events_test, label_translator.cuts, label_translator
