import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, TimeSeriesSplit
from sklearn_pandas import DataFrameMapper
from pycox.datasets import metabric

from pycox.evaluation import EvalSurv
import pandas as pd
import torch  # For building the networks
import torchtuples as tt  # Some useful functions
from collections import defaultdict

from estimators import LogisticHazardEstimator
from settings import Config, normal_standard_scalar, label_transform

if __name__ == '__main__':
    assert Config.test_size + Config.val_size + Config.train_size == 1, "total size is not equal to 1"
    np.random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    df = pd.read_csv(Config.train_dataset_fp)

    print("[INFO] Splitting Dataset. Stratify is set to `hormone subgroup`")
    df_train, df_test_val = train_test_split(df, test_size=Config.test_size + Config.val_size, random_state=Config.seed,
                                             stratify=df[Config.study_feature])
    df_val, df_test = train_test_split(df_test_val, test_size=Config.test_size / (Config.test_size + Config.val_size),
                                       random_state=Config.seed,
                                       stratify=df_test_val[Config.study_feature])

    x_train, x_val, x_test = normal_standard_scalar(df_train, df_val, df_test)

    feature_index = Config.cols_standardize.index(Config.study_feature)
    print(f"[INFO] Study feature is {Config.study_feature} with index: {feature_index}")
    y_train, y_val, durations_test, events_test, cuts, label_translator = label_transform(Config.num_durations,
                                                                                          df_train, df_val, df_test)

    get_features = lambda df: (df['time'].values, df['event'].values)
    duration_train, event_train = get_features(df_train)
    duration_val, event_val = get_features(df_val)
    duration_test, event_test = get_features(df_test)
    stack = lambda duration, event: np.hstack([duration.reshape((-1, 1)), event.reshape((-1, 1))])
    x_train_val = np.concatenate([x_train,
                                  x_val,
                                  # x_test
                                  ])
    y_train_val = np.concatenate([stack(duration_train, event_train),
                                  stack(duration_val, event_val),
                                  # stack(duration_test, event_test)
                                  ])
    estimator = LogisticHazardEstimator(label_translator, x_train.shape[1])
    gs = GridSearchCV(estimator, Config.params, refit=True, cv=KFold(n_splits=Config.cv), verbose=1, n_jobs=2)

    # gs = GridSearchCV(estimator, Config.params, refit=True,
    #                   cv=TimeSeriesSplit(n_splits=5).get_n_splits([x_train_val, y_train_val]), verbose=1)

    # x_train_val = np.concatenate([x_train, x_val])
    # y_train_val = np.concatenate([np.vstack(y_train).T, np.vstack(y_val).T])
    print("[INFO]", x_train_val.shape, y_train_val.shape)
    gs.fit(x_train_val, y_train_val)
    print(gs.best_score_, gs.best_params_)
