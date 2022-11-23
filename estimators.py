from pycox.evaluation import EvalSurv
from pycox.models import LogisticHazard, CoxTime, MTLR
from pycox.models.cox_time import MLPVanillaCoxTime
from sklearn.base import BaseEstimator
import torchtuples as tt
import numpy as np
from sklearn.model_selection import train_test_split
from torchtuples.practical import MLPVanilla


class LogisticHazardEstimator(BaseEstimator):
    def __init__(
            self,
            label_translator,
            input_shape,
            learning_rate=1e-4,
            batch_norm=True,
            dropout=0.0,
            num_nodes=[32, 32],
            batch_size=256,
            epochs=100,
            output_features=10,
            early_stopping_patience=10,
            cnocordance_td_mode="antolini"
    ):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_features = output_features
        self.cnocordance_td_mode = cnocordance_td_mode
        self.label_translator = label_translator
        self._net = self.model = None
        self.early_stopping_patience = early_stopping_patience

    def fit(self, X, y, verbose=False):
        self._net = tt.practical.MLPVanilla(
            in_features=self.input_shape,
            num_nodes=self.num_nodes,
            out_features=self.output_features,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            output_bias=True,
        )
        x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1)
        self.model = LogisticHazard(self._net, tt.optim.Adam(self.learning_rate),
                                    duration_index=self.label_translator.cuts)

        # Sklearn needs the y inputs to be arranged as a matrix with each row
        # corresponding to an example but CoxPH needs a tuple with two arrays
        # y_ = (y[:, 0].astype(np.int64), y[:, 1].astype(np.float32))
        duration_train, events_train = (y_train[:, 0], y_train[:, 1].astype(np.int64))
        y_train = self.label_translator.transform(duration_train, events_train)

        duration_val, events_val = (y_dev[:, 0], y_dev[:, 1].astype(np.int64))
        y_val = self.label_translator.transform(duration_val, events_val)
        callbacks = [tt.callbacks.EarlyStopping(patience=self.early_stopping_patience)]
        self.model.fit(
            x_train,
            y_train,
            self.batch_size,
            self.epochs,
            val_data=(x_dev, y_val),
            verbose=verbose,
            callbacks=callbacks
        )
        return self

    def score(self, X, y):
        surv = self.model.predict_surv_df(X)

        ev = EvalSurv(
            surv,
            y[:, 0],  # time to event
            y[:, 1],  # event
            censor_surv="km",
        )
        score = ev.concordance_td(self.cnocordance_td_mode)
        return score

    def score_split(self, x, duration, events):
        return self.score(x, np.hstack([duration.reshape((-1, 1)), events.reshape((-1, 1))]))

    def predict(self, X):
        return self.model.predict(X)


class CoxTimeEstimator(BaseEstimator):
    def __init__(
            self,
            label_translator,
            in_features,
            learning_rate=1e-4,
            batch_norm=True,
            dropout=0.0,
            num_nodes=[32, 32],
            batch_size=256,
            epochs=100,
            output_features=10,
            early_stopping_patience=10,
            cnocordance_td_mode="adj_antolini"
    ):
        self.in_features = in_features
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_features = output_features
        self.cnocordance_td_mode = cnocordance_td_mode
        self.label_translator = label_translator
        self._net = self.model = None
        self.early_stopping_patience = early_stopping_patience

    def fit(self, X, y, verbose=False):
        # in_features, num_nodes, batch_norm = True, dropout = None, activation = nn.ReLU,
        # w_init_ = lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')
        self._net = MLPVanillaCoxTime(
            in_features=self.in_features,
            num_nodes=self.num_nodes,
            batch_norm=self.batch_norm,
            dropout=self.dropout)
        x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1)
        self.model = CoxTime(self._net, tt.optim.Adam(self.learning_rate), labtrans=self.label_translator)
        # Sklearn needs the y inputs to be arranged as a matrix with each row
        # corresponding to an example but CoxPH needs a tuple with two arrays
        # y_ = (y[:, 0].astype(np.int64), y[:, 1].astype(np.float32))
        duration_train, events_train = (y_train[:, 0], y_train[:, 1].astype(np.int64))
        y_train = self.label_translator.transform(duration_train, events_train)

        duration_val, events_val = (y_dev[:, 0], y_dev[:, 1].astype(np.int64))
        y_val = self.label_translator.transform(duration_val, events_val)
        callbacks = [tt.callbacks.EarlyStopping(patience=self.early_stopping_patience)]
        self.model.fit(
            x_train,
            y_train,
            self.batch_size,
            self.epochs,
            val_data=(x_dev, y_val),
            verbose=verbose,
            callbacks=callbacks
        )
        return self

    def score(self, X, y):
        _ = self.model.compute_baseline_hazards()
        surv = self.model.predict_surv_df(X)
        ev = EvalSurv(
            surv,
            y[:, 0],  # time to event
            y[:, 1],  # event
            censor_surv="km",
        )
        score = ev.concordance_td(self.cnocordance_td_mode)
        return score

    def score_split(self, x, duration, events):
        return self.score(x, np.hstack([duration.reshape((-1, 1)), events.reshape((-1, 1))]))

    def predict(self, X):
        return self.model.predict(X)


class MLTREstimator(BaseEstimator):
    def __init__(
            self,
            label_translator,
            in_features,
            learning_rate=1e-4,
            batch_norm=True,
            dropout=0.0,
            num_nodes=[32, 32],
            batch_size=256,
            epochs=100,
            output_features=10,
            early_stopping_patience=10,
            cnocordance_td_mode="antolini"
    ):
        self.in_features = in_features
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_features = output_features
        self.cnocordance_td_mode = cnocordance_td_mode
        self.label_translator = label_translator
        self._net = self.model = None
        self.early_stopping_patience = early_stopping_patience

    def fit(self, X, y, verbose=False):
        # in_features, num_nodes, batch_norm = True, dropout = None, activation = nn.ReLU,
        # w_init_ = lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')
        self._net = MLPVanilla(
            in_features=self.in_features,
            num_nodes=self.num_nodes,
            out_features=self.output_features,
            batch_norm=self.batch_norm,
            dropout=self.dropout)
        x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1)
        self.model = MTLR(self._net, tt.optim.Adam(self.learning_rate), duration_index=self.label_translator.cuts)
        # Sklearn needs the y inputs to be arranged as a matrix with each row
        # corresponding to an example but CoxPH needs a tuple with two arrays
        duration_train, events_train = (y_train[:, 0], y_train[:, 1].astype(np.int64))
        y_train = self.label_translator.transform(duration_train, events_train)

        duration_val, events_val = (y_dev[:, 0], y_dev[:, 1].astype(np.int64))
        y_val = self.label_translator.transform(duration_val, events_val)
        callbacks = [tt.callbacks.EarlyStopping(patience=self.early_stopping_patience)]
        self.model.fit(
            x_train,
            y_train,
            self.batch_size,
            self.epochs,
            val_data=(x_dev, y_val),
            verbose=verbose,
            callbacks=callbacks
        )
        return self

    def score(self, X, y):
        surv = self.model.predict_surv_df(X)
        ev = EvalSurv(
            surv,
            y[:, 0],  # time to event
            y[:, 1],  # event
            censor_surv="km",
        )
        score = ev.concordance_td(self.cnocordance_td_mode)
        return score

    def score_split(self, x, duration, events):
        return self.score(x, np.hstack([duration.reshape((-1, 1)), events.reshape((-1, 1))]))

    def predict(self, X):
        return self.model.predict(X)


class DeepHitSingleEstimator(BaseEstimator):
    def __init__(
            self,
            label_translator,
            in_features,
            learning_rate=1e-4,
            batch_norm=True,
            dropout=0.0,
            num_nodes=[32, 32],
            batch_size=256,
            epochs=100,
            output_features=10,
            early_stopping_patience=10,
            cnocordance_td_mode="antolini"
    ):
        self.in_features = in_features
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_features = output_features
        self.cnocordance_td_mode = cnocordance_td_mode
        self.label_translator = label_translator
        self._net = self.model = None
        self.early_stopping_patience = early_stopping_patience

    def fit(self, X, y, verbose=False):
        # in_features, num_nodes, batch_norm = True, dropout = None, activation = nn.ReLU,
        # w_init_ = lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')
        self._net = MLPVanilla(
            in_features=self.in_features,
            num_nodes=self.num_nodes,
            out_features=self.output_features,
            batch_norm=self.batch_norm,
            dropout=self.dropout)
        x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1)
        self.model = MTLR(self._net, tt.optim.Adam(self.learning_rate), duration_index=self.label_translator.cuts)
        # Sklearn needs the y inputs to be arranged as a matrix with each row
        # corresponding to an example but CoxPH needs a tuple with two arrays
        duration_train, events_train = (y_train[:, 0], y_train[:, 1].astype(np.int64))
        y_train = self.label_translator.transform(duration_train, events_train)

        duration_val, events_val = (y_dev[:, 0], y_dev[:, 1].astype(np.int64))
        y_val = self.label_translator.transform(duration_val, events_val)
        callbacks = [tt.callbacks.EarlyStopping(patience=self.early_stopping_patience)]
        self.model.fit(
            x_train,
            y_train,
            self.batch_size,
            self.epochs,
            val_data=(x_dev, y_val),
            verbose=verbose,
            callbacks=callbacks
        )
        return self

    def score(self, X, y):
        surv = self.model.predict_surv_df(X)
        ev = EvalSurv(
            surv,
            y[:, 0],  # time to event
            y[:, 1],  # event
            censor_surv="km",
        )
        score = ev.concordance_td(self.cnocordance_td_mode)
        return score

    def score_split(self, x, duration, events):
        return self.score(x, np.hstack([duration.reshape((-1, 1)), events.reshape((-1, 1))]))

    def predict(self, X):
        return self.model.predict(X)
