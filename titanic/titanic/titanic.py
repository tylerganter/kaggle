#!/usr/bin/env python
'''

'''
from abc import ABCMeta, abstractmethod
from six import string_types

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from tensorflow.keras import models, layers

preds_template_path = 'predictions/{}.csv'


class _TitanicModel(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        #   output: confidences, predictions
        #   accuracy
        #   ROC curve
        #   AUC
        pass


class SexModel(_TitanicModel):
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        predictions = (X['Sex'] == 'female').astype(int)
        predictions.name = 'Survived'
        return predictions

    def evaluate(self, X, y):
        preds = self.predict(X)
        is_correct = preds == y
        accuracy = np.sum(is_correct) / len(is_correct)
        return accuracy


class SklearnRFCModel(_TitanicModel):
    
    def __init__(self):
        self._model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=1)
    
    def fit(self, X, y):
        return self._model.fit(self._transform_data(X), y)
    
    def predict(self, X):
        preds = self._model.predict(self._transform_data(X))
        return pd.Series(preds, index=X.index, name='Survived')

    def evaluate(self, X, y):
        return self._model.score(self._transform_data(X), y)

    def _transform_data(self, X):
        features = ["Pclass", "Sex", "SibSp", "Parch"]
        return pd.get_dummies(X[features])


class SklearnModel(_TitanicModel):
    
    def __init__(self):
        self._model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            criterion='gini',
            bootstrap=True,
            random_state=1
        )
        
        # self._model = LinearSVC(
        #     max_iter=10000
        # )
        
        # self._model = SVC(gamma='scale', max_iter=10000)

        train_path = 'data/train.csv'
        train_data = pd.read_csv(train_path, index_col=0)

        train_data = self._transform_data(train_data, initializing=True)

        self._mean = np.mean(train_data, axis=0)
        self._std = np.std(train_data, axis=0)
    
    def fit(self, X, y):
        return self._model.fit(self._normalize(self._transform_data(X)), y)
    
    def predict(self, X):
        preds = self._model.predict(self._normalize(self._transform_data(X)))
        return pd.Series(preds, index=X.index, name='Survived')

    def evaluate(self, X, y):
        return self._model.score(self._normalize(self._transform_data(X)), y)

    def _transform_data(self, X, initializing=False):
        # return pd.get_dummies(X[features])

        features = [
            'Pclass', 'Sex', 'SibSp', 'Parch', 'Age', 'Fare', 'Embarked',
            'Cabin'
        ]

        X2 = X[features].copy()

        # sex
        X2['Sex'] = (X2['Sex'] == 'female').astype(int)

        # cabin count
        X2['Cabin Count'] = pd.Series(
            (len(x.split(' ')) if isinstance(x, string_types)
             else np.nan for x in X2['Cabin']),
            index=X2.index
        )

        # # cabin letters part 1
        # X2['cabin letter'] = pd.Series(
        #     (x[0] if isinstance(x, string_types) else x for x in X2['Cabin']),
        #     index=X2.index,
        #     name='cabin letter'
        # )

        del X2['Cabin']

        X2 = pd.get_dummies(X2)

        # cabin letters part 2
        # if initializing:
        #     self._cabin_cols = [x for x in X2.columns if x.startswith('cabin letter')]
        # else:
        #     for cabin_col in self._cabin_cols:
        #         if cabin_col not in X2:
        #             X2[cabin_col] = pd.Series(
        #                 np.zeros((X2.shape[0])),
        #                 index=X2.index,
        #                 name=cabin_col
        #             )

        X2 = X2.reindex(sorted(X2.columns), axis=1)

        return X2

    def _normalize(self, X):
        for feature in X.columns:
            X.loc[pd.isna(X[feature]), feature] = self._mean[feature]

        X -= self._mean
        X /= self._std

        return X


class KerasModel(_TitanicModel):

    def __init__(self):
        self._model = models.Sequential()
        self._model.add(layers.Dense(16, activation="relu", input_shape=(10,)))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(1, activation="sigmoid"))

        self._model.compile(
            optimizer="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        train_path = 'data/train.csv'
        train_data = pd.read_csv(train_path, index_col=0)

        train_data = self._transform_data(train_data, initializing=True)

        self._mean = np.mean(train_data, axis=0)
        self._std = np.std(train_data, axis=0)

    def fit(self, X, y):
        return self._model.fit(
            self._normalize(self._transform_data(X)), y,
            epochs=20,
            batch_size=64,
            verbose=0
        )

    def predict(self, X):
        preds = self._model.predict(self._normalize(self._transform_data(X)))
        preds = preds.flatten()
        return pd.Series(preds, index=X.index, name='Survived')

    def evaluate(self, X, y):
        loss, accuracy = self._model.evaluate(
            self._normalize(self._transform_data(X)), y)
        return accuracy

    def _transform_data(self, X, initializing=False):
        # return pd.get_dummies(X[features])

        features = [
            'Pclass', 'Sex', 'SibSp', 'Parch', 'Age', 'Fare', 'Embarked',
            'Cabin'
        ]

        X2 = X[features].copy()

        # sex
        X2['Sex'] = (X2['Sex'] == 'female').astype(int)

        # cabin count
        X2['Cabin Count'] = pd.Series(
            (len(x.split(' ')) if isinstance(x, string_types)
             else np.nan for x in X2['Cabin']),
            index=X2.index
        )

        # # cabin letters part 1
        # X2['cabin letter'] = pd.Series(
        #     (x[0] if isinstance(x, string_types) else x for x in X2['Cabin']),
        #     index=X2.index,
        #     name='cabin letter'
        # )

        del X2['Cabin']

        X2 = pd.get_dummies(X2)

        # cabin letters part 2
        # if initializing:
        #     self._cabin_cols = [x for x in X2.columns if x.startswith('cabin letter')]
        # else:
        #     for cabin_col in self._cabin_cols:
        #         if cabin_col not in X2:
        #             X2[cabin_col] = pd.Series(
        #                 np.zeros((X2.shape[0])),
        #                 index=X2.index,
        #                 name=cabin_col
        #             )

        X2 = X2.reindex(sorted(X2.columns), axis=1)

        return X2

    def _normalize(self, X):
        for feature in X.columns:
            X.loc[pd.isna(X[feature]), feature] = self._mean[feature]

        X -= self._mean
        X /= self._std

        return X


def split_data(train_data, num_folds, fold=0):
    np.random.seed(1)

    # randomly shuffle rows
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data.copy().iloc[indices, :]

    num_val_samples = len(train_data) // num_folds
    
    val_X = train_data.iloc[
        fold * num_val_samples:(fold + 1) * num_val_samples].copy()
    partial_train_X = pd.concat(
        [
            train_data.iloc[:fold * num_val_samples].copy(),
            train_data.iloc[(fold + 1) * num_val_samples:].copy()
        ],
        axis=0
    )

    partial_train_y = partial_train_X.pop('Survived')
    val_y = val_X.pop('Survived')
    
    return partial_train_X, val_X, partial_train_y, val_y


def train_and_evaluate(train_data, model_class, num_folds):
    accuracies = []

    for fold in range(num_folds):
        # split train/val
        partial_train_X, val_X, partial_train_y, val_y = \
            split_data(train_data, num_folds=num_folds, fold=fold)
        
        # create model
        model = model_class()
        
        # train
        model.fit(partial_train_X, partial_train_y)
        
        # evaluate
        accuracies.append(model.evaluate(val_X, val_y))

    X = train_data.copy()
    y = X.pop('Survived')
    model = model_class()
    model.fit(X, y)

    return np.mean(accuracies), model


def process_model(train_data, test_data, model_class, num_folds=5):
    model_name = model_class.__name__
    accuracy, model = train_and_evaluate(
        train_data, model_class=model_class, num_folds=num_folds)
    print('{} accuracy: {:.3f}'.format(model_name, accuracy))
    preds = pd.DataFrame(model.predict(test_data))
    preds.to_csv(preds_template_path.format(model_name))
