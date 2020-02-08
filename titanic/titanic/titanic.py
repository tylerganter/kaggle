#!/usr/bin/env python
'''

'''
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


preds_template_path = 'predictions/{}.csv'


class _TitanicModel(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y):
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

    def score(self, X, y):
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

    def score(self, X, y):
        return self._model.score(self._transform_data(X), y)

    def _transform_data(self, X):
        features = ["Pclass", "Sex", "SibSp", "Parch"]
        return pd.get_dummies(X[features])


def split_data(train_data, num_folds, fold=0):
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
        accuracies.append(model.score(val_X, val_y))

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
