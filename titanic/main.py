#!/usr/bin/env python
'''
Titanic: Machine Learning from Disaster

www.kaggle.com/c/titanic
'''
import os
import sys
sys.path.append(os.path.abspath("./titanic"))

import pandas as pd

from titanic import process_model, SexModel, SklearnRFCModel, SklearnModel, KerasModel

train_path = 'data/train.csv'
test_path = 'data/test.csv'

# read in the data
train_data = pd.read_csv(train_path, index_col=0)
test_data = pd.read_csv(test_path, index_col=0)

# train, evaluate and write out predictions
print("BEST YET: 0.830")
process_model(train_data, test_data, SexModel)
# process_model(train_data, test_data, SklearnRFCModel)
process_model(train_data, test_data, SklearnModel)
# process_model(train_data, test_data, KerasModel)
