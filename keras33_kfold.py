from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout 
import numpy as np
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

from keras import models
from keras import layers

k = 5
num_val_samples = len(train_data)//k
num_epochs = 100
all_scores = []

def build_model():
    # 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(14, activation='relu'))
    model.add(layers.Dense(5, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(26, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) #mae(Mean Absolute Error)
    return model

#from sklearn.model_selection import StratifiedKFold

seed = 77
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
model = KerasRegressor(build_fn=build_model, epochs=10, 
                       batch_size=1, verbose=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(model, train_data, train_targets, cv=kfold)

import numpy as np 
print(results)
print(np.mean(results))

