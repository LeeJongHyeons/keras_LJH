import numpy as np 
import keras 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,BatchNormalization

a = np.array(range(1,101)) #x_train: 1~100
batch_size = 1
size = 5
size = 5
def split_5(seq, size):  
    aaa =[]
    for i in range(len(a)-size+1):
        subset = a[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, size)
print("===============================")
print(dataset)
print(dataset.shape)

x_train = dataset[:, 0:4]
y_train = dataset[:, 4] 

x_train = np.reshape(x_train, (len(x_train), size-1, 1))

x_test = x_train + 100 
y_test = y_train + 100 

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_test[0])

#2. 모델구성
model = Sequential()
model.add(LSTM(16, batch_input_shape=(batch_size,4,1),stateful=True)) 
model.add(Dense(5))
model.add(Dense(9))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

num_epochs = 50

import keras
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='mse', patience=3, mode='auto') 

his_mse = []
his_val_mse = []

for epoch_idx in range(num_epochs):
    print('epochs:' + str(epoch_idx))
    history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=False, validation_data=(x_test, y_test), callbacks=[early_stopping,tb_hist]) 
    his_mse.append(history.history['mean_squared_error'])
    his_val_mse.append(history.history['val_mean_squared_error'])

    model.reset_states() 

mse, _ = model.evaluate(x_train, y_train, batch_size=batch_size)
print("mse :", mse)
model.reset_states()

y_predict = model.predict(x_test, batch_size=batch_size)

print(y_predict[0:10])

from sklearn.metrics import mean_squared_error  
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)

#mse : 0.5883747592991616    mse : 0.9462501629588284           
#RMSE : 58.73912906189553    RMSE : 50.25731585638838
#R2 : -3.493046488920265     R2 : -2.2891561112387895
#[[ 99.50634 ]               [[101.8643 ]
# [100.31115 ]                [107.72613]
# [100.34896 ]                [109.15685]
# [100.37209 ]                [109.50226]
# [100.39368 ]                [109.62988]
# [100.413994]                [109.71341]
# [100.43312 ]                [109.78643]
# [100.451126]                [109.85566]
# [100.46807 ]                [109.90283]
# [100.484024]]               [109.93777]]
