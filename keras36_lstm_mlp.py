import numpy as np 
from keras.models import Sequential 
from keras.layers import Dense, LSTM 

x = np.array(range(1,101))

size = 5
def split_5(seq, size):
    aaa = []
    for i in range(len(x)-size+1):
        subset = x[i:(i+size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

dataset =split_5(x, size)
print("=============================")
print(dataset)

x_train = dataset[:, 0:4]
y_train = dataset[:,4]

print(x_train.shape)  
print(y_train.shape)  

x_train = np.reshape(x_train, (len(x)-size+1,4,1))

print(x_train.shape) 

x_test = np.array([[[11],[12],[13],[14]],[[12],[13],[14],[15]],[[13],[14],[15],[16]],[[14],[15],[16],[17]]])
y_test = np.array([15,16,17,18])

print(x_test.shape)
print(y_test.shape)

# 모델구성
model = Sequential()
model.add(LSTM(32, input_shape=(4,1), return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10)) 

model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print('loss :', loss)
print('acc :', acc)
print('y_predict(x_test) : \n', y_predict)

from sklearn.metrics import mean_squared_error  
def RMSE(y_test, y_predict): # 원래값, 예측값
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)
print("loss :", loss)