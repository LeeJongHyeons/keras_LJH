import numpy as np  
# 데이터 구성
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 10행 1열
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 10행 1열
x_test = np.array([11,12,13,14,15,16,17,18,19,20]) # 10행 1열
y_test = np.array([11,12,13,14,15,16,17,18,19,20]) # 10행 1열
x3 = np.array([101,102,103,104,105,106])  # 6행 1열
x2 = np.array([22,33,44,55,66,77,88,99,110,122,133,144,155,166,177]) # 15행 1열

# input_dim = 1 컬럼의 갯수: 행과는 상관없이 열만 맞음 


# 모델 구성 
from keras.models import Sequential 
from keras.layers import Dense  
model = Sequential()

# 레이어의 깊이와 노드의 갯수를 조절
#model.add(Dense(5, input_dim =1, activation = 'relu')) 
model.add(Dense(5, input_shape =(1,), activation = 'relu')) # input_shape: 컬럼이 1개
model.add(Dense(3))   
model.add(Dense(4))   
model.add(Dense(1))
import numpy as np  
# 데이터 구성
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 10행 1열
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 10행 1열
x_test = np.array([11,12,13,14,15,16,17,18,19,20]) # 10행 1열
y_test = np.array([11,12,13,14,15,16,17,18,19,20]) # 10행 1열
x3 = np.array([101,102,103,104,105,106])  # 6행 1열
x2 = np.array([22,33,44,55,66,77,88,99,110,122,133,144,155,166,177]) # 15행 1열

# input_dim = 1 컬럼의 갯수: 행과는 상관없이 열만 맞음 


# 모델 구성 
from keras.models import Sequential 
from keras.layers import Dense  
model = Sequential()

# 레이어의 깊이와 노드의 갯수를 조절
#model.add(Dense(5, input_dim =1, activation = 'relu')) 
model.add(Dense(5, input_shape =(1,), activation = 'relu')) # input_shape: 컬럼이 1개
model.add(Dense(3))   
model.add(Dense(4))   
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(11))
model.add(Dense(1))

   
#model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(x,y, epochs=100, batch_size=3) # epochs=100: 모델링을 100번 돌림
model.fit(x_train, y_train, epochs=100)

# 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print("acc :", acc)


y_predict = model.predict(x2)
print(y_predict)

   
#model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(x,y, epochs=100, batch_size=3) # epochs=100: 모델링을 100번 돌림
model.fit(x_train, y_train, epochs=100)

# 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print("acc :", acc)


y_predict = model.predict(x_test)
print(y_predict)

# RMSE(루트 평균 제곱 계산) 구하기

from sklearn.metrics import mean_squared_error  
def RMSE(y_test, y_predict): # 원래값, 예측값
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))