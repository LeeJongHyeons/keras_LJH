import numpy as np  
# 데이터 구성
#x = np.array(range(1,101))
#y = np.array(range(1,101))

x = np.array([range(100), range(311,411),range(100)])
y = np.array([range(501,601)])

# 행렬 변환
x = np.transpose(x) 
y = np.transpose(y)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=66, test_size=0.4
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test,y_test, random_state=66, test_size=0.5
)

print(x_test.shape)


# 모델 구성 
from keras.models import Sequential 
from keras.layers import Dense  
model = Sequential()

# 레이어의 깊이와 노드의 갯수를 조절
#model.add(Dense(5, input_dim =1, activation = 'relu')) 
model.add(Dense(5, input_shape =(3,), activation = 'relu'))
model.add(Dense(3))   
model.add(Dense(4))   
model.add(Dense(1))

#model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
#model.fit(x,y, epochs=100, batch_size=3) # epochs=100: 모델링을 100번 돌림
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

# 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print("acc :", acc)

y_predict = model.predict(x_test)
print(y_predict)

#model.summary()

# 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
#model.fit(x,y, epochs=100, batch_size=3) # epochs=100: 모델링을 100번 돌림
model.fit(x_train, y_train, epochs=100)

# 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc :", acc)


y_predict = model.predict(x_test)
print(y_predict)

# RMSE(루트 평균 제곱 계산) 구하기

from sklearn.metrics import mean_squared_error  
def RMSE(y_test, y_predict): # 원래값, 예측값
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)