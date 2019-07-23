import numpy as np  

x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 10행 1열
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 10행 1열
x_test = np.array([11,12,13,14,15,16,17,18,19,20]) # 10행 1열
y_test = np.array([11,12,13,14,15,16,17,18,19,20]) # 10행 1열
x4 = np.array([101,102,103,104,105,106])  # 6행 1열

from keras.models import Sequential 
from keras.layers import Dense  
model = Sequential()

model.add(Dense(5, input_shape =(1,), activation = 'relu')) # input_shape: 컬럼이 1개
model.add(Dense(3))   
model.add(Dense(4))   
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(x,y, epochs=100, batch_size=3) # epochs=100: 모델링을 100번 돌림
model.fit(x_train, y_train, epochs=100)

# 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print("acc :", acc)


y_predict = model.predict(x4)
print(y_predict)