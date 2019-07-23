import numpy as np  
# 데이터 구성
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

# 모델 구성 
from keras.models import Sequential 
from keras.layers import Dense  
model = Sequential()

# 레이어의 깊이와 노드의 갯수를 조절
model.add(Dense(5, input_dim =1, activation = 'relu')) 
model.add(Dense(3))   
model.add(Dense(4))   
model.add(Dense(2))
model.add(Dense(1))
   


#model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(x,y, epochs=100, batch_size=3) # epochs=100: 모델링을 100번 돌림
model.fit(x_train, y_train, epochs=100)

# 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=4)
print("acc :", acc)


y_predict = model.predict(x_test)
print(y_predict)