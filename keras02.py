import numpy as np  
# 데이터 구성
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([4,5,6])
# 모델 구성 
from keras.models import Sequential 
from keras.layers import Dense  
model = Sequential()

# 레이어의 깊이와 노드의 갯수를 조절
model.add(Dense(5, input_dim =1, activation = 'relu')) 
model.add(Dense(16))   
model.add(Dense(32))   
model.add(Dense(22))   
model.add(Dense(50))  
model.add(Dense(10))  
model.add(Dense(1))  

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x,y, epochs=100, batch_size=3) # epochs=100: 모델링을 100번 돌림
model.fit(x, y, epochs=100)

# 평가 예측
loss, acc = model.evaluate(x, y, batch_size=3)
print("acc :", acc)


y_predict = model.predict(x2)
print(y_predict)