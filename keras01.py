import numpy as np  
# 데이터 구성
x = np.array([1,2,3])
y = np.array([1,2,3])
# 모델 구성 
from keras.models import Sequential 
from keras.layers import Dense  
model = Sequential()

# 레이어의 깊이와 노드의 갯수를 조절
model.add(Dense(5, input_dim =1, activation = 'relu')) 
model.add(Dense(3))   
model.add(Dense(4))   
model.add(Dense(1))   

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, epochs=100, batch_size=1)  # epochs=100: 모델링을 100번 돌림

# 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("acc :", acc)

