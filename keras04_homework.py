# 모델의 깊이, 노드의 개수,  epochs, batch_size
# (x_train = 1~100, y_train= 501~600) (x-test: 1001~1100, y_test:1101~1200)

import numpy as np 
from keras.models import Sequential 
from keras.layers import Dense  

x_train = []
y_train = []
x_test = []
y_test = []

x_train

model = Sequential()
model.add(Dense(5, input_dim =1, activation = 'relu')) 
model.add(Dense(15))   
model.add(Dense(30))   
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(40))

model.add(Dense)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#model.fit(x,y, epochs=100, batch_size=3) # epochs=100: 모델링을 100번 돌림
model.fit(x_train, y_train, epochs=100)

# 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=4)
print("acc :", acc)


y_predict = model.predict(x_test)
print(y_predict)