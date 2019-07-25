from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12]]) # x.shape(4행.3열) # input_shape: 행을 무시, (3,1)
y = array([4,5,6,7,8,9,10,11,12,13]) # y.shape(1행.4열)
                                            
print("x.shape:", x.shape)
print("y.shape:", y.shape)

x = x.reshape((x.shape[0], x.shape[1], 1))  # 결과값의 개수: 3연 1열, 행 무시
print("x.shape:", x.shape)

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(Dense(5))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(1))

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1)

x_input = array([25,35,45]) 
x_input =x_input.reshape((1,3,1)) # 3행 1개씩 

yhat = model.predict(x_input)
print(yhat)

# model.summary()







