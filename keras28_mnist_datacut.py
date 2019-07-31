from keras.datasets import mnist # mnist를 dataset에 다운받아, 6만개의 데이터를 받아옴
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping  

import matplotlib.pyplot as plt 
import numpy
import os 
import tensorflow as tf  

# 데이터 불러오기
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
X_train = X_train[:300]
X_test = X_test[:300]
Y_train = Y_train[:300]
Y_test = Y_test[:300]

X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')/255

print(Y_train.shape) # (60000,)
print(Y_test.shape)  # (10000,)
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape)  # (60000, 10)
print(Y_test.shape)   # (10000, 10)

print(X_train.shape) #(60000, 28, 28, 1) 
print(X_test.shape)  #(10000, 28, 28, 1)

# OneHotEncoding: 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 
# X_train = 7, 3, 5, 6 ...
# 7: 0 0 0 0 0 0 0 1 0 0
# 3: 0 0 0 1 0 0 0 0 0 0
# 5: 0 0 0 0 0 1 0 0 0 0
# 6: 0 0 0 0 0 0 1 0 0 0

# 컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.24))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 최적화 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10000)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=300, batch_size=25, verbose=1, callbacks=[early_stopping_callback])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))