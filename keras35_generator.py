from keras.preprocessing.image import ImageDataGenerator 
from keras.datasets import mnist # mnist를 dataset에 다운받아, 6만개의 데이터를 받아옴
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping  

import matplotlib.pyplot as plt 
import numpy
import os 
import tensorflow as tf  

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

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

data_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.02, height_shift_range=0.02, horizontal_flip=True)
                                    # 회전값(rotation), 넓이(width), 높이(height), 수평(horizontal_flip)
model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=600), #generator
steps_per_epoch=len(X_train)//32,epochs=300,validation_data=(X_test, Y_test), verbose=1) #, callbacks=callbacks
# X_train을 32로 나눔, 200번돌리고,X_test, Y_test에 검증, 실행하기전에 이미지를 만든뒤 실행

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))