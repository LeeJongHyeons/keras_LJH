import numpy as np  
# 데이터 구성
#x = np.array(range(1,101))
#y = np.array(range(1,101))

x = np.array([range(1000), range(3110,4110),range(1000)])
y = np.array([range(5010,6010)])

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
from keras.layers import Dense, BatchNormalization, Dropout  
model = Sequential()
from keras import regularizers

# 레이어의 깊이와 노드의 갯수를 조절

model.add(Dense(10, input_shape =(3,), activation = 'relu', kernel_regularizer=regularizers.l1(0.01)))  # 일반화: regularizer
model.add(Dense(10))   
model.add(Dense(100))  
model.add(Dense(100)) 
model.add(Dropout(0,7))
model.add(Dense(28)) 
model.add(Dense(56)) 
model.add(Dense(38))     
model.add(Dense(87))   
model.add(Dense(2))   
model.add(Dense(50,kernel_regularizer=regularizers.l1(0.12)))  
model.add(Dense(1))

model.save("keras_savetest01.h5")
print("저장 잘 했다.")
