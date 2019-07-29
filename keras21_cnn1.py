from keras.models import Sequential

filter_size = 7     # 자른 데이터 추가
kernel_size = (2,2) # 이미지를 몇개씩 잘라서 작업
model = Sequential()

from keras.layers import Conv2D 
model = Sequential()
model.add(Conv2D(filter_size, kernel_size, input_shape=(5,5,1))) #Dense: Conv2D, output(filter_size, kernel_size)
                                         # input_shape=(28,28,1): 1은 흑백, 3은 칼라
model.add(Conv2D(16,(2,2))) # CNN: 데이터를 쌓은후에 더 좋은 속성값을 추출하여 판단
model.add(Conv2D(8,(2,2)))

model.summary()

#filter_size = 32     # 자른 데이터를 32장 추가
#kernel_size = (3,3)  (3X3)*3*3 데이터가 늘어남
#input_shape=(28,28,1) (3*3+1(Bios)*32 = 320)
# =======================================================
#  conv2d_1 (Conv2D)           (None, 26,26,32)    320
# =======================================================
#  conv2d_2 (Conv2D)          (None, 24,24,16)    4624
# =======================================================
#  conv2d_3 (Conv2D)          (None, 23,23,8)      520
# =======================================================
# Total params: 5,464
# Trainable params: 5,464
# Non-trainable params: 0

#filter_size = 7     # 자른 데이터를 7개 추가
#kernel_size = (2,2) # 2개씩 잘라서 데이터를 작업 (2,2)*4*4
#input_shape=(5,5,1) # 가로 세로 (5,5), 1은 흑백
# (None, 4, 4, 7): 7장을 만들어, (5,5,1)을 (2,2)를 잘라서 7장을 만듬
# 파라미터 개수 35는 2*2+1(Bios)*7=35
#=================================================================
#conv2d_1 (Conv2D)            (None, 4, 4, 7)           35
#=================================================================
#conv2d_2 (Conv2D)            (None, 3, 3, 16)          464
#=================================================================
#conv2d_2 (Conv2D)            (None, 2, 2, 8)           520
#=================================================================
#Total params: 1,019
#Trainable params: 1,019
#Non-trainable params: 0


