from keras.models import Sequential

filter_size = 7     # 자른 데이터 추가
kernel_size = (2,2) # 이미지를 몇개씩 잘라서 작업
model = Sequential()

# padding: Convolution 레이어의 출력 데이터 크기를 조정할 목적으로 사용
# 입력 데이터의 외각에 지정된 픽셀만큼, 특정 값으로 채워넣어 보통 패딩값을 0으로 채움 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(7, (2,2), padding='same', input_shape=(5,5,1))) #Dense: Conv2D, output(filter_size, kernel_size)
                                         # input_shape=(28,28,1): 1은 흑백, 3은 칼라
model.add(Conv2D(16,(2,2))) # CNN: 데이터를 쌓은후에 더 좋은 속성값을 추출하여 판단
model.add(MaxPooling2D(2,2)) # 풀링에 있는 최대 몇개를 자를건지 제시(중복값 없이)
model.add(Flatten(2,2))
model.add(Dense(2,2))
model.add(Conv2D(8,(2,2)))

model.summary()