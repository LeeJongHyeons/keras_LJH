from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]]) 
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 
# =================================  StandardScaler, MinMaxScaler ==================================================
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#scaler = StandardScaler() 
scaler = MinMaxScaler()
scaler.fit(x) # fit은 한번만 사용
x = scaler.transform(x)
print(x) 
# StandardScaler => 최대값: 2.5468398, 최소값: -0.89648761
# MinMaxScaler => 최대값: 0.8245614, 최소값: 0.01754386
# ==================================================================================================================

# 기본 스케일, 평균과 표준편차를 사용
# 평균을 제거하고 데이터를 단위 분산을 조정하여 이상치가 있다면, 평균과 표준편차에 영향을 미쳐서 변환된 데이터의 확산은 매우 달라짐
# from sklearn.preprocessing import StandardScaler
# standardScaler = StandardScaler()
# print(standardScaler.fit(train_data))
# train_data_standardScaled = standardScaler.transform(train_data)

# 최대/최소값이 각각 1, 0이 되도록 스케일링
# 모든 feature 값이 0~1 사이에 있도록 데이터를 재조정, 이상치 있는 경우 변환된 값이 매우 좁은 범위로 압축
# from sklearn.preprocessing import MinMaxScaler
# minMaxScaler = MinMaxScaler()
# print(minMaxScaler.fit(train_data))
# train_data_minMaxScaled = minMaxScaler.transform(train_data)
