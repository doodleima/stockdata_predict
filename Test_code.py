import pandas as pd
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

look_back = 1


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# 파일 불러오기
df = pd.read_csv(".\\주가데이터\\삼성전자 주가추이.csv", encoding = 'UTF-8-SIG', index_col="Date")

# nparray 변환
nparr = df['Close'].values[::-1]
nparr.astype('float32')
print(nparr)

# 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
nptf = scaler.fit_transform(nparr)

# 훈련/테스트 데이터 스플릿
train_size = int(len(nptf) * 0.9)
test_size = len(nptf) - train_size
train, test = nptf[0:train_size], nptf[train_size:len(nptf)]
print(len(train), len(test))

# 데이터셋 생성
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 입력값 reshape
# [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# 예측값 생성
testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Train Score: %.2f RMSE' % testScore)

# 가장 마지막 하루에 대한 값 예측
lastX = nptf[-1]
lastX = np.reshape(lastX, (1, 1, 1))
lastY = model.predict(lastX)
lastY = scaler.inverse_transform(lastY)
print('Predict the Close value of final day: %d' % lastY)  # 데이터 입력 마지막 다음날 종가 예측

# plot
plt.plot(testPredict)
plt.plot(testY)
plt.show()

"""
df1 = pd.read_csv("C:\\Users\\Lim\\Desktop\\7월 27일 크롤링\\SK하이닉스 기사 헤드라인.csv", encoding ='UTF-8-SIG')
df2 = pd.read_csv("C:\\Users\\Lim\\Desktop\\7월 27일 크롤링\\NAVER 기사 헤드라인.csv", encoding ='UTF-8-SIG')
df3 = pd.read_csv("C:\\Users\\Lim\\Desktop\\7월 27일 크롤링\\훈련 데이터(헤드라인).csv", encoding ='UTF-8-SIG')

#print(df1.isnull().sum()) # 각 컬럼별 결측치 갯수 합 확인
#print(df2.isnull().sum())
#print(df2.head(10))

total_df = [df1, df2, df3]

for i in total_df :
    del i['Unnamed: 0']
    #print(i.tail())
    i.dropna(inplace = True) # 모든 컬럼에 대해 결측치가 있는 행을 삭제

total_df = pd.concat(total_df, axis = 0)
total_df.reset_index(drop = True, inplace = True)
#df1.to_csv("SK하이닉스.csv", encoding = 'UTF-8-SIG')
#df2.to_csv("NAVER.csv", encoding = 'UTF-8-SIG')

#print(total_df.tail(10))

total_df.to_csv(".\\훈련데이터\\훈련데이터(헤드라인종합).csv", encoding = 'UTF-8-SIG')
"""
"""
df1 = pd.read_csv("훈련데이터\\삼성전자 기사 헤드라인.csv", encoding ='UTF-8-SIG')
df2 = pd.read_csv("훈련데이터\\LG전자 기사 헤드라인.csv", encoding ='UTF-8-SIG')
df3 = pd.read_csv("훈련데이터\\SK하이닉스 기사 헤드라인.csv", encoding ='UTF-8-SIG')
df4 = pd.read_csv("훈련데이터\\현대자동차 기사 헤드라인.csv", encoding ='UTF-8-SIG')
df5 = pd.read_csv("훈련데이터\\NAVER 기사 헤드라인.csv", encoding ='UTF-8-SIG')

df_list = [df1, df2, df3, df4, df5]

for i in df_list :
    del i['Unnamed: 0']

total_df = pd.concat(df_list, axis = 0)
total_df.drop_duplicates(subset = ['헤드라인'], inplace = True)
total_df.reset_index(drop = True, inplace = True)

total_df.to_csv(".\\훈련데이터\\훈련 데이터(헤드라인).csv", encoding = 'UTF-8-SIG')
"""