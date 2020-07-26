from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

total_data = pd.read_csv(".\\주가데이터\\삼성전자 주가추이(1년).csv", encoding = "UTF-8-SIG")
cols = ['Open', 'High', 'Low', 'Close']
predict = ['Close']

# MinMaxScaler를 사용한 데이터 정규화
scaler = MinMaxScaler()
normed_cols = scaler.fit_transform(total_data[cols])
normed_data = pd.DataFrame(normed_cols) # 데이터프레임 생성(원본 객체 변경 X)
normed_data.columns = cols

# 데이터셋 생성
train_data = normed_data[0:274]
test_data = normed_data[274:]

feature = train_data[cols]
label = test_data[predict]

cols_data = []
predict_data = []

for i in range(len(feature)-31) :
    cols_data.append(np.array(feature.iloc[i:i+7]))
    predict_data(np.array(label.iloc[i+7]))

feature = np.array(cols_data)
label = np.array(predict_data)

X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size = 25)
print(X_train)
"""
X_train = np.load(open('.\\word_dir\\input.npy', 'rb')) # 훈련 데이터 입력값
Y_train = np.load(open('.\\word_dir\\label.npy', 'rb')) # 훈련 데이터 레이블

X_test = np.load(open('.\\word_dir\\test_data.npy', 'rb')) # 테스트 데이터 입력값
Y_test = np.load(open('.\\word_dir\\test_label.npy', 'rb')) # 테스트 데이터 레이블

data_configs = json.load(open('.\\word_dir\\data_configs.json', 'r', -1, "UTF-8-SIG"))
total_words_len = data_configs['vocab_size']

##################################################
model = Sequential()
#model.add(Embedding(total_words_len, 128)) # 임베딩 벡터 차원 128
model.add(LSTM(16))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 시그모이드 함수 활성화

ckpoint = ModelCheckpoint('stock_model.h5', monitor = 'val_accuracy', mode = 'max',verbose = 1, save_best_only = True)

model.compile(loss = 'mean_squared_error', optimizer= 'adam', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size = 64, epochs = 5, validation_split=0.25, callbacks = ckpoint)

print('가장 높은 모델의 테스트 정확도 : %.4f' % (load_model('stock_model.h5').evaluate(X_test, Y_test)[1]))

#### 그래프 그리기 ####
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.legend(['accuracy', 'loss'], loc = 'upper left')

plt.show()
"""