####  LSTM
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

import json
import numpy as np
import matplotlib.pyplot as plt

X_train = np.load(open('.\\word_dir\\input.npy', 'rb')) # 훈련 데이터 입력값
Y_train = np.load(open('.\\word_dir\\label.npy', 'rb')) # 훈련 데이터 레이블

X_test = np.load(open('.\\word_dir\\test_data.npy', 'rb')) # 테스트 데이터 입력값
Y_test = np.load(open('.\\word_dir\\test_label.npy', 'rb')) # 테스트 데이터 레이블

data_configs = json.load(open('.\\word_dir\\data_configs.json', 'r', -1, "UTF-8-SIG"))
total_words_len = data_configs['vocab_size']

model = Sequential()
model.add(Embedding(total_words_len, 128)) # 임베딩 벡터 차원 128
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 시그모이드 함수 활성화

ckpoint = ModelCheckpoint('model.h5', monitor = 'val_accuracy', mode = 'max',verbose = 1, save_best_only = True)

model.compile(loss = 'binary_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size = 64, epochs = 5, validation_split=0.25, callbacks = ckpoint)

print('가장 높은 모델의 테스트 정확도 : %.4f' % (load_model('model.h5').evaluate(X_test, Y_test)[1]))

#### 그래프 그리기 ####
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.legend(['accuracy', 'loss'], loc = 'upper left')

plt.show()