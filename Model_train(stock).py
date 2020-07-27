import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model

from Model_definition import Stock_Model

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
# 구현중

######################################################################################################

os.makedirs('stock_model', exist_ok= True)
ckpoint = ModelCheckpoint('.\\stock_model\\model.h5', monitor = 'val_accuracy', mode = 'max',verbose = 1, save_best_only = True)

model = Stock_Model()

model.compile(loss = 'mean_square_error', optimizer= 'adam', metrics=['accuracy'])
model.summary()

hist = model.fit(X_train, Y_train, batch_size = 512, epochs = 10, validation_split=0.25, callbacks = ckpoint)

print('가장 높은 모델의 테스트 정확도 : %.4f' % (load_model('.\\stock_model\\model.h5').evaluate(X_test, Y_test)[1]))

#### 그래프 그리기 ####
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.legend(['accuracy', 'loss'], loc = 'upper left')

plt.show()