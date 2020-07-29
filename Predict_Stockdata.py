import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Model_definition import Stock_Model, dataset

scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
feature_cols = ['Open', 'High', 'Low', 'Volume']
label_cols = ['Close']

df = pd.read_csv(".\\주가데이터\\삼성전자 주가추이.csv", encoding = "UTF-8-SIG")
#df = pd.read_csv(".\\주가데이터\\" + str(jongmok) + " 주가추이.csv", encoding = "UTF-8-SIG")

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

SIZE = int(len(df_scaled) * 0.75) #int(len(normed_data) * 0.75)

train = df_scaled[:SIZE]
test = df_scaled[SIZE:]

train_feature = train[feature_cols]
train_label = train[label_cols]

test_feature = test[feature_cols]
test_label = test[label_cols]

# 훈련 데이터셋과 테스트 데이터셋
train_feature, train_label = dataset(train_feature, train_label, 31)
test_feature, test_label = dataset(test_feature, test_label, 7)

# 훈련셋과 검증셋 스플릿
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

erstoping = EarlyStopping(monitor='val_loss', patience=5)
ckpoint = ModelCheckpoint('.\\stock_model\\model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

model = Stock_Model(train_feature)

model.compile(loss = 'mean_squared_error', optimizer = 'adam')
#model.summary()

hist = model.fit(x_train, y_train, epochs=64, batch_size=16, validation_data=(x_valid, y_valid), callbacks=[erstoping, ckpoint])
predict = model.predict(test_feature)

plt.figure(figsize=(9, 5))
plt.plot(test_label, label='original')
plt.plot(predict, label='predict')
plt.legend()
plt.show()