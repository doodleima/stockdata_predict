#### CNN ####
# i7-8750H / RTX 2060(8GB)

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from Model_definition import NLP_Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# 훈련 데이터
X_train = np.load(open('..\\word_dir\\input.npy', 'rb'))
Y_train = np.load(open('..\\word_dir\\label.npy', 'rb'))

# 테스트 데이터
X_test = np.load(open('..\\word_dir\\test_data.npy', 'rb'))
Y_test = np.load(open('..\\word_dir\\test_label.npy', 'rb'))

data_configs = json.load(open('..\\word_dir\\data_configs.json', 'r', -1, "UTF-8-SIG"))
total_words_len = data_configs['vocab_size']

os.makedirs('model', exist_ok= True)
ckpoint = ModelCheckpoint('../model/model.h5', monitor ='val_accuracy', mode ='max', verbose = 1, save_best_only = True)
erstoping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)

model = NLP_Model(total_words_len)

model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
model.summary()

hist = model.fit(X_train, Y_train, batch_size = 256, epochs = 20, validation_split=0.25, callbacks = [ckpoint, erstoping])#ckpoint)

print('가장 높은 모델의 테스트 정확도 : %.4f' % (load_model('../model/model.h5').evaluate(X_test, Y_test)[1]))

#### 그래프 그리기 ####
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.legend(['accuracy', 'loss'], loc = 'upper left')

plt.show()
