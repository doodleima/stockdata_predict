import pandas as pd
import numpy as np

from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
t = Okt()
to = Tokenizer()

def train_sentimentdata(jongmok) :
    X_test = np.load(open('.\\word_dir\\test_data.npy', 'rb')) # 테스트 데이터 입력값
    Y_test = np.load(open('.\\word_dir\\test_label.npy', 'rb')) # 테스트 데이터 레이블

    print('테스트 정확도 : %.4f' % (load_model('.\\model\\model.h5').evaluate(X_test, Y_test)[1]))

def sentiment_predict(words):
  stopwords_data = pd.read_csv('.\\헤드라인\\불용어 사전.csv', encoding = 'UTF-8-SIG')
  stopwords = stopwords_data['words'].values

  words = t.morphs(words, stem=True) # 토큰화
  words = [word for word in words if not word in stopwords] # 불용어 제거
  index = to.texts_to_sequences([words]) # 정수 인코딩
  pad = pad_sequences(index, maxlen = 30) # 패딩
  score = float(load_model('.\\model\\model.h5').predict(pad)) # 예측
  print(score)

  #if(score > 0.5):
  #  print("{:.2f}% 확률로 긍정입니다.\n".format(score * 100))
  #else:
  #  print("{:.2f}% 확률로 부정입니다.\n".format((1 - score) * 100))