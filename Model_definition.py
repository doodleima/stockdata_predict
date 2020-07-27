# 양방향 LSTM 사용 모델 정의 메소드

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Bidirectional, Dropout, GlobalMaxPool1D

def NLP_Model(total_words_len) :
    model = Sequential()
    model.add(Embedding(total_words_len, 128, input_length = 200)) # 임베딩 벡터 차원 128
    model.add(Bidirectional(LSTM(64, return_sequences = True)))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu')) # ReLU
    model.add(Dense(1, activation='sigmoid')) # 시그모이드

    return model

def Stock_Model() :
    model = Sequential()
    model.add(Embedding(64, 128, input_length = 200)) # 임베딩 벡터 차원 64 바꿔줘야 함
    model.add(Bidirectional(LSTM(64, return_sequences = True)))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu')) # ReLU
    model.add(Dense(1, activation='sigmoid')) # 시그모이드

    return model