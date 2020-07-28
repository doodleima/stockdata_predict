# 모델 정의 메소드

from tensorflow.keras.layers import Embedding, Dense, LSTM, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Bidirectional, Dropout, GlobalMaxPool1D

def NLP_Model(total_words_len) :
    model = Sequential()
    model.add(Embedding(total_words_len+1, 128, input_length = 30))
    model.add(Dropout(0.1))
    model.add(Conv1D(768, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(256)) # , activation='relu'
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    """
    model = Sequential()
    model.add(Embedding(total_words_len, 512, input_length = 200)) # 임베딩 벡터 차원 512
    model.add(Dropout(0.1))
    model.add(Conv1D(768, 12, activation = 'relu')) # GeLU
    model.add(GlobalMaxPool1D()) # 가장 큰 벡터를 반환
    model.add(Dense(12, activation='relu')) # 은닉 계층
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid')) # 마지막 계층 활성화 함수로 시그모이드
    """
    return model

def Stock_Model() :
    model = Sequential()
    model.add(Embedding(64, 128, input_length = 200)) # 임베딩 벡터 차원 변경해야 함
    model.add(Bidirectional(LSTM(64, return_sequences = True)))
    model.add(GlobalMaxPool1D()) # 가장 큰 벡터를 반환
    model.add(Dropout(0.2)) # 0.2 확률로 은닉층 유닛 제거
    model.add(Dense(64, activation='relu')) # ReLU
    model.add(Dense(1, activation='sigmoid')) # 시그모이드

    return model
