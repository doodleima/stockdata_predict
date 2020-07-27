# 자연어처리 기반 데이터 전처리 코드
# 프로그램에는 포함되지 않음, input.npy/label.npy 및 data_configs.json 생성을 위해 필요

# Naver Sentiment Movie Corpus data(ratings.txt)
# data from https://github.com/e9t/nsmc/

# stopwords data from https://www.ranks.nl/stopwords/korean
import json

import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

t = Okt()

# NSMC 데이터 : ratings.txt(영화 리뷰/평점 텍스트 데이터)
total_data = pd.read_csv('.\\훈련데이터\\영화 리뷰 평점 데이터.csv', encoding = 'UTF-8-SIG')
del total_data['Unnamed: 0'] # 불필요 컬럼 제거(이전 index)

# stopwords 데이터 : 불용어 사전.csv
# csv 파일에 있는 값들을 불용어 사전에 추가
stopwords_data = pd.read_csv('.\\헤드라인\\불용어 사전.csv', encoding = 'UTF-8-SIG')
stopwords = stopwords_data['words'].values

# 한글, 스페이스 외에 모든 문자를 제거함 : 정규표현식 사용
total_data['document'] = total_data['document'].str.replace('[^ㄱ-ㅎ 가-힣 ㅏ-ㅣ]', '')

# drop_duplicates()로 중복을 제거함 - inplace 옵션 활성화로 원본 객체 변경
total_data.drop_duplicates(subset=['document'], inplace = True)

# 훈련 셋과 테스트 셋으로 데이터를 나눔 : 75%, 25%
train_data = total_data[0:150000]
test_data = total_data[150000:]

# X는 데이터, Y는 레이블
X_train = []
X_test = []

#### 데이터 토크나이징 ####
# 1. 훈련 데이터 토크나이징
for words in tqdm(train_data['document']) :
    X_token = []
    X_token = t.morphs(words, stem = True)
    X_token  = [token_words for token_words in X_token if not token_words in stopwords]
    X_train.append(X_token)

# 2. 테스트 데이터 토크나이징
for words in tqdm(test_data['document']) :
    X_token = []
    X_token = t.morphs(words, stem = True)
    X_token  = [token_words for token_words in X_token if not token_words in stopwords]
    X_test.append(X_token)
#### 데이터 토크나이징 ####

#### 인덱싱 ####
to = Tokenizer(19141, oov_token= 'OOV') # 단어셋 크기를 19140+1 으로 설정, out of vocabulary, 단어셋에 없는 단어 예외처리
to.fit_on_texts(X_train)

total_words = to.word_index
total_words_len = len(to.word_index) # 단어 총 갯수

X_train = to.texts_to_sequences(X_train)
X_test = to.texts_to_sequences(X_test)

# Y_train, Y_test
Y_train = np.array(train_data['label'])
Y_test = np.array(test_data['label'])

drop_data = [index for index, words in enumerate(X_train) if len(words) < 1]
X_train = np.delete(X_train, drop_data, axis = 0)
Y_train = np.delete(Y_train, drop_data, axis = 0)

# 데이터별 최대 허용 길이를 30로 설정(패딩)
len_data = 30
X_train = pad_sequences(X_train, maxlen = len_data)
X_test = pad_sequences(X_test, maxlen = len_data)

data_configs = {}

data_configs['vocab'] = total_words
data_configs['vocab_size'] = total_words_len

os.makedirs('word_dir', exist_ok= True)
np.save(open('.\\word_dir\\input.npy', 'wb'), X_train)
np.save(open('.\\word_dir\\label.npy', 'wb'), Y_train)

json.dump(data_configs, open('.\\word_dir\\data_configs.json', 'w', -1, "UTF-8-SIG"), ensure_ascii = False)



