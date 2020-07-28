# 기사 헤드라인 레이블링 검증 필요

import json
import os
import pandas as pd
import numpy as np

from tqdm import tqdm
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

t = Okt()
to = Tokenizer()

#total_data = pd.read_csv('.\\헤드라인\\' + str(jongmok) + ' 기사 헤드라인.csv', encoding = 'UTF-8-SIG')

total_data = pd.read_csv('훈련데이터(NSMC&Headline)\\훈련데이터(헤드라인종합).csv', encoding ='UTF-8-SIG')

del total_data['Unnamed: 0'] # 불필요 컬럼 제거(이전 index)

stopwords = ['삼성', '네이버', '카카오', '하이닉스'
             '은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한']

total_data['헤드라인'] = total_data['헤드라인'].str.replace('[^ㄱ-ㅎ 가-힣 ㅏ-ㅣ]', '')
total_data.drop_duplicates(subset = ['헤드라인'], inplace = True)

#print(len(total_data)) #2611 (1958/2611)

# 훈련 셋과 테스트 셋으로 데이터를 나눔 : 약75%/25%
train_data = total_data[0:2000]
test_data = total_data[2000:]

X_train = []
X_test = []

# 1. 훈련 데이터 토크나이징
for words in tqdm(train_data['헤드라인']) :
    X_token = []
    X_token = t.morphs(words, stem = True)
    X_token  = [token_words for token_words in X_token if not token_words in stopwords]
    X_train.append(X_token)

# 2. 테스트 데이터 토크나이징
for words in tqdm(test_data['헤드라인']) :
    X_token = []
    X_token = t.morphs(words, stem = True)
    X_token  = [token_words for token_words in X_token if not token_words in stopwords]
    X_test.append(X_token)

to = Tokenizer(3751, oov_token= 'OOV') # 단어셋 크기 3749+2, out of vocabulary, 단어셋에 없는 단어 예외처리
to.fit_on_texts(X_train)

total_words = to.word_index
total_words_len = len(to.word_index) # 단어 총 갯수

X_train = to.texts_to_sequences(X_train)
X_test = to.texts_to_sequences(X_test)

# Y_train, Y_test
Y_train = np.array(train_data['레이블'])
Y_test = np.array(test_data['레이블'])

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
print(total_words_len)

os.makedirs('word_dir', exist_ok= True)
np.save(open('.\\word_dir\\headline_input.npy', 'wb'), X_train)
np.save(open('.\\word_dir\\headline_label.npy', 'wb'), Y_train)

json.dump(data_configs, open('.\\word_dir\\data_configs.json', 'w', -1, "UTF-8-SIG"), ensure_ascii = False)
