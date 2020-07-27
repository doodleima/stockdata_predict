# stopwords data from https://www.ranks.nl/stopwords/korean

#def train_sentimentdata(jongmok) :
#jongmok = '삼성전자'

import pandas as pd
import numpy as np

from tqdm import tqdm
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

t = Okt()
to = Tokenizer()

#total_data = pd.read_csv('.\\헤드라인\\' + str(jongmok) + ' 기사 헤드라인.csv', encoding = 'UTF-8-SIG')

#total_data = pd.read_csv('.\\헤드라인\\삼성전자 기사 헤드라인.csv', encoding = 'UTF-8-SIG')
#total_data = pd.read_csv('.\\헤드라인\\LG전자 기사 헤드라인.csv', encoding = 'UTF-8-SIG')
total_data = pd.read_csv('.\\헤드라인\\현대자동차 기사 헤드라인.csv', encoding = 'UTF-8-SIG')

del total_data['Unnamed: 0'] # 불필요 컬럼 제거(이전 index)

stopwords_data = pd.read_csv('.\\헤드라인\\불용어 사전.csv', encoding = 'UTF-8-SIG')
stopwords = stopwords_data['words'].values

total_data['헤드라인'] = total_data['헤드라인'].str.replace('[^ㄱ-ㅎ 가-힣 ㅏ-ㅣ]', '')
total_data.drop_duplicates(subset = ['헤드라인'], inplace = True)

X_test = []
Y_test = []

for words in tqdm(total_data['헤드라인']) :
    X_token = []
    X_token = t.morphs(words, stem = True)
    X_token = [token_words for token_words in X_token if not token_words in stopwords]
    X_test.append(X_token)

X_test = to.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen = 30) # len_data = 30
Y_test = np.array(total_data['레이블'])

np.save(open('.\\word_dir\\test_data.npy', 'wb'), X_test)
np.save(open('.\\word_dir\\test_label.npy', 'wb'), Y_test)