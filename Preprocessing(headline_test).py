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

#total_data = pd.read_csv('.\\훈련데이터\\종합.csv', encoding = 'UTF-8-SIG')
#total_data = pd.read_csv('.\\훈련데이터\\훈련데이터(헤드라인종합).csv', encoding = 'UTF-8-SIG')
df1 = pd.read_csv('.\\헤드라인\\삼성전자 기사 헤드라인.csv', encoding = 'UTF-8-SIG')
df2 = pd.read_csv('.\\헤드라인\\LG전자 기사 헤드라인.csv', encoding = 'UTF-8-SIG')
df3 = pd.read_csv('.\\헤드라인\\현대자동차 기사 헤드라인.csv', encoding = 'UTF-8-SIG')

total_data = [df1, df2, df3]

for i in total_data :
    #del i['Unnamed: 0']
    #print(i.tail())
    i.dropna(inplace = True) # 모든 컬럼에 대해 결측치가 있는 행을 삭제

total_data = pd.concat(total_data, axis = 0)
total_data.reset_index(drop = True, inplace = True)


del total_data['Unnamed: 0'] # 불필요 컬럼 제거(이전 index)

#stopwords_data = pd.read_csv('.\\헤드라인\\불용어 사전.csv', encoding = 'UTF-8-SIG')
#stopwords = stopwords_data['words'].values
stopwords = ['삼성', '네이버', '카카오', '하이닉스',
             '은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한']

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