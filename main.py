import pandas as pd

from Financial_Crawler import krx_crawler
from Marketdata_Crawler import get_market_data
from News_crawler import headline_crawler
from Predict_Headline import train_sentimentdata, sentiment_predict

# pip import lxml, html5lib

##krx_crawler() # 주식시장에 상장된 기업 리스트들을 파일로 가져오는 메소드

jongmok = input("종목 이름 입력 : ")

#headline_crawler(jongmok) # 뉴스 헤드라인 웹 크롤링 메소드
get_market_data(jongmok) # yfinance를 활용한 입력한 종목에 대한 1년간의 주가 시계열 데이터 수집 메소드

## 구현부분 ##
# predict # 주가 예측
# precict # 기사 예측(긍부정척도 반환)
#train_sentimentdata(jongmok) # 뉴스 헤드라인 감성분석
#csv_df = pd.read_csv(".\\헤드라인\\삼성전자 기사 헤드라인.csv", encoding = 'UTF-8-SIG')
#sentence = csv_df['헤드라인'].values

#for i in sentence :
#    print(i)
#    sentiment_predict(i)
#train_marketdata() # LSTM 사용 주가 시계열 데이터 예측