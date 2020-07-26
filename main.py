from financial_crawler import krx_crawler
from marketdata_crawler import get_market_data
from news_crawler import headline_crawler
from train_headline import train_sentimentdata
# pip import lxml, html5lib

krx_crawler() # 주식시장에 상장된 기업 리스트들을 파일로 가져오는 메소드

jongmok = input("종목 이름 입력 : ")

#headline_crawler(jongmok) # 뉴스 헤드라인 웹 크롤링 메소드
#get_market_data(jongmok) # yfinance를 활용한 입력한 종목에 대한 1년간의 주가 시계열 데이터 수집 메소드

## 구현부분 ##
train_sentimentdata(jongmok) # 뉴스 헤드라인 감성분석
#train_marketdata() # LSTM 사용 주가 시계열 데이터 예측

# 결과출력 - 파이썬 GUI로 프로그램 구동 가능하게끔..? Tkinter()