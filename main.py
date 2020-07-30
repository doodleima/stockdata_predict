import pandas as pd

from Financial_Crawler import krx_crawler
from Marketdata_Crawler import get_market_data
from News_Crawler import headline_crawler
from Predict_Headline import train_sentimentdata, sentiment_predict

from Predict_Stockdata import predict_stock

def search(moonja) :
    krx_crawler() # 주식시장에 상장된 기업 리스트들을 파일로 가져오는 메소드
    jongmok = moonja

    headline_crawler(jongmok) # 뉴스 헤드라인 웹 크롤링 메소드
    get_market_data(jongmok) # yfinance를 활용한 입력한 종목에 대한 1년간의 주가 시계열 데이터 수집 메소드

    status = predict_stock(jongmok)

    csv_df = pd.read_csv(".\\헤드라인\\" + str(jongmok) + " 기사 헤드라인.csv", encoding = 'UTF-8-SIG')
    sentence = csv_df['헤드라인'].values

    count = 0
    percent = float(0)
    negative = 0
    positive = 0

    for i in sentence :
        count += 1
        percent_add = sentiment_predict(i)
        percent += percent_add

        if percent_add > 0.5 :
            positive += 1
        else :
            negative += 1

    if negative == 0 :
        pone = "긍정"
    elif positive == 0 :
        pone = "부정"
    else :
        if (positive / negative) > 1:
            pone = "긍정"
        elif (positive / negative) < 1 :
            pone = "부정"
        else :
            pone = "동일"

    acc = percent / count

    print(positive, negative)
    moonja1 = "주가 상승/하락 여부 : " + status
    moonja2 = "헤드라인 긍/부정 여부 : " + pone
    moonja3 = "정확도 : % .2f" % acc

    return moonja1, moonja2, moonja3