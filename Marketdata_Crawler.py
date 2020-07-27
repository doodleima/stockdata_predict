import pandas as pd
import yfinance as yf

def get_market_data(jongmok) :
    jongmoklist = ['Volume','Dividends','Stock Splits']

    jongmokdb = pd.read_csv('.\\종목코드\\회사명 및 종목코드.csv', encoding='UTF-8-SIG') # 주식시장에 상장된 기업 리스트 파일 불러오기
    del jongmokdb['Unnamed: 0'] # 불필요 인덱스 제거, 안해도 됨

    search_code = jongmokdb['종목코드'][jongmokdb['회사명'] == jongmok]
    str_code = str(search_code.values)
    len_code = len(str_code[1:-1])

    ## 종목코드 완성
    if (len_code == 4) :
        jongmok_code = str('00') + str_code[1:-1]

    elif (len_code == 5) :
        jongmok_code = str('0') + str_code[1:-1]

    else :
        jongmok_code = str_code[1:-1]

    print("===== " + jongmok + "(" + jongmok_code + ")의 주가 데이터(1Y)를 불러옵니다. =====")
    # yahoo finance 라이브러리를 사용하여 해당 종목코드에 대한 1년간의 주가 데이터 불러옴
    market_data = yf.Ticker(jongmok_code + ".KS")
    jongmok_history = market_data.history(period='1y')

    for i in jongmoklist : # 반복문 돌려 필요 없는 컬럼은 제거
        del jongmok_history[i]
    print("===== " + jongmok + "(" + jongmok_code + ")의 주가 데이터(1Y)를 불러오기 완료 =====")
    jongmok_history.to_csv(str(jongmok) + " 주가추이(1년).csv", encoding = 'UTF-8-SIG')

    # 주식 데이터 가져오기 : API 쓸건지? 아니면 그냥 야후파이낸셜? 네이버?
    # 기간 정해서 해당 기간만큼만 가져와야지~

