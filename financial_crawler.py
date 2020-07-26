import pandas as pd

def krx_crawler() :
    url_base = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType='

    # 한 번에 모두 다운받을 수 있으나 데이터프레임 병합 과정 실습을 위해 나눔
    kospi_url =  url_base + 'stockMkt'
    kosdaq_url = url_base + 'kosdaqMkt'

    kospi_df = pd.read_html(kospi_url, header=0)[0]
    kosdaq_df = pd.read_html(kosdaq_url, header=0)[0]

    jongmok_list = [kospi_df, kosdaq_df]
    column_list = ['업종', '주요제품', '상장일', '결산월', '대표자명', '홈페이지', '지역']

    for i in jongmok_list :
        for j in column_list :
            del i[j]

    total_df = pd.concat([kospi_df, kosdaq_df], axis = 0)
    total_df.reset_index(drop=True, inplace=True)

    total_df.to_csv(".\\종목코드\\회사명 및 종목코드.csv", encoding = 'UTF-8-SIG')

