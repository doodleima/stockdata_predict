import os
import sys
import argparse

import pandas as pd
import yfinance as yf

from pathlib import Path
sys.path.append(Path.cwd())

def get_market_data(inc_name, mode='test') :
    inc_db = pd.read_csv(os.path.join(Path.cwd(), 'data', 'template', 'codes.csv'), encoding='UTF-8-SIG') 
    del inc_db['Unnamed: 0'] # delete no needed idx

    search_code = inc_db['종목코드'][inc_db['회사명'] == inc_name]
    str_code = str(search_code.values)
    len_code = len(str_code[1:-1])

    ## combinate inc full code with digit 0
    if (len_code == 4): inc_code = str('00') + str_code[1:-1]
    elif (len_code == 5): inc_code = str('0') + str_code[1:-1]
    else: inc_code = str_code[1:-1]


    market_data = yf.Ticker(inc_code + ".KS")
    inc_history = market_data.history(period='3Y')
    print(inc_history)

    # inc_history['Date'] = pd.to_datetime(inc_history['Date']).dt.strftime('%Y-%m-%d')
    for i in ['Dividends','Stock Splits']: del inc_history[i]
    inc_history.to_csv(os.path.join(Path.cwd(), 'data', mode, f"{inc_name}.csv"), encoding = 'UTF-8-SIG')

    print(inc_history.keys())

def krx_crawler() :
    url_base = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType='

    ### kospi / kosdaq split
    kospi_url =  url_base + 'stockMkt'
    kosdaq_url = url_base + 'kosdaqMkt'

    kospi_df = pd.read_html(kospi_url, header=0)[0]
    kosdaq_df = pd.read_html(kosdaq_url, header=0)[0]

    code_list = [kospi_df, kosdaq_df]
    column_list = ['업종', '주요제품', '상장일', '결산월', '대표자명', '홈페이지', '지역']

    for i in code_list :
        for j in column_list : del i[j]

    total_df = pd.concat([kospi_df, kosdaq_df], axis = 0)
    total_df.reset_index(drop=True, inplace=True)

    total_df.to_csv(os.path.join(Path.cwd(), 'data', 'template', 'codes.csv'), encoding = 'UTF-8-SIG')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inc_name", type=str, help="incorporate name: 'LG전자', '네이버', ...", default=None)

    args = parser.parse_args()

    krx_crawler()
    if args.inc_name: get_market_data(args.inc_name)