from bs4 import BeautifulSoup
from tqdm import tqdm
from random import randint

import requests, time
import pandas as pd

### PARAM ###
day_start = '2019.01.01'
day_end = '2020.12.30'
target_num = 30  # page number
page_num = target_num * 10  # 1 page = 10 sections(news)
key_word = 'LG전자'  # keyword
RANDOM_WAIT = randint(3, 7)  # prevent site-ban(DDOS)

### URL ###
base_url = 'https://search.naver.com/search.naver?where=news&sm=tab_jum'
# sort = 0(관련도순), 1(최신순), 2(오래된순) / pd = 1(1주), 2(1개월), 3(지정) / photo = 3(지면기사)
sort_opt = '&sm=tab_srt&sort=0&pd=3&photo=3'
day_opt = '&ds={daystart}&de={dayend}'.format(daystart=day_start, dayend=day_end)

### CONTENTS ### - store to DB or CSV file
news_dic = {
    'news_title': [],
    'news_company': [],
    'news_link': [],
}

try:
    for i in tqdm(range(1, page_num + 2, 10)):  # tdqm
        # for i in range(1, page_num+2, 10) : # normal
        # CP = lambda n: str(int(n/10)+1) # current page number
        # print('======== PAGE {page} ======='.format(page = CP(i)))
        key_page_opt = '&start={pagenum}&query={keyword}'.format(pagenum=str(i), keyword=key_word)
        full_url = base_url + key_page_opt + sort_opt + day_opt

        html = requests.get(full_url).text
        soup = BeautifulSoup(html, 'html.parser')
        main_area = soup.find('div', {'class': 'group_news'})

        for bx in main_area.find_all('li', {'class': 'bx'}):
            news_area = bx.find('div', {'class': 'news_area'})
            news_com_raw = news_area.find('a', href=True)
            # news_com_link = news_com_raw['href'] 언론사 홈페이지
            news_com = news_com_raw.text.replace('언론사 선정', '')
            news_title_raw = news_area.find('a', {'class': 'news_tit'}, href=True)
            news_title = news_title_raw.text
            news_tit_link = news_title_raw['href']
            # print('Title: {NT}({link})\nCompany: {NC}\n'.format(NT=news_com, link=news_tit_link, NC=news_title))

            if news_title not in news_dic['news_title']:  # if title not duplicated
                # and news_com not in news_dic['news_company'] : # restriction(additional)
                news_dic['news_title'].append(news_title)
                news_dic['news_company'].append(news_com)
                news_dic['news_link'].append(news_tit_link)

        time.sleep(RANDOM_WAIT)

except:
    try:
        df = pd.DataFrame(news_dic)
        df.to_csv(".\\" + key_word + " 기사 헤드라인.csv", encoding='UTF-8-SIG')
    except:
        print('긁어온 기사가 없습니다.\n')
    # return

df = pd.DataFrame(news_dic)
df.to_csv(".\\" + key_word + " 기사 헤드라인.csv", encoding='UTF-8-SIG')  # linux = '/', windows = '\\'