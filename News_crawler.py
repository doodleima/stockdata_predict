# 네이버 뉴스 크롤링

from bs4 import BeautifulSoup
from urllib.request import urlopen

import urllib
import pandas as pd

def headline_crawler(jongmok) :
    base = 'https://search.naver.com/search.naver?where=news&sm=tab_jum'
    page = '&start={pagenum}&query={keyword}'
    option = '&sm=tab_srt&sort=0&pd=1&photo=3' #&field=1 - 영역 설정하니까 뉴스가 매우 많이 줄음, 비활성화
    # sort는 정렬방식(0 관련도순, 1 최신순, 2 오래된순), pd는 기간(1 1주, 2 1개월), photo는 기사 유형(3은 지면기사), field는 영역(내용, 1이 제목)

    url_full = base + page + option  # 전체 URL 주소

    wordslist = []  # 빈 리스트 하나 생성

    ## 불용어 사전 정의 ##
    words = ['뉴스검색 가이드', '관련도순', '최신순', '오래된순', '정지', '시작', '네이버뉴스', '보내기', '\n', '',
             '이전페이지', '다음페이지', '이전 년도', '다음 년도', '이전 달 ', '다음 달 ']
    newscount = []
    pagecount = []
    ## 불용어 사전 정의 끝 ##
    pagenum = 1

    for i in range(0, 110):
        newscount.append('관련뉴스 ' + str(i) + '건 전체보기') # 불용어 사전 리스트에 값 집어넣기
        pagecount.append(str(i)) # 불용어 사전 리스트에 값 집어넣기


    for num in range(0, 300, 10): # 50개의 페이지에 대해 수행, 30개가 적당할지도? 긁어올 페이지 수 고려해봐야..
        print("===== " + str(pagenum) +"번째 페이지의 헤드라인을 가져옵니다. =====")
        res = urlopen(url_full.format(pagenum=num, keyword=urllib.parse.quote(jongmok)))
        soup = BeautifulSoup(res, 'html.parser')
        tags = soup.find_all('a', {'_sp_each_title': ""})

        to_show = 0

        for k in range(0, 200):  # 110~ 173
            if tags[k].text == '뉴스검색 가이드':  # 시작점
                to_show = 1;
            if to_show == 0:  # 뉴스검색 가이드가 나오기 전까지
                continue;  # 출력하지 않음

            if tags[k].text == '24시간센터':  # 끝점
                break;  # 24시간센터를 만나면 루프 탈출, 그렇지 않으면 반복

            if tags[k].text not in words and tags[k].text not in newscount and tags[k].text not in pagecount:
                if tags[k].text not in wordslist:  # 리스트에 중복된 기사 제목이 없으면
                    wordslist.append(tags[k].text)  # 리스트에 추가
                print(tags[k].text)
        print("===== " + str(pagenum) + "번째 페이지 탐색 완료 =====\n")
        pagenum += 1

    df = pd.DataFrame(wordslist)
    df.columns = ['헤드라인']
    df.to_csv(str(jongmok) + " 기사 헤드라인.csv", encoding = 'UTF-8-SIG') #"C:\\Users\\Lim\\Desktop\\" +

