import pandas as pd

df1 = pd.read_csv("훈련데이터\\삼성전자 기사 헤드라인.csv", encoding ='UTF-8-SIG')
df2 = pd.read_csv("훈련데이터\\LG전자 기사 헤드라인.csv", encoding ='UTF-8-SIG')
df3 = pd.read_csv("훈련데이터\\SK하이닉스 기사 헤드라인.csv", encoding ='UTF-8-SIG')
df4 = pd.read_csv("훈련데이터\\현대자동차 기사 헤드라인.csv", encoding ='UTF-8-SIG')
df5 = pd.read_csv("훈련데이터\\NAVER 기사 헤드라인.csv", encoding ='UTF-8-SIG')

df_list = [df1, df2, df3, df4, df5]

for i in df_list :
    del i['Unnamed: 0']

total_df = pd.concat(df_list, axis = 0)
total_df.drop_duplicates(subset = ['헤드라인'], inplace = True)
total_df.reset_index(drop = True, inplace = True)

total_df.to_csv(".\\훈련데이터\\훈련 데이터(헤드라인).csv", encoding = 'UTF-8-SIG')