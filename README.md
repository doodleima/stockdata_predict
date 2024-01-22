1. Usage
```
### 윈도우의 경우 경로를 \\와 같이 표기해야 할 수 있음
### 순서대로 실행

### install essential packages
pip install -r ./requirements.txt

### 1: 회사이름(data/template/codes.csv에 존재해야 함)
python ./util/util_cralwer.py --inc_name {1}   ### inc_name 인자값 주지 않을경우 krx_crawler만 실행됨

### 1: train or test | 2: 회사이름(data/template/codes.csv에 존재해야 함)
python ./src/train.py --mode {1} --inc_name {2}
```

2. Etc
  - 현재 data(for train)는 LG전자, 네이버, 삼성전자, 현대자동차 4개의 셀로 구성
  - 직접 dataset을 수집하고 싶다면 util/util_crawler의 line 26(get_market_data)의 period 값을 변경하여 시도할 것
    ```
    inc_history = market_data.history(period='10Y')
    ```
  - Training set과 External Validation Set 구분을 위해 Period 대신 start / end date를 지정하여 가져오는 방법도 있음
    ```
    start_date="2023-01-01"
    end_date="2023-12-31"

    inc_history = market_data.history(start=start_date, end=end_date)
    ```