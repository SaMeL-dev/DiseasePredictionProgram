import pandas as pd

#2를 0으로 바꾸는 함수(아니오:2 -> 아니오:0), (여자:2 -> 여자:0)
def cols20(df):
    cols = ['BPHIGH4', 'CVDCRHD4', 'CVDSTRK3', 'CHCKIDNY', 'DIABETE3',
                   'PREGNANT', 'EXERANY2', '_MICHD', 'DRNKANY5', 'SEX']
    df[cols] = df[cols].replace(2, 0)
    return df

#null을 0으로 처리 > 원핫 인코딩> drop_first로 젤 앞에있는거 삭제
def colsonehot(df):
    cols = ['SMOKDAY2', '_SMOKER3', '_PACAT1']
    for col in cols:
        df[col] = df[col].fillna(0)
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype('Int64')
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
    return df


#888를 0으로 데이터 처리
#101~199의 값을 가진 데이터면 백의 자리날리고(102면 2만 남게)  *4
#201~299의 값을 가진 데이터면 백의 자리 날리고(223면 23만 남게) 그대로 두기
def convert_cols1(df):
    def convert(value):
        if value == 888:
            return 0
        elif 101 <= value <= 199:
            return (value % 100) * 4
        elif 201 <= value <= 299:
            return value % 100
        return value 
    for col in ['ALCDAY5', 'STRENGTH']:
        df[col] = df[col].apply(convert)
    return df


#555를 0으로 데이터 처리
#101~199의 값을 가진 데이터면 백의 자리날리고(102면 2만 남게)  *30
#201~299의 값을 가진 데이터면 백의 자리 날리고(223면 23만 남게) *4
#301~399의 값을 가진 데이터면 백의 자리 날리고(323면 23만 남게) 그대로
def convert_cols2(df):
    def convert(value):
        if value == 555:
            return 0
        elif 101 <= value <= 199:
            return (value % 100) * 30
        elif 201 <= value <= 299:
            return (value % 100) * 4
        elif 301 <= value <= 399:
            return value % 100
        return value  
    food_cols = ['FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN', 'FVORANG', 'VEGETAB1']
    for col in food_cols:
        df[col] = df[col].apply(convert)
    return df



#1을 0으로/ 2를 1으로 바꾸기
def convert_cols3(df):
    df[['_FRUITEX', '_VEGETEX']] = df[['_FRUITEX', '_VEGETEX']].replace({1: 0, 2: 1})
    return df

#전체 전처리함수
def preprocess_data(df):
    df = cols20(df)
    df = colsonehot(df)
    df = convert_cols1(df)
    df = convert_cols2(df)
    df = convert_cols3(df)
    return df


df = pd.read_excel("BRFSS_2015ver6.xlsx")
df = preprocess_data(df)
df.to_excel("BRFSS_2015ver7.xlsx", index=False)
