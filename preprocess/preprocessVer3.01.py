import pandas as pd
import numpy as np
#------------------------------------------------
#데이터형 전처리 방향 str로 변환 -> 숫자형 그대로 변환
#-----------------------------------------------

#yes가 1이고 no가 0임
#---종속변수--
def target01(df):
    cols = ['BPHIGH4', 'DIABETE3']

    def convert(value):
        if value in [1, 2, 4]:
            return 1  # Yes
        elif value == 3:
            return 0  # No
        elif value in [7, 9]:
            return np.nan  # Unknown

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df



def target02(df):
    cols = ['CVDCRHD4', 'CVDSTRK3', 'CHCKIDNY']

    def convert(value):
        if value == 1:
            return 1  # Yes
        elif value == 2:
            return 0  # No
        elif value in [7, 9]:
            return np.nan  # Unknown

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df




# ----값이 예/아니오로 된거 처리---------

# 1 = 'yes', 2 = 'no', 7 or 9 = -1, 빈칸 = null
def yesno01(df):
    cols = [
        'HLTHPLN1', 'MEDCOST', 'BPMEDS', 'BLOODCHO',
        'TOLDHI2', 'CVDINFR4', 'ASTHMA3', 'ASTHNOW',
        'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD1',
        'ADDEPEV2', 'VETERAN3', 'INTERNET', 'QLACTLM2',
        'USEEQUIP', 'DECIDE', 'DIFFWALK', 'DIFFDRES',
        'DIFFALON', 'SMOKE100', 'STOPSMK2', 'LMTJOIN3',
        'ARTHDIS2', 'FLUSHOT6', 'PNEUVAC3', 'HIVTST6', 'PDIABTST',
        'INSULIN', 'DIABEYE', 'CIMEMLOS',
        '_RFHLTH', '_HCVU651', '_CHOLCHK',
        '_DRDXAR1', '_HISPANC', '_RACEG21', '_PASTAE1',
        '_FLSHOT6', '_PNEUMO2', '_AIDTST3'
        , '_FRTLT1', '_VEGLT1', 'PREGNANT',
        'EXERANY2', '_MICHD', 'DRNKANY5', '_PAINDX1', '_PASTRNG'
    ]

    def convert(value):
        if value == 1:
            return 1
        elif value == 2:
            return 2
        elif value in [7, 9]:
            return -1
        # 그 외 값은 NaN 유지

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

# 1 = 'Yes', 2 = 'No', 7, 9, 빈칸 = -1
def yesno02(df):
    cols = ['HAVARTH3']

    def convert(value):
        if pd.isnull(value):
            return -1
        elif value in [7, 9]:
            return -1
        elif value == 1:
            return 1
        elif value == 2:
            return 2

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 1 = 'Yes', 2 = 'No', 7/9 = -1
def yesno03(df):
    cols = ['PREDIAB1']

    def convert(value):
        if value in [1, 2]:
            return 1
        elif value == 3:
            return 2
        elif value in [7, 9]:
            return -1

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

# 1 = 'Yes', 2 = 'No',  7/9 = -1
def yesno04(df):
    cols = ['TRNSGNDR']

    def convert(value):
        if value in [1, 2, 3]:
            return 1
        elif value == 4:
            return 2
        elif value in [7, 9]:
            return -1

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df
'''
# 변환
def yesno05(df):
    cols = ['BPHIGH4', 'DIABETE3']

    def convert(value):
        if pd.isnull(value):
            return None
        elif value in [1, 2, 4]:
            return 'Yes'
        elif value == 3:
            return 'No'
        elif value in [7, 9]:
            return 'Unknown'

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df
'''

# 1 = 'No', 2 = 'Yes', 7,9 = 'Unknown', 빈칸 = null
def yesno06(df):
    cols = ['_RFCHOL', '_LTASTH1', '_CASTHM1', '_RFBING5', '_FRUITEX', '_VEGETEX','_RFDRHV5']

    def convert(value):
        if value == 1:
            return 2
        elif value == 2:
            return 1
        elif value in [7, 9]:
            return -1

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

# 1 = 'No', 2 = 'Yes', 빈칸 = null
def yesno07(df):
    cols = ['_RFHYPE5']

    def convert(value):
        if value == 1:
            return 2
        elif value == 2:
            return 1

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


#-------단위통합하기-----

# 101–199 = (백의자리 제거) * 30, 201–299 = * 4, 301–399 = 그대로, 401–499 = /12 후 반올림
# 777,999 = -1, 888 = 0, 빈칸 = null
def Unit_integration01(df):
    cols = ['BLDSUGAR', 'FEETCHK2']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        try:
            value = int(value)
        except:
            return np.nan

        if value in [777, 999]:
            return -1   # Unknown → -1
        elif value == 888:
            return 0    # Zero → 0
        elif 101 <= value <= 199:
            return (value % 100) * 30
        elif 201 <= value <= 299:
            return (value % 100) * 4
        elif 301 <= value <= 399:
            return value % 100
        elif 401 <= value <= 499:
            return int(round((value % 100) / 12))
        else:
            return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df



# 101–199 = 주당 *4, 201–299 = 그대로, 777,999 = -1, 빈칸 = null
def Unit_integration02(df):
    cols = ['EXEROFT1', 'EXEROFT2']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        try:
            value = int(value)
        except:
            return np.nan

        if value in [777, 999]:
            return -1
        elif 101 <= value <= 199:
            return (value % 100) * 4
        elif 201 <= value <= 299:
            return value % 100
        else:
            return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 모든 값을 시간*60 + 분으로 변환 후 str형으로 변환
# 777, 999 = -1, 888 = 0, 빈칸 = null
def Unit_integration03(df):
    cols = ['EXERHMM1', 'EXERHMM2']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        try:
            value = int(value)
        except:
            return np.nan

        if value in [777, 999]:
            return -1
        elif value == 888:
            return 0
        else:
            hours = value // 100
            minutes = value % 100
            total_minutes = hours * 60 + minutes
            return total_minutes

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 101~199 = 주당 음주일 수 * 4, 201~299 = 지난 30일 음주일 수 그대로
# 888 = 0, 777, 999 = -1, BLANK = null
def Unit_integration04(df):
    cols = ['ALCDAY5']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        try:
            value = int(value)
        except:
            return np.nan

        if value == 888:
            return 0
        elif value in [777, 999]:
            return -1
        elif 101 <= value <= 199:
            return (value % 100) * 4
        elif 201 <= value <= 299:
            return value % 100
        else:
            return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 101~199 = (백의 자리 제거) * 30, 201~299 = *4, 301~399 = 그대로
# 300, 555 = 0, 777,999 = -1, BLANK = null
def Unit_integration05(df):
    cols = ['FRUITJU1', 'FRUIT1', 'FVBEANS', 'FVGREEN', 'FVORANG', 'VEGETAB1']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        try:
            value = int(value)
        except:
            return np.nan

        if 101 <= value <= 199:
            return (value % 100) * 30
        elif 201 <= value <= 299:
            return (value % 100) * 4
        elif 301 <= value <= 399:
            return value % 100
        elif value in [300, 555]:
            return 0
        elif value in [777, 999]:
            return -1
        else:
            return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

# 101~199 = (백의 자리 제거) * 4, 201~299 = 그대로, 888 = 0, 777,999 = -1, BLANK = null
def Unit_integration06(df):
    cols = ['STRENGTH']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        try:
            value = int(value)
        except:
            return np.nan

        if value == 888:
            return 0
        elif value in [777, 999]:
            return -1
        elif 101 <= value <= 199:
            return (value % 100) * 4
        elif 201 <= value <= 299:
            return value % 100
        else:
            return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df





#---------정수 str로 변환하기--------

#1~30을 그대로, 88 = 0,  77,99= -1, 빈칸 null
def Integer_conversion01(df):
    cols = ['PHYSHLTH', 'MENTHLTH', 'POORHLTH']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 88:
            return 0
        elif value in [77, 99]:
            return -1
        elif 1 <= value <= 30:
            return int(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


#0~10은 그대로, 77, 99 : -1, 빈칸 null
def Integer_conversion02(df):
    cols = ['JOINPAIN']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value in [77, 99]:
            return -1
        elif 0 <= value <= 10:
            return int(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


#11985 - 122016 는 그대로, 777777,999999 : -1, BLANK : null
def Integer_conversion03(df):
    cols = ['HIVTSTD3']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value in [777777, 999999]:
            return -1
        elif 11985 <= value <= 122016:
            return int(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


#1~76을 그대로 88 = 0, 77,99= -1, 빈칸 null
def Integer_conversion04(df):
    cols = ['DOCTDIAB', 'FEETCHK', 'DRNK3GE5', 'MAXDRNKS', 'AVEDRNK2']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 88:
            return 0
        elif value in [77, 99]:
            return -1
        elif 1 <= value <= 76:
            return int(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

#0–8590 그대로, 99900 = -1
def Integer_conversion05(df):
    cols = ['FC60_']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 99900:
            return -1
        #elif value == 0:
        #    return 'None'
        elif 0 <= value <= 8590:
            return value

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df



# 88 = 0, 77,99 = -1, 98 = -1, BLANK는 null, 나머지를 다 그대로
def Integer_conversion06(df):
    cols = ['EXRACT11', 'EXRACT21']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 88:
            return 0
        elif value in [77, 99]:
            return -1
        elif value == 98:
            return -1
        else:
            return int(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 0–98999 정수 그대로, 99900 = -1
def Integer_conversion07(df):
    cols = ['_DRNKWEK']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 99900:
            return -1
        #elif value == 0:
        #    return 'None'
        elif 0 <= value <= 98999:
            return int(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 0–9999 그대로 정수
def Integer_conversion08(df):
    cols = ['FTJUDA1_', 'FRUTDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        #elif value == 0:
        #    return 'None'
        elif 0 <= value <= 9999:
            return int(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

'''
# 0–99998 = str형으로 변환, BLANK = null
#전처리 필요 X
def Integer_conversion09(df):
    cols = ['_FRUTSUM', '_VEGESUM']

    def convert(value):
        if pd.isnull(value):
            return None
        elif 0 <= value <= 99998:
            return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 0–599 = str형으로 변환, BLANK = null
#PADUR1_ → 최솟값: 1.0, 개수: 890
#PADUR2_ → 최솟값: 1.0, 개수: 748   이걸 문자형으로
#전처리 필요 X
def Integer_conversion10(df):
    cols = ['PADUR1_', 'PADUR2_']

    def convert(value):
        if pd.isnull(value):
            return None
        elif 0 <= value <= 599:
            return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df
'''

# 0–98999 그대로, 99000 =-1, 빈칸 = null
def Integer_conversion11(df):
    cols = ['PAFREQ1_', 'PAFREQ2_', 'STRFREQ_']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 99000:
            return -1
        #elif value == 0:
        #    return 'None'
        elif 0 <= value <= 98999:
            return int(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

'''
# 0–99999 = str형으로 변환, BLANK = null
#전처리 필요 X
def Integer_conversion12(df):
    cols = ['_MINAC11', '_MINAC21', 'PAMIN11_', 'PAMIN21_', 'PAVIG11_', 'PAVIG21_']

    def convert(value):
        if pd.isnull(value):
            return None
        elif 0 <= value <= 99999:
            return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df



# str형으로 변환, 빈칸 = null
#전처리 필요 X
def Integer_conversion13(df):
    cols = ['HTM4']

    def convert(value):
        if pd.isnull(value):
            return None
        #elif value == 0:
        #    return 'None'
        return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df
'''

# 2300–29500 그대로, 99999 = -1
def Integer_conversion14(df):
    cols = ['WTKG3']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 99999:
            return -1
        elif 2300 <= value <= 29500:
            return int(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

'''
# str형으로 변환
#전처리 필요 X
def Integer_conversion15(df):
    cols = ['_BMI5']

    def convert(value):
        if pd.isnull(value):
            return 'Unknown'
        return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df
'''




# 정수 그대로, 900 = -1, 빈칸 = null
def Integer_conversion16(df):
    cols = ['DROCDY3_']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 900:
            return -1
        return int(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 1~76 그래로, 88 = 0, 77,98,99 = -1', 빈칸 = null
def Integer_conversion17(df):
    cols = ['CHKHEMO3']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif 1 <= value <= 76:
            return int(value)
        elif value == 88:
            return 0
        elif value in [77, 98, 99]:
            return -1

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


#------범주형으로 변환-

def categorical(df):
    mappings = {
        'GENHLTH': [7, 9],
        'PERSDOC2': [7, 9],
        'CHECKUP1': [7, 8, 9],
        'CHOLCHK': [7, 8, 9],
        'MARITAL': [9],
        'EDUCA': [9],
        'RENTHOM1': [7, 9],
        'EMPLOY1': [9],
        'INCOME2': [77, 99],
        'LASTSMK2': [9, 99],
        'USENOW3': [7, 9],
        'ARTHSOCL': [7, 9],
        'WHRTST10': [77, 99],
        'SXORIENT': [7, 9],
        'MSCODE': [],
        '_ASTHMS1': [9],
        '_MRACE1': [77, 99],
        '_RACE': [9],
        '_INCOMG': [9],
        'ACTIN11_': [],
        'ACTIN21_': [],
        '_PA300R2': [9],
        '_PAREC1': [9],
        '_BMI5CAT': [],
        '_PA150R2': [9],
        'SEX': [],
        'SMOKDAY2': [7, 9],
        '_SMOKER3': [9],
        '_PACAT1': [9],
        '_AGEG5YR': []
    }

    for col, bad_values in mappings.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: -1 if x in bad_values else x)

    return df


def bound_variable(df):
    target_cols = ['BPHIGH4', 'CVDCRHD4', 'CVDSTRK3', 'CHCKIDNY', 'DIABETE3']
    other_cols = [col for col in df.columns if col not in target_cols]
    return df[target_cols + other_cols]

#전체 전처리함수
def preprocess_data(df):

    df = target01(df)
    df = target02(df)

    df = yesno01(df)
    df = yesno02(df)
    df = yesno03(df)
    df = yesno04(df)
    #df = yesno05(df)
    df = yesno06(df)
    df = yesno07(df)

    df = Unit_integration01(df)
    df = Unit_integration02(df)
    df = Unit_integration03(df)
    df = Unit_integration04(df)
    df = Unit_integration05(df)
    df = Unit_integration06(df)

    df = Integer_conversion01(df)
    df = Integer_conversion02(df)
    df = Integer_conversion03(df)
    df = Integer_conversion04(df)
    df = Integer_conversion05(df)
    df = Integer_conversion06(df)
    df = Integer_conversion07(df)
    df = Integer_conversion08(df)
    #df = Integer_conversion09(df)
    #df = Integer_conversion10(df)
    df = Integer_conversion11(df)
    #df = Integer_conversion12(df)
    #df = Integer_conversion13(df)
    df = Integer_conversion14(df)
    #df = Integer_conversion15(df)
    df = Integer_conversion16(df)
    df = Integer_conversion17(df)

    df = categorical(df)
    df = bound_variable(df)
    return df


df = pd.read_csv("BRFSS_2015ver11.csv")
df = preprocess_data(df)
df.to_csv("BRFSS_2015ver17.csv", index=False)

