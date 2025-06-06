import pandas as pd
import numpy as np

#---종속변수--
def target01(df):
    cols = ['BPHIGH4', 'DIABETE3']

    def convert(value):
        if value in [1, 2, 4]:
            return 1  # Yes
        elif value == 3:
            return 0  # No
        elif value in [7, 9]:
            return 2  # Unknown

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
            return 2  # Unknown

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df




# ----값이 예/아니오로 된거 처리---------

# 1 = 'yes', 2 = 'no', 7 or 9 = 'unknown', 빈칸 = null
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
            return 'Yes'
        elif value == 2:
            return 'No'
        elif value in [7, 9]:
            return 'Unknown'
        # 그 외 값은 NaN 유지

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

# 1 = 'Yes', 2 = 'No', 7, 9, 빈칸 = 'Unknown'
def yesno02(df):
    cols = ['HAVARTH3']

    def convert(value):
        if pd.isnull(value):
            return 'Unknown'
        elif value in [7, 9]:
            return 'Unknown'
        elif value == 1:
            return 'Yes'
        elif value == 2:
            return 'No'

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 1,2 = 'Yes', 3 = 'No', 7,9 = 'Unknown', 빈칸 = null
def yesno03(df):
    cols = ['PREDIAB1']

    def convert(value):
        if value in [1, 2]:
            return 'Yes'
        elif value == 3:
            return 'No'
        elif value in [7, 9]:
            return 'Unknown'

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

# 1,2,3 = 'Yes', 4 = 'No', 7,9 = 'Unknown', 빈칸 = null
def yesno04(df):
    cols = ['TRNSGNDR']

    def convert(value):
        if value in [1, 2, 3]:
            return 'Yes'
        elif value == 4:
            return 'No'
        elif value in [7, 9]:
            return 'Unknown'

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df
'''
# 1,2,4 = 'Yes', 3 = 'No', 7,9 = 'Unknown', 빈칸 = null
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
            return 'No'
        elif value == 2:
            return 'Yes'
        elif value in [7, 9]:
            return 'Unknown'

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

# 1 = 'No', 2 = 'Yes', 3과 빈칸 = null
def yesno07(df):
    cols = ['_RFHYPE5']

    def convert(value):
        if value == 1:
            return 'No'
        elif value == 2:
            return 'Yes'

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


#-------단위통합하기-----

# 101–199 = (백의자리 제거) * 30, 201–299 = * 4, 301–399 = 그대로, 401–499 = /12 후 반올림
# 777,999 = "Unknown", 888 = "None", 빈칸 = null
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
            return 'Unknown'
        elif value == 888:
            return 'Zero'
        elif 101 <= value <= 199:
            return str((value % 100) * 30)
        elif 201 <= value <= 299:
            return str((value % 100) * 4)
        elif 301 <= value <= 399:
            return str(value % 100)
        elif 401 <= value <= 499:
            return str(int(round((value % 100) / 12)))
        else:
            return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 101–199 = 주당 *4, 201–299 = 그대로, 777,999 = "Unknown", 빈칸 = null
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
            return 'Unknown'
        elif 101 <= value <= 199:
            return str((value % 100) * 4)
        elif 201 <= value <= 299:
            return str(value % 100)
        else:
            return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 모든 값을 시간*60 + 분으로 변환 후 str형으로 변환
# 777, 999 = "Unknown", 888 = "None", 빈칸 = null
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
            return 'Unknown'
        elif value == 888:
            return 'Zero'
        else:
            hours = value // 100
            minutes = value % 100
            total_minutes = hours * 60 + minutes
            return str(total_minutes)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 101~199 = 주당 음주일 수 * 4, 201~299 = 지난 30일 음주일 수 그대로
# 888 = "None", 777, 999 = "Unknown", BLANK = null
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
            return 'Zero'
        elif value in [777, 999]:
            return 'Unknown'
        elif 101 <= value <= 199:
            return str((value % 100) * 4)
        elif 201 <= value <= 299:
            return str(value % 100)
        else:
            return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 101~199 = (백의 자리 제거) * 30, 201~299 = *4, 301~399 = 그대로
# 300, 555 = 'None', 777,999 = 'Unknown', BLANK = null
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
            return str((value % 100) * 30)
        elif 201 <= value <= 299:
            return str((value % 100) * 4)
        elif 301 <= value <= 399:
            return str(value % 100)
        elif value in [300, 555]:
            return 'Zero'
        elif value in [777, 999]:
            return 'Unknown'
        else:
            return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

# 101~199 = (백의 자리 제거) * 4, 201~299 = 그대로, 888 = "None", 777,999 = "Unknown", BLANK = null
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
            return 'Zero'
        elif value in [777, 999]:
            return 'Unknown'
        elif 101 <= value <= 199:
            return str((value % 100) * 4)
        elif 201 <= value <= 299:
            return str(value % 100)
        else:
            return np.nan

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df





#---------정수 str로 변환하기--------

#1~30을 Str, 88 = 'None' 77,99= 'Unknown', 빈칸 null
def Integer_conversion01(df):
    cols = ['PHYSHLTH', 'MENTHLTH', 'POORHLTH']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 88:
            return 'Zero'
        elif value in [77, 99]:
            return 'Unknown'
        elif 1 <= value <= 30:
            return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


#0~10은 str형으로 변환, 77, 99 : Unknown, 빈칸 null
def Integer_conversion02(df):
    cols = ['JOINPAIN']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value in [77, 99]:
            return 'Unknown'
        elif 0 <= value <= 10:
            return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


#11985 - 122016 > str 형으로 변환, 777777,999999 : Unknown, BLANK : null
def Integer_conversion03(df):
    cols = ['HIVTSTD3']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value in [777777, 999999]:
            return 'Unknown'
        elif 11985 <= value <= 122016:
            return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


#1~76을 Str으로 변환, 88 = None, 77,99= Unknown로 변환, 빈칸 null
def Integer_conversion04(df):
    cols = ['DOCTDIAB', 'FEETCHK', 'DRNK3GE5', 'MAXDRNKS', 'AVEDRNK2']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 88:
            return 'Zero'
        elif value in [77, 99]:
            return 'Unknown'
        elif 1 <= value <= 76:
            return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df

#⚠️ 왜 float로 바뀌는가?
#pandas는 컬럼 내의 값이 다음 두 조건을 만족하면 float64 dtype으로 자동 설정합니다:
#값들이 숫자처럼 생긴 문자열 ('0', '1234' 등)
#None 혹은 np.nan과 같이 결측값이 함께 있음 -지피티

#그럼 0을 'None'으로 바꾸는 방법
#다른 컬럼보니까 8, 888 이런값이 전혀안함 이라는 의미를 가지고 있어서 None으로 처리했음

#0–8590 = str형으로 변환, 99900 = 'Unknown'
def Integer_conversion05(df):
    cols = ['FC60_']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 99900:
            return 'Unknown'
        #elif value == 0:
        #    return 'None'
        elif 0 <= value <= 8590:
            return str(value)

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df



# 88 = "None", 77,99 = "Unknown", 98 = "Other", BLANK는 null, 나머지를 다 str로 변환
def Integer_conversion06(df):
    cols = ['EXRACT11', 'EXRACT21']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 88:
            return 'Zero'
        elif value in [77, 99]:
            return 'Unknown'
        elif value == 98:
            return 'Other'
        else:
            return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 0–98999 = str형으로 변환, 99900 = Unknown
def Integer_conversion07(df):
    cols = ['_DRNKWEK']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 99900:
            return 'Unknown'
        #elif value == 0:
        #    return 'None'
        elif 0 <= value <= 98999:
            return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 0–9999 = str형으로 변환, BLANK = 'Unknown'
def Integer_conversion08(df):
    cols = ['FTJUDA1_', 'FRUTDA1_', 'BEANDAY_', 'GRENDAY_', 'ORNGDAY_', 'VEGEDA1_']

    def convert(value):
        if pd.isnull(value):
            return 'Unknown'
        #elif value == 0:
        #    return 'None'
        elif 0 <= value <= 9999:
            return str(int(value))

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

# 0–98999 = str형으로 변환, 99000 = Unknown, 빈칸 = null
def Integer_conversion11(df):
    cols = ['PAFREQ1_', 'PAFREQ2_', 'STRFREQ_']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 99000:
            return 'Unknown'
        #elif value == 0:
        #    return 'None'
        elif 0 <= value <= 98999:
            return str(int(value))

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

# 2300–29500 = str형으로 변환, 99999 = Unknown, 널도 Unknown (4천개) 임의로 null을 unknown으로 바꿨음(문자형변환위해서)
def Integer_conversion14(df):
    cols = ['WTKG3']

    def convert(value):
        if pd.isnull(value):
            return 'Unknown'
        elif value == 99999:
            return 'Unknown'
        elif 2300 <= value <= 29500:
            return str(int(value))

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




# 정수 = str형으로 변환, 900 = Unknown, 빈칸 = null
def Integer_conversion16(df):
    cols = ['DROCDY3_']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif value == 900:
            return 'Unknown'
        return str(int(value))

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


# 1~76 = str형 변환, 88 = 'None', 77,98,99 = 'Unknown', 빈칸 = null
def Integer_conversion17(df):
    cols = ['CHKHEMO3']

    def convert(value):
        if pd.isnull(value):
            return np.nan
        elif 1 <= value <= 76:
            return str(int(value))
        elif value == 88:
            return 'Zero'
        elif value in [77, 98, 99]:
            return 'Unknown'

    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(convert)

    return df


#------범주형으로 변환-

def categorical(df):
    mappings = {
        'GENHLTH': {1: 'Excellent', 2: 'VeryGood', 3: 'Good', 4: 'Fair', 5: 'Poor', 7: 'Unknown', 9: 'Unknown'},
        'PERSDOC2': {1: 'YesOne', 2: 'YesMulti', 3: 'No', 7: 'Unknown', 9: 'Unknown'},
        'CHECKUP1': {1: 'Within1Y', 2: 'Within2Y', 3: 'Within5Y', 4: 'Over5Y', 7: 'Unknown', 8: 'Never', 9: 'Unknown'},
        'CHOLCHK': {1: 'Within1Y', 2: 'Within2Y', 3: 'Within5Y', 4: 'Over5Y', 7: 'Unknown', 8: 'Never', 9: 'Unknown'},
        'MARITAL': {1: 'Mar', 2: 'Div', 3: 'Wid', 4: 'Sep', 5: 'Nev', 6: 'UnP', 9: 'Unknown'},
        'EDUCA': {1: 'None', 2: 'Elem', 3: 'MidHS', 4: 'HS', 5: 'SomeCol', 6: 'ColGrad', 9: 'Unknown'},
        'RENTHOM1': {1: 'Own', 2: 'Rent', 3: 'Other', 7: 'Unknown', 9: 'Unknown'},
        'EMPLOY1': {1: 'Employed', 2: 'SelfEmp', 3: 'Unemp1Y+', 4: 'Unemp<1Y', 5: 'Homemaker', 6: 'Student', 7: 'Retired', 8: 'Unable', 9: 'Unknown'},
        'INCOME2': {1: '<10K', 2: '10-15K', 3: '15-20K', 4: '20-25K', 5: '25-35K', 6: '35-50K', 7: '50-75K', 8: '75K+', 77: 'Unknown', 99: 'Unknown'},
        'LASTSMK2': {1: 'Within1M', 2: 'Within3M', 3: 'Within6M', 4: 'Within1Y', 5: 'Within5Y', 6: 'Within10Y', 7: 'Y10plus', 8: 'Never', 9: 'Unknown', 99: 'Unknown'},
        'USENOW3': {1: 'Daily', 2: 'Some', 3: 'Never', 7: 'Unknown', 9: 'Unknown'},
        'ARTHSOCL': {1: 'High', 2: 'Moderate', 3: 'No', 7: 'Unknown', 9: 'Unknown'},
        'WHRTST10': {1: 'Private', 2: 'Center', 3: 'Inpatient', 4: 'Clinic', 5: 'Prison', 6: 'DrugTx', 7: 'Home', 8: 'Other', 9: 'ER', 77: 'Unknown', 99: 'Unknown'},
        'SXORIENT': {1: 'Hetero', 2: 'LG', 3: 'Bi', 4: 'Other', 7: 'Unknown', 9: 'Unknown'},
        'MSCODE': {1: 'Central', 2: 'Fringe', 3: 'Suburban', 5: 'NonMSA'},
        '_ASTHMS1': {1: 'Current', 2: 'Former', 3: 'Never', 9: 'Unknown'},
        '_MRACE1': {1: 'White', 2: 'Black', 3: 'Native', 4: 'Asian', 5: 'Pacific', 6: 'Other', 7: 'Multi', 77: 'Unknown', 99: 'Unknown'},
        '_RACE': {1: 'White', 2: 'Black', 3: 'Native', 4: 'Asian', 5: 'PacificI', 6: 'Other', 7: 'Multi', 8: 'Hispani', 9: 'Unknown'},
        '_INCOMG': {1: '<15K', 2: '15-25K', 3: '25-35K', 4: '35-50K', 5: '50K+', 9: 'Unknown'},
        'ACTIN11_': {0: 'Low', 1: 'Moderate', 2: 'Vigorous'},
        'ACTIN21_': {0: 'Low', 1: 'Moderate', 2: 'Vigorous'},
        '_PA300R2': {1: '300plus', 2: '1to299', 3: 'None', 9: 'Unknown'},
        '_PAREC1': {1: 'BothMet', 2: 'AerobicOnly', 3: 'StrengthOnly', 4: 'Neither', 9: 'Unknown'},
        '_BMI5CAT': {1: 'Underweight', 2: 'Normal', 3: 'Overweight', 4: 'Obese'},
        '_PA150R2': {1: '150plus', 2: '1to149', 3: 'None', 9: 'Unknown'},
        'SEX': {1: 'male', 2: 'female'},
        'SMOKDAY2': {1: 'Daily', 2: 'Some', 3: 'Never', 7: 'Unknown', 9: 'Unknown'},
        '_SMOKER3': {1: 'Daily', 2: 'Some', 3: 'Former', 4: 'Never', 9: 'Unknown'},
        '_PACAT1': {1: 'VeryActive', 2: 'Active', 3: 'Insufficient', 4: 'Inactive', 9: 'Unknown'},
        '_AGEG5YR': {1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79', 13: '80+'}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            df[col] = df[col].fillna(value=pd.NA)

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
df.to_csv("BRFSS_2015ver14.csv", index=False)

