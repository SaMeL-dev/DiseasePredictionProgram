import pandas as pd
#순서 정리
df = pd.read_excel("BRFSS_2015ver7.xlsx")

desired_columns = [
    'BPHIGH4', 'CVDCRHD4', 'CVDSTRK3', 'CHCKIDNY', 'DIABETE3', '_MICHD',

    '_AGEG5YR', 'SEX', 'PREGNANT', 'HTM4', 'WTKG3', '_BMI5',

    '_FRUITEX', 'FRUIT1', 'FRUITJU1', '_VEGETEX', 'FVGREEN', 'FVORANG', 'VEGETAB1', 'FVBEANS',

    '_PACAT1_1.0', '_PACAT1_2.0', '_PACAT1_3.0', '_PACAT1_4.0', 'EXERANY2', 'STRENGTH',

    'SMOKDAY2_1.0', 'SMOKDAY2_2.0', 'SMOKDAY2_3.0',

    '_SMOKER3_1.0', '_SMOKER3_2.0', '_SMOKER3_3.0', '_SMOKER3_4.0',

    'DRNKANY5', 'ALCDAY5', 'AVEDRNK2', 'MAXDRNKS', 'DROCDY3_'
]

df[desired_columns].to_excel("BRFSS_2015ver8.xlsx", index=False, engine="openpyxl")
