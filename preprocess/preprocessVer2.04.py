import pandas as pd
#null이 많은 컬럼 데이터상에서 삭제
df = pd.read_csv("BRFSS_2015ver15.csv")
#null의 수
#ASTHNOW:382047, INSULIN:412032, ARTHSOCL:305162,
# JOINPAIN:307728, INSULIN:412032, BLDSUGAR:412035, DOCTDIAB:412,038
#CHKHEMO3:412039, FEETCHK:412317, DIABEYE:412040, CIMEMLOS:324727
#SXORIENT:274459, TRNSGNDR:274549 ,


columns_to_drop = ['ASTHNOW', 'ARTHSOCL', 'JOINPAIN', 'INSULIN', 'BLDSUGAR',
                   'DOCTDIAB','CHKHEMO3', 'FEETCHK', 'DIABEYE','CIMEMLOS',
                   'SXORIENT', 'TRNSGNDR']

df = df.drop(columns=columns_to_drop, errors='ignore')

df = pd.read_csv("BRFSS_2015ver16.csv", low_memory=False)

print("삭제 완료")