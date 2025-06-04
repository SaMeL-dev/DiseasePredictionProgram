import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # ────────────────────────────────────────────────────────────────────
    # 1) CSV 로드 & NaN/비이진(0,1) 제거
    # ────────────────────────────────────────────────────────────────────
    df = pd.read_csv('BRFSS_2015ver14.csv', low_memory=False)

    target_cols = ['BPHIGH4', 'CVDCRHD4', 'CVDSTRK3', 'CHCKIDNY', 'DIABETE3']
    # 타깃에 NaN 있거나 0,1 이외의 값이면 제거
    df = df.dropna(subset=target_cols)
    for col in target_cols:
        df = df[df[col].isin([0, 1])]

    # ────────────────────────────────────────────────────────────────────
    # 2) 피처/타깃 분리
    # ────────────────────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols].copy()
    y = df[target_cols].astype(int)

    # ────────────────────────────────────────────────────────────────────
    # 3) 문자열(범주형) 컬럼 → OrdinalEncoder로 정수 인코딩
    # ────────────────────────────────────────────────────────────────────
    obj_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoder = None
    if obj_cols:
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        X[obj_cols] = encoder.fit_transform(X[obj_cols])

    # (필요 시) 숫자형 피처 NaN을 -1로 채우려면 아래 주석 해제
    # X = X.fillna(-1)

    # ────────────────────────────────────────────────────────────────────
    # 4) Train/Test 분할 (80:20), stratify=y['BPHIGH4'] (대표 타깃 하나)
    # ────────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y['BPHIGH4']
    )

    # ────────────────────────────────────────────────────────────────────
    # 5) 분할 결과 출력
    # ────────────────────────────────────────────────────────────────────
    print(f"훈련 데이터셋 크기: {X_train.shape}")
    print(f"테스트 데이터셋 크기: {X_test.shape}\n")

    # ────────────────────────────────────────────────────────────────────
    # 6) 모델 학습
    # ────────────────────────────────────────────────────────────────────
    base = LGBMClassifier(objective='binary', random_state=42)
    model = MultiOutputClassifier(base, n_jobs=-1)
    model.fit(X_train, y_train)

    # ────────────────────────────────────────────────────────────────────
    # 7) 예측 & 평가 (테스트셋)
    # ────────────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print("=== 최종 테스트 성능 ===")
    for idx, col in enumerate(target_cols):
        acc = accuracy_score(y_test.iloc[:, idx], y_pred[:, idx])
        print(f"\n[{col}] 정확도: {acc:.4f}")
        print(classification_report(
            y_test.iloc[:, idx],
            y_pred[:, idx],
            target_names=['음성(0)', '양성(1)'],
            digits=4
        ))

    # ────────────────────────────────────────────────────────────────────
    # 8) 사용자 입력 단계: 모든 컬럼 이름과 설명을 한꺼번에 출력
    # ────────────────────────────────────────────────────────────────────
    print("\n\n--- 사용자 입력으로 단일 샘플 예측 ---\n")
    print("총 피처 수:", len(feature_cols))
    print("아래는 모든 피처의 이름과 설명입니다. 순서대로 입력해 주세요.\n")

    # (A) 가능한 “한글 설명”을 담아둔 딕셔너리
    #     설명이 없는 컬럼은 “BRFSS 2015 CodeBook 참조”로 표시
    feature_desc = {
    'BPHIGH4':   '의사, 간호사 또는 보건 전문가로부터 고혈압 진단을 받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CVDCRHD4':  '심장 질환(심부전, 관상동맥질환 등) 진단 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CVDSTRK3':  '뇌졸중(Stroke) 진단 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CHCKIDNY':  '신장 질환(Kidney Disease) 진단 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'DIABETE3':  '당뇨병(Diabetes) 진단 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'GENHLTH':   "전반적인 건강 상태에 대한 자가 평가 (예: 1: 'Excellent', 2: 'VeryGood', 3: 'Good', 4: 'Fair', 5: 'Poor', 7/9='Unknown')",
    'PHYSHLTH':  '지난 30일간 신체 건강이 좋지 않았던 일수 (예: "1~30" 사이 정수, 88=Zero, 77/99=Unknown)',
    'MENTHLTH':  '지난 30일간 정신 건강(스트레스·우울·감정 문제 포함)이 좋지 않았던 일수 (예: "1~30" 사이 정수, 88=Zero, 77/99=Unknown)',
    'POORHLTH':  '지난 30일 동안 신체적 또는 정신적 건강 문제로 평소 하던 활동을 못한 일수 (예: "1~30" 사이 정수, 88=Zero, 77/99=Unknown)',
    'HLTHPLN1':  '건강 보험, HMO, Medicare 등 건강 보장 제도가 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'PERSDOC2':  '개인 주치의(Primary Care Provider)가 있다고 생각하는지 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'MEDCOST':   '지난 12개월 동안 비용 문제로 의사의 진료를 받지 못한 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CHECKUP1':  '마지막으로 일반 건강검진(정기신체검사)을 받은 시점 (예: 연도슬래시월 형식, 예) 2023/05; 77/99=Unknown; 빈칸=null)',
    'BPMEDS':    '현재 고혈압 약을 복용 중인지 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'BLOODCHO':  '혈중 콜레스테롤 수치를 측정한 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CHOLCHK':   '지난 5년 이내 콜레스테롤 검사 받은 시점 (예: 연도/월 형식, 예) 2022/11; 77/99=Unknown; 빈칸=null)',
    'TOLDHI2':   '고혈압(High Blood Pressure)을 의사에게 들은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CVDINFR4':  '심근경색(Heart Attack or Myocardial Infarction) 진단 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'ASTHMA3':   '천식(Asthma) 진단을 받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'ASTHNOW':   '현재 천식이 있는지 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CHCSCNCR':  '피부암이외의 암(예: 유방암, 폐암 등) 진단 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CHCOCNCR':  '피부암(Skin Cancer) 진단 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CHCCOPD1':  'COPD(만성폐쇄성폐질환) 진단 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'HAVARTH3':  '의사에게 관절염(Arthritis) 진단받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'ADDEPEV2':  '의사나 보건전문가로부터 우울장애(Depression) 진단받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'MARITAL':   '현재 결혼 상태는 어떻게 되는가? (예: 1=Married, 2=Divorced, 3=Widowed, 4=Separated, 5=Never married, 6=Member of unmarried couple, 7/9=Unknown)',
    'EDUCA':     '마지막 학력 수준 (예: 1=Never attended school/Grade 1–8; 2=Grade 9–11; 3=High school graduate; 4=Some college or technical school; 5=College graduate; 7/9=Unknown)',
    'RENTHOM1':  '현재 거주 형태 (예: 1=Own, 2=Rent, 3=Other; 7/9=Unknown; 빈칸=null)',
    'VETERAN3':  '재향군인 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'EMPLOY1':   '현재 고용 상태 (예: 1=Employed for wages; 2=Self-employed; 3=Out of work >1 year; 4=Out of work <1 year; 5=Homemaker; 6=Student; 7=Retired; 8=Unable to work; 77/99=Unknown)',
    'INCOME2':   '연간 가구 소득 범위 (예: 1=<\$10K; 2=\$10–<\$15K; 3=\$15–<\$20K; …; 9=\$75K or more; 77/99=Unknown)',
    'INTERNET':  '지난 30일간 인터넷 사용 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'QLACTLM2':  '지난 30일간 신체 활동 수준(주당 얼마나 활동을 했는지) (예: 0=No activity; 1=Light; 2=Moderate; 3=Vigorous; 7/9=Unknown)',
    'USEEQUIP':  '신체 활동 시 장비(웨이트 머신, 러닝머신 등) 사용 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'DECIDE':    '일상 활동(자기관리, 업무, 여가)을 결정하거나 계획하는 데 어려움이 있었는지 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'DIFFWALK':  '걷거나 이동하는 데 어려움이 있었는지 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'DIFFDRES':  '자기 옷 입고 벗는 데 어려움이 있었는지 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'DIFFALON':  '혼자서 집 안에서 활동하는 데 어려움이 있었는지 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'SMOKE100':  '평생에 100개비 이상 담배를 피운 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'STOPSMK2':  '현재 담배를 피우고 있지 않은가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'LASTSMK2':  '마지막으로 담배를 피운 횟수 또는 시점 (예: 연도/월 또는 특정 수치; 777/999=Unknown; 888=Zero; 빈칸=null)',
    'USENOW3':   '현재 전자담배를 포함하여 담배 제품 사용 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'LMTJOIN3':  '최근 30일간 운동 모임 등 그룹 활동 참여 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'ARTHDIS2':  '관절염/관절 증상이 현재 취업, 업무량 등에 영향을 주는지 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'ARTHSOCL':  '지난 30일간 관절염으로 인해 사회 활동(쇼핑, 영화, 종교 등)에 방해를 받은 정도 (예: 1=High, 2=Moderate, 3=None, 7/9=Unknown, 빈칸=null)',
    'JOINPAIN':  '통증 관리 프로그램(운동/물리치료 등)에 참여해본 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'FLUSHOT6':  '지난 12개월 동안 독감 예방접종(Flu Shot) 받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'PNEUVAC3':  '폐렴 예방접종(Pneumococcal Vaccine) 받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'HIVTST6':   'HIV 검사받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'HIVTSTD3':  '마지막으로 HIV 검사받은 시점 (예: 연도/월, 777/999=Unknown, 빈칸=null)',
    'WHRTST10':  '마지막 유방암(여성)/전립선암(남성) 검사 시점 (예: 연도/월, 777/999=Unknown, 빈칸=null)',
    'PDIABTST':  '당뇨병 선별검사(예: 공복혈당) 받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'PREDIAB1':  '당뇨병 전단계(Pre-diabetes) 진단 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'INSULIN':   '인슐린 사용 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'BLDSUGAR':  '혈당 수치를 측정하는 빈도(월 단위) (예: “1회”, “2회”; 777/999=Unknown; 888=Zero; 빈칸=null)',
    'FEETCHK2':  '발(Feet) 검사(예: 신경 손상 검사)받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'DOCTDIAB':  '당뇨병 진료 목적으로 최근 병원/의사 방문 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CHKHEMO3':  '헤모글로빈 A1C 검사받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'FEETCHK':   '또 다른 발 검사 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'DIABEYE':   '당뇨망막증(Diabetic Eye Disease) 검사받은 적이 있는가? (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'CIMEMLOS':  '혈관내막 또는 미세혈관 이상 여부(예: 검사 결과) (예: 구체적 검사 수치, 777/999=Unknown; 빈칸=null)',
    'SXORIENT':  '성적 지향(Sexual Orientation) (예: 1=Heterosexual, 2=Homosexual, 3=Bisexual, 4=Other, 7/9=Unknown)',
    'TRNSGNDR':  '트랜스젠더 여부 (예: 1=Yes, 2=No, 7/9=Unknown; 빈칸=null)',
    'MSCODE':    '조사 실시 주(State) 코드 (예: 01=Alabama, 02=Alaska, …; 99=Unknown; 빈칸=null)',
    '_RFHLTH':   '건강 상태 자가평가(숫자로 재코딩된 값) (예: 빈칸=null; 1=Excellent, 2=VeryGood, 3=Good, 4=Fair, 5=Poor)',
    '_HCVU651':  'Cervical Cancer Screening 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    '_CHOLCHK':  '콜레스테롤 검사 여부(재코딩 변수) (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    '_RFCHOL':   '고콜레스테롤 진단 여부(재코딩 변수) (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    '_LTASTH1':  '최근 12개월간 천식 발작 여부(예: 빈칸=null; 1=Yes, 2=No, 7/9=Unknown)',
    '_CASTHM1':  '천식 치료를 받은 적이 있는가? (예: 빈칸=null; 1=Yes, 2=No, 7/9=Unknown)',
    '_ASTHMS1':  '최근 12개월간 천식 증상(예: 호흡 곤란) 여부 (예: 빈칸=null; 1=Yes, 2=No, 7/9=Unknown)',
    '_DRDXAR1':  '관절염 진단 여부(재코딩 변수) (예: 빈칸=null; 1=Yes, 2=No, 7/9=Unknown)',
    '_MRACE1':   '인종/민족 자가 보고(재코딩 변수) (예: 예시: 1=White, 2=Black, 3=Native, 4=Asian, 5=PacIslander, 6=Other, 7=Multi, 8=Hispanic, 9=Unknown; 빈칸=null)',
    '_HISPANC':  '히스패닉/라티노 여부(재코딩 변수) (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    '_RACE':     '상세 인종 보고 (예: 1=White, 2=Black, 3=Native, 4=Asian, 5=PacificI, 6=Other, 7=Multi, 8=Hispani, 9=Unknown)',
    '_RACEG21':  '비히스패닉 백인 vs 기타 인종 그룹 구분 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    '_INCOMG':   '연간 가구 소득 재코딩(binned) (예: 1=<\$10K, 2=\$10–<\$15K, …, 9=\$75K+, 77/99=Unknown)',
    'FC60_':     '최근 60일간 신체 활동 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'PASTAE1':   '과거 에어로빅 활동 참여 여부 (예: 빈칸=null; 1=Yes, 2=No, 7/9=Unknown)',
    '_FLSHOT6':  '지난 12개월 독감 예방접종 여부(재코딩 변수) (예: 빈칸=null; 1=Yes, 2=No, 7/9=Unknown)',
    '_PNEUMO2':  '폐렴 백신 접종 여부(재코딩 변수) (예: 빈칸=null; 1=Yes, 2=No, 7/9=Unknown)',
    '_AIDTST3':  'HIV 검사 여부(재코딩 변수) (예: 빈칸=null; 1=Yes, 2=No, 7/9=Unknown)',
    'EXRACT11':  '첫 번째 주요 신체 활동 유형 코드 (예: 01=Walking, 02=Running, 03=Cycling, …, 88=None; 77/99=Unknown; 빈칸=null)',
    'EXEROFT1':  '첫 번째 주요 신체 활동 빈도(주당 일수) (예: 0~7; 77/99=Unknown; 빈칸=null)',
    'EXERHMM1':  '첫 번째 주요 신체 활동 지속 시간(한 번에 몇 분) (예: 10~300, 777/999=Unknown; 빈칸=null)',
    'EXRACT21':  '두 번째 주요 신체 활동 유형 코드 (예: 위와 동일; 빈칸=null)',
    'EXEROFT2':  '두 번째 주요 신체 활동 빈도(주당 일수) (예: 0~7; 77/99=Unknown; 빈칸=null)',
    'EXERHMM2':  '두 번째 주요 신체 활동 지속 시간(분) (예: 10~300; 777/999=Unknown; 빈칸=null)',
    'DRNK3GE5':  '과거 폭음 여부 (예: 남성: 한 번에 5잔↑, 여성: 4잔↑ 마신 적이 있는가?; 빈칸=null; 1=no, 2=yes, 7/9=Unknown)',
    '_RFBING5':  '폭음 여부(computed) (예: 재코딩된 이진 변수; 1=no, 2=yes, 7/9=Unknown; 빈칸=null)',
    'DRNKWEK':   '지난 30일간 음주 일수 (예: 0~30; 777/999=Unknown; 빈칸=null)',
    '_RFDRHV5':  '고위험 음주 여부(Heavy Alcohol Consumption, computed) (예: 남성>주당14잔, 여성>주당7잔 시 Yes; 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'FTJUDA1_':  '하루 과일 주스(오렌지/사과 등) 섭취 횟수 (예: 0.25=¼회, 1=1회, 2.5=2.5회, 777/999=Unknown; 빈칸=null)',
    'FRUTDA1':   '하루 과일(사과, 배 등) 섭취 횟수 (예: 0.25=¼회, 1=1회; 777/999=Unknown; 빈칸=null)',
    'FVBEANS':   '하루 콩(또는 렌틸콩 등) 섭취 횟수 (예: 0.25=¼회; 777/999=Unknown; 빈칸=null)',
    'FVGREEN':   '하루 짙은 녹색 채소 섭취 횟수(예: 0.25=¼회; 777/999=Unknown; 빈칸=null)',
    'FVORANG':   '하루 주황색 채소(당근 등) 섭취 횟수(예: 0.25=¼회; 777/999=Unknown; 빈칸=null)',
    'VEGEDA1':   '하루 기타 채소 섭취 횟수(예: 0.25=¼회; 777/999=Unknown; 빈칸=null)',
    '_FRUTSUM':  '하루 과일 섭취 총합 (예: FTJUDA1 + FRUTDA1 + FVBEANS; 777/999=Unknown; 빈칸=null)',
    '_VEGESUM':  '하루 채소 섭취 총합 (예: FVGREEN + FVORANG + VEGEDA1; 777/999=Unknown; 빈칸=null)',
    '_FRTLT1':   '하루 과일 1회 이상 섭취 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    '_VEGLT1':   '하루 채소 1회 이상 섭취 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'ACTIN11_':  "첫 번째 주요 활동(EXRACT11) 강도 수준 (예: 0=Low, 1=Moderate, 2=Vigorous, 빈칸=null)",
    'ACTIN21_':  "두 번째 주요 활동(EXRACT21) 강도 수준 (예: 0=Low, 1=Moderate, 2=Vigorous, 빈칸=null)",
    'PADUR1_':   '첫 번째 신체 활동 세션 지속 시간(분) (예: 10~300; 777/999=Unknown; 빈칸=null)',
    'PADUR2_':   '두 번째 신체 활동 세션 지속 시간(분) (예: 10~300; 777/999=Unknown; 빈칸=null)',
    'PAFREQ1_':  '첫 번째 신체 활동 빈도(주당) (예: 0~7; 77/99=Unknown; 빈칸=null)',
    'PAFREQ2_':  '두 번째 신체 활동 빈도(주당) (예: 0~7; 77/99=Unknown; 빈칸=null)',
    '_MINAC11':  '첫 번째 활동별 총 분(분 단위)(예: PAHOURS1*60 + PADUR1; 777/999=Unknown; 빈칸=null)',
    '_MINAC21':  '두 번째 활동별 총 분(분 단위)(예: PAHOURS2*60 + PADUR2; 777/999=Unknown; 빈칸=null)',
    'STRFREQ_':  '강도별 신체 활동 빈도(예: 부족한 강도 계산 이후 값; 777/999=Unknown; 빈칸=null)',
    'PAMIN11_':  '첫 번째 활동별 주당 최소 권장 시간(분) 충족 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'PAMIN21_':  '두 번째 활동별 주당 최소 권장 시간(분) 충족 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'PAVIG11_':  '첫 번째 활동별 집중 강도(예: 빈칸=null; 1=low/mod, 2=vigorous, 7/9=Unknown)',
    'PAVIG21_':  '두 번째 활동별 집중 강도(예: 빈칸=null; 1=low/mod, 2=vigorous, 7/9=Unknown)',
    '_PAINDX1':  '신체 활동 지수(Physical Activity Index; 계산된 값) (예: 1=Low, 2=Moderate, 3=High; 빈칸=null)',
    '_PA300R2':  '총 운동량(300 MET-minutes와 비교) 충족 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    '_PASTRNG':  '스트렝스(강도) 운동 수행 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    '_PAREC1':   '유산소(Aerobic) 운동 수행 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    '_BMI5CAT':  '체질량 지수(BMI) 카테고리 (예: 1=Underweight, 2=Normal, 3=Overweight, 4=Obese, 빈칸=null)',
    'PA150R2':   '유산소 운동 150분 이상/주 권장 충족 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'SEX':       '성별 (예: 1=Male, 2=Female, 빈칸=null)',
    'PREGNANT':  '임신 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'SMOKDAY2':  '마지막으로 담배를 핀 이후 경과 일수 (예: "0~999"; 777/999=Unknown; 빈칸=null)',
    'ALCDAY5':   '지난 30일간 음주 일수 (예: 0~30; 777/999=Unknown; 빈칸=null)',
    'AVEDRNK2':  '평균 주당 음주량(음주 잔 수) (예: 0~21; 777/999=Unknown; 빈칸=null)',
    'MAXDRNKS':  '한 번에 마시는 최대 음주 잔 수 (예: 0~30; 777/999=Unknown; 빈칸=null)',
    'FRUITJU1':  '하루 과일 주스 섭취 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'FRUIT1':    '하루 과일 섭취 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'FVBEANS':   '하루 콩 섭취 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'FVGREEN':   '하루 짙은 녹색 채소 섭취 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'FVORANG':   '하루 주황색 채소 섭취 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'VEGETAB1':  '하루 기타 채소 섭취 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'EXERANY2':  '지난 30일간 어떠한 신체 활동이라도 했는지 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'STRENGTH':  '근력 운동(웨이트 트레이닝 등) 수행 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    '_RFHYPE5':  '고혈압 이력 및 치료 여부 (예: 빈칸=null; 1=Yes, 2=No, 7/9=Unknown)',
    '_MICHD':    '관상동맥 심장질환(Coronary Heart Disease) 이진 변수 (예: 빈칸=null; 1=Yes, 2=No, 7/9=Unknown)',
    'HTM4':      '키(cm) (예: 150~200; 777/999=Unknown; 빈칸=null)',
    'WTKG3':     '몸무게(kg) (예: 30~200; 777/999=Unknown; 빈칸=null)',
    '_BMI5':     '계산된 BMI 값(Body Mass Index) (예: 18.5~40; 777/999=Unknown; 빈칸=null)',
    '_SMOKER3':  '흡연 상태(재코딩된 값) (예: 1=Current Smoker, 2=Former Smoker, 3=Never Smoked, 7/9=Unknown)',
    'DRNKANY5':  '지난 30일간 음주 여부 (예: 빈칸=null; 1=yes, 2=no, 7/9=Unknown)',
    'DROCDY3_':  '지난 30일간 하루 평균 음주 잔 수 (예: 0~99; 777/999=Unknown; 빈칸=null)',
    '_FRUITEX':  '일일 과일 섭취량(g, 계산값) (예: 0~999; 777/999=Unknown; 빈칸=null)',
    '_VEGETEX':  '일일 채소 섭취량(g, 계산값) (예: 0~999; 777/999=Unknown; 빈칸=null)',
    '_PACAT1':   '신체 활동 카테고리 (예: 1=Inactive, 2=Insufficiently Active, 3=Active; 빈칸=null)',
    '_AGEG5YR':  '5년 단위 연령 그룹 (예: 1=18–24, 2=25–29, 3=30–34, …, 13=80+; 빈칸=null)',
    }

    # (B) 숫자형 피처 최소/최대/예시값 계산
    numeric_info = {}
    for col in feature_cols:
        if col not in obj_cols:
            col_num = pd.to_numeric(df[col], errors='coerce')
            if col_num.notna().any():
                min_val = col_num.min()
                max_val = col_num.max()
                example = int((min_val + max_val) / 2)
                numeric_info[col] = (min_val, max_val, example)
            else:
                numeric_info[col] = (None, None, None)

    # (C) 범주형 피처의 카테고리 목록 수집
    categorical_info = {}
    if obj_cols and encoder is not None:
        for idx, col in enumerate(obj_cols):
            cats = list(encoder.categories_[idx])
            categorical_info[col] = cats

    # ────────────────────────────────────────────────────────────────────
    # 9) 사용자 입력 루프: “모든 컬럼”과 “설명”을 출력한 뒤, 값을 차례대로 입력받음
    # ────────────────────────────────────────────────────────────────────
    new_data = {}
    for col in feature_cols:
        # 이 컬럼의 한글 설명이 있는지 확인, 없으면 “CodeBook 참조” 안내
        desc = feature_desc.get(col, "설명 없음(‘BRFSS 2015 CodeBook’ 참조)")

        if col in obj_cols:
            # 범주형 컬럼: 가능한 카테고리 예시 보여주기
            cats = categorical_info.get(col, [])
            example_cats = cats[:5]
            prompt = (
                f"  - [{col}] {desc}\n"
                f"      가능한 카테고리 예시: {example_cats} … 총 {len(cats)}개\n"
                f"      (문자열 그대로 입력, 예: {example_cats[0] if example_cats else '값'})\n"
                f"      → "
            )
            val = input(prompt)
            new_data[col] = val
        else:
            # 숫자형 컬럼: 범위와 예시값 안내
            min_val, max_val, example = numeric_info.get(col, (None, None, None))
            if min_val is not None:
                prompt = (
                    f"  - [{col}] {desc}\n"
                    f"      범위: [{int(min_val)}, {int(max_val)}], 예시값: {example}\n"
                    f"      (숫자만 입력)\n"
                    f"      → "
                )
            else:
                prompt = (
                    f"  - [{col}] {desc}\n"
                    f"      (유효한 숫자 정보 없음, 건너뛰려면 Enter)\n"
                    f"      → "
                )
            val = input(prompt)
            new_data[col] = val

    # ────────────────────────────────────────────────────────────────────
    # 10) 입력값 DataFrame으로 변환 & 전처리
    # ────────────────────────────────────────────────────────────────────
    new_df = pd.DataFrame([new_data])

    # (1) 범주형 인코딩
    if obj_cols and encoder is not None:
        new_df[obj_cols] = encoder.transform(new_df[obj_cols])

    # (2) 숫자형 변환
    for col in feature_cols:
        if col not in obj_cols:
            try:
                new_df[col] = pd.to_numeric(new_df[col])
            except:
                new_df[col] = pd.NA

    # (3) NaN 발생 시 -1로 채움
    if new_df.isna().any(axis=None):
        new_df = new_df.fillna(-1)

    # ────────────────────────────────────────────────────────────────────
    # 11) predict_proba로 “양성 확률” 계산 → 출력
    # ────────────────────────────────────────────────────────────────────
    probas_list = model.predict_proba(new_df)
    print("\n=== 사용자 입력 샘플 예측 결과(확률) ===")
    for idx, col in enumerate(target_cols):
        probas = probas_list[idx][0]
        prob_positive = probas[1]
        prob_negative = probas[0]
        print(f"  - [{col}] 양성 확률: {prob_positive * 100:.2f}%  (음성 확률: {prob_negative * 100:.2f}%)")

    print("\n예측이 완료되었습니다.")

if __name__ == "__main__":
    main()
