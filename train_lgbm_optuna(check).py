import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    # 1) CSV 로드 & NaN/비이진(0,1) 제거
    df = pd.read_csv('data/BRFSS_2015ver15.csv', low_memory=False)
    target_cols = ['BPHIGH4', 'CVDCRHD4', 'CVDSTRK3', 'CHCKIDNY', 'DIABETE3']
    df = df.dropna(subset=target_cols)
    for col in target_cols:
        df = df[df[col].isin([0, 1])]

    # 2) 피처/타깃 분리
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols].copy()
    y = df[target_cols].astype(int)

    # 3) 문자열(범주형) 컬럼 → OrdinalEncoder로 정수 인코딩
    obj_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoder = None
    if obj_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[obj_cols] = encoder.fit_transform(X[obj_cols])

    # 4) Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y['BPHIGH4']
    )

    # 5) 학습 및 평가
    print(f"훈련 데이터셋 크기: {X_train.shape}")
    print(f"테스트 데이터셋 크기: {X_test.shape}\n")
    base = LGBMClassifier(objective='binary', random_state=42)
    model = MultiOutputClassifier(base, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("=== 최종 테스트 성능 ===")
    for idx, col in enumerate(target_cols):
        acc = accuracy_score(y_test.iloc[:, idx], y_pred[:, idx])
        print(f"\n[{col}] 정확도: {acc:.4f}")
        print(classification_report(
            y_test.iloc[:, idx], y_pred[:, idx], target_names=['음성(0)', '양성(1)'], digits=4
        ))

    # 6) 사용자 입력
    print("\n\n--- 사용자 입력 예측 ---\n")
    print(f"총 피처 수: {len(feature_cols)}\n")

    # A) 설명용 딕셔너리
    feature_desc = {
        'GENHLTH':   "전반적인 건강 상태에 대한 자가 평가 (예: 1: 'Excellent', 2: 'VeryGood', 3: 'Good', 4: 'Fair', 5: 'Poor', 7/9='null')",
        'PHYSHLTH':  "지난 30일간 신체 건강이 좋지 않았던 일수 (예: 1~30 사이 정수, 88=Zero, 77/99=UnkNown)",
        'MENTHLTH':  "지난 30일간 정신 건강(스트레스, 우울, 감정 문제 포함s던 활동을 하지 못한 일수 (예: 1~30 사이 정수, 88=Zero, 77/99=UnkNown)",
        'POORHLTH': "지난 30일 동안 신체적 또는 정신적 건강 문제로 평소 하던 활동 (자가관리, 업무, 여가 등)을 하지 못한 일수 (예: 1~30=일수, 88=Zero, 77/99 = Unknown)",
        'HLTHPLN1':  "건강 보험, HMO(선불 건강관리), Medicare, 인디언 보건 서비스 등 어떤 형태의 건강 보장 제도를 갖고 있는가? (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'PERSDOC2':  "개인 주치의 또는 건강관리 제공자가 있다고 생각하는지 여부 (예: 1=Yesone, 2=Morethan, 3=No, 7/9=UnkNown)",
        'MEDCOST':   "지난 12개월 동안 비용 문제로 의사의 진료를 받지 못한 적이 있는지 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'CHECKUP1':  "마지막으로 정기 건강검진을 받은 시점은 언제인가? (예: 1=Within1Y, 2=Within2Y, 3=Within5Y, 4=Over5Y, 8=Never, 7/9=UnkNown; 빈칸=null)",
        'BPMEDS':    "현재 고혈압 약을 복용 중인지 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'BLOODCHO':  "혈중 콜레스테롤 수치를 측정한 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'CHOLCHK':   "마지막으로 혈중 콜레스테롤 수치를 측정한 시점은 언제인가? (예: 1=Within1Y, 2=Within2Y, 3=Within5Y, 4=Over5Y, 8=Never, 7/9=UnkNown; 빈칸=null)",
        'TOLDHI2':   "보건 전문가로부터 혈중 콜레스테롤 수치가 높다고 들은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'CVDINFR4':  "보건 전문가로부터 심근경색(심장마비)을 진단받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'ASTHMA3':   "보건 전문가로부터 천식 진단을 받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'ASTHNOW':   "현재 천식이 있는지 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'CHCSCNCR':  "보건 전문가로부터 피부암(스킨 캔서) 진단을 받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'CHCOCNCR':  "보건 전문가로부터 피부암을 제외한 다른 암을 진단받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'CHCCOPD1':  "보건 전문가로부터 COPD, 폐기종, 기관지염 등 진단을 받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'HAVARTH3':  "보건 전문가로부터 관절염 진단을 받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'ADDEPEV2':  "보건 전문가로부터 우울 장애 진단을 받은 적이 있는가? (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'MARITAL':   "혼인 상태 (예: 1=Mar, 2=Div, 3=Wid, 4=Sep, 5=Nev, 6=UnP, 9=UnkNown)",
        'EDUCA':     "최고로 수료한 학년 또는 학력 수준 (예: 1=None, 2=Elem, 3=MidHS, 4=HS, 5=SomeCol, 6=ColGrad, 9=UnkNown)",
        'RENTHOM1':  "주택 소유 형태 (예: 1=Own, 2=Rent, 3=Other; 7/9=UnkNown)",
        'VETERAN3':  "군 복무 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'EMPLOY1':   "현재 고용 상태 (예: 1=Employed, 2=SelfEmp, 3=Unemp1Y+, 4=Unemp<1Y, 5=Homemaker, 6=Student, 7=Retired, 8=Unable, 9=UnkNown)",
        'INCOME2':   "가구 연간 소득 (예: 1=<10K, 2=10-15K, 3=15-20K, 4=20-25K, 5=25-35K, 6=35-50K, 7=50-75K, 8=75K+, 77/99=UnkNown, BLANK=null)",
        'INTERNET':  "지난 30일 이내 인터넷 사용 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'QLACTLM2':  "신체적/정신적/감정적 문제로 활동 제한 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'USEEQUIP':  "지팡이/휠체어/특수장비 사용 필요 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'DECIDE':    "집중/기억/의사결정 어려움 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'DIFFWALK':  "걷기/계단 오르기 어려움 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'DIFFDRES':  "옷 입기/목욕하기 어려움 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'DIFFALON':  "병원 방문/쇼핑 등 혼자 볼일 보기 어려움 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'SMOKE100':  "생애 동안 담배 100개비 이상 흡연 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'STOPSMK2':  "지난 12개월 금연 시도 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'LASTSMK2':  "마지막 흡연 시기 (예: 1=Within1M, 2=Within3M, 3=Within6M, 4=Within1Y, 5=Within10Y, 7=Y10plus, 8=Never, 9/99=UnkNown, 빈칸=null)",
        'USENOW3':   "현재 씹는 담배/스너프/스누스 사용 여부 (예: 1=Daily, 2=Some, 3=Never, 7/9=UnkNown, BLANK=null)",
        'LMTJOIN3':  "현재 관절염/관절 증상으로 활동 제한 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'ARTHDIS2':  "관절염/관절 증상이 취업/업무에 미치는 영향 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'ARTHSOCL':  "지난 30일 관절염/관절 증상으로 쇼핑/영화/종교/사교 활동에 방해 정도 (예: 1=High, 2=Moderate, 3=No, 7/9=UnkNown, 빈칸=null)",
        'JOINPAIN':  "지난 30일 관절 통증 정도 (예: 0~10 숫자, 7/9=UnkNown, 빈칸=null)",
        'FLUSHOT6':  "지난 12개월 독감 예방접종 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'PNEUVAC3':  "폐렴 백신 접종 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'HIVTST6':   "HIV 검사 받은 적 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'HIVTSTD3':  "마지막 HIV 검사 시기 (월/년, 예: 2015년 5월=052015, 777777/999999=UnkNown, BLANK=null)",
        'WHRTST10':  "마지막 HIV 검사 장소 (예: 1=Private, 2=Center, 3=Inpatient, 4=Clinic, 5=Prison, 6=DrugTx, 7=Home, 8=Other, 9=ER, 77/99=UnkNown)",
        'PDIABTST':  "지난 3년 내 고혈당/당뇨병 검사 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'PREDIAB1':  "당뇨 전단계 진단 여부 (예: 빈칸=null, 1/2=Yes, 3=No, 7/9=UnkNown)",
        'INSULIN':   "현재 인슐린 사용 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'BLDSUGAR':  "혈당 확인 빈도 (예: 월간횟수, 777/999=UnkNown, 888=Zero, 빈칸=null)",
        'FEETCHK2':  "발 상태 확인 빈도 (예: 월간횟수, 777/999=UnkNown, 888=Zero, 빈칸=null)",
        'DOCTDIAB':  "지난 12개월 당뇨병 진료 횟수 (예: 1~76 직접 입력, 88=Zero, 77/99=UnkNown, 빈칸=null)",
        'CHKHEMO3':  "수최근 12개월당화혈색소(A1C) 검사 횟수 (예: 1~76 직접 입력, 88=한 번도 없음, 77/98/99=모름, 빈칸=null)",
        'FEETCHK':   "지난 12개월 발 검사 횟수 (예: 1~76 직접 입력, 88=Zero, 77/99=UnkNown, 빈칸=null)",
        'DIABEYE':   "당뇨 합병증(망막병증) 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'CIMEMLOS':  "지난 12개월 기억력 저하/혼란 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'SXORIENT':  "성적 지향 (예: 1=이성애자, 2=레즈비언/게이, 3=양성애자, 4=기타, 7/9=UnkNown)",
        'TRNSGNDR':  "트랜스젠더 여부 (예: 1/2/3=Yes, 4=No, 7/9=UnkNown)",
        'MSCODE':    "거주 지역 대도시권 상태 (예: 1=MSA 중심도시, 2=MSA 주변, 3=MSA 교외, 5=비MSA, BLANK=null)",
        '_RFHLTH':   "전반적 건강 상태 (예: 1=좋음/매우좋음/보통, 2=나쁨/매우나쁨, 7/9=UnkNown)",
        '_HCVU651':  "18~64세 응답자 건강보험 여부 (예: 1=있음, 2=없음, 7/9=UnkNown)",
        '_CHOLCHK':  "최근 5년 내 콜레스테롤 검사 여부 (예: 1=예, 2=아니요, 7/9=UnkNown)",
        '_RFCHOL':   "콜레스테롤 수치 높음 진단 여부 (예: 1=아니요, 2=예, 7/9=UnkNown)",
        '_LTASTH1':  "성인 후 천식 진단 여부 (예: 1=아니요, 2=예, 7/9=UnkNown)",
        '_CASTHM1':  "현재 천식 여부 (예: 1=아니요, 2=예, 7/9=UnkNown)",
        '_ASTHMS1':  "현재/과거 천식 여부 (예: 1=현재, 2=과거, 3=없음, 9=UnkNown)",
        '_DRDXAR1':  "관절염 진단 여부 (예: 1=진단받음, 2=진단받지않음, 7/9=UnkNown)",
        '_MRACE1':   "다인종 인종 분류 (예: 1=White, 2=Black, 3=Native, 4=Asian, 5=Pacific, 6=Other, 7=Multi, 77/99=UnkNown)",
        '_HISPANC':  "히스패닉/라티노 여부 (예: 1=예, 2=아니요, 7/9=UnkNown)",
        '_RACE':     "인종/민족 범주 (예: 1=White, 2=Black, 3=Native, 4=Asian, 5=PacificI, 6=Other, 7=Multi, 8=Hispanic, 9=UnkNown)",
        '_RACEG21':  "비히스패닉 백인 여부 (예: 1=Yes, 2=No, 7/9=UnkNown, 빈칸=null)",
        '_INCOMG':   "연간 가구 소득(재코딩) (예: 1=<15K, 2=15–25K, 3=25–35K, 4=35–50K, 5=50K+, 9=UnkNown)",
        'FC60_':     "기능적 수행 능력 추정치 (예: 0–8590, 99900=UnkNown)",
        '_PASTAE1':  "운동 권장 기준 충족 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        '_FLSHOT6':  "65세 이상 독감 예방접종 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        '_PNEUMO2':  "65세 이상 폐렴 백신 접종 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        '_AIDTST3':  "HIV 검사 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'EXRACT11':  "첫 번째 주 활동 유형 코드 (예: 1=액티브 게이밍, 2=에어로빅, 3=배낭여행, 4=배드민턴, 5=농구, 6=실내 자전거, 7=자전거, 8=카누/카약, 9=볼링, 10=복싱, 11=맨몸 체조, 12=카누 경기, 13=목공, 14=춤, 15=일립티컬/EFX, 16=낚시, 17=프리스비, 18=원예, 19=골프(전동), 20=골프(도보), 21=핸드볼, 22=하이킹, 23=하키, 24=승마, 25=대형 사냥, 26=소형 사냥, 27=인라인 스케이트, 28=조깅, 29=라크로스, 30=등산, 31=잔디 깎기, 32=패들볼, 33=집 페인트칠, 34=필라테스, 35=라켓볼, 36=낙엽 긁기, 37=달리기, 38=암벽등반, 39=줄넘기, 40=로잉 머신, 41=럭비, 42=스쿠버 다이빙, 43=스케이트보드, 44=스케이트, 45=썰매 탑승, 46=스노클링, 47=눈 치우기(기계), 48=눈 치우기(삽), 49=스키, 50=스노슈잉, 51=축구, 52=소프트볼/야구, 53=스쿼시, 54=계단 오르기, 55=낚시(장화), 56=서핑, 57=수영, 58=레인 수영, 59=탁구, 60=태극권, 61=테니스, 62=터치 풋볼, 63=배구, 64=걷기, 66=수상스키, 67=웨이트 트레이닝, 68=레슬링, 69=요가, 71=돌봄 활동, 72=농장/목장 일, 73=가사 활동, 74=무술/가라데, 75=상체 자전거(에르고), 76=마당일, 98=기타, 77=모름, 99=응답거부, BLANK=null)",
        'EXEROFT1':  "지난 한 달 동안 이 활동(걷기, 달리기, 조깅, 수영)을 주당 또는 월당 몇 회 했습니까? (예: 월간횟수, 777/999=UnkNown, BLANK=null)",
        'EXERHMM1':  "EXRACT11에서 선택한 운동 한 번 할 때 지속 시간 (예: 분 단위, 777=모름, 999=응답거부, BLANK=null)",
        'EXRACT21':  "지난 한 달 동안 두 번째로 많이 한 신체 활동 유형 코드 (예: 38=암벽등반, 39=줄넘기, 40=로잉 머신, 41=럭비, 42=스쿠버 다이빙, 43=스케이트보드, 44=스케이트, 45=썰매 탑승, 46=스노클링, 47=눈 치우기(제설기), 48=눈 치우기(삽), 49=스키, 50=스노슈잉, 51=축구, 52=소프트볼/야구, 53=스쿼시, 54=계단 오르기, 55=낚시(물속), 56=서핑, 57=수영, 58=레인 수영, 59=탁구, 60=태극권, 61=테니스, 62=터치 풋볼, 63=배구, 64=걷기, 66=수상스키, 67=웨이트 트레이닝, 68=레슬링, 69=요가, 71=돌봄 활동, 72=농장/목장 일, 73=가사 활동, 74=가라데/무술, 75=상체 자전거(에르고), 76=마당일, 77=모름, 88=다른 활동 없음, 98=기타, 99=응답거부, BLANK=null)",
        'EXEROFT2':  "지난 한 달 동안 두 번째 활동을 주당/월당 몇 회 했는가? (예: 월간횟수, 777/999=UnkNown, BLANK=null)",
        'EXERHMM2':  "두 번째 활동 한 번 할 때 지속 시간 (예: 분 단위, 777=모름, 999=응답거부, BLANK=null)",
        'DRNK3GE5':  "지난 30일 폭음 횟수 (남성: 한 번에 5잔 이상, 여성: 4잔 이상) (예: 1–76=횟수, 88=없음, 77/99=UnkNown, BLANK=null)",
        '_RFBING5':  "폭음 여부 (예: 1=아니요, 2=예, 7/9=UnkNown)",
        '_DRNKWEK':  "주당 음주 횟수 (예: 0=음주하지 않음, 1–98999=횟수, 99900=UnkNown)",
        '_RFDRHV5':  "고위험 음주 여부 (예: 1=No, 2=Yes, 9=UnkNown)",
        'FTJUDA1_':  "과일 주스 1일 섭취 횟수 (예: 125=1.25회, BLANK=null)",
        'FRUTDA1_':  "과일 섭취 횟수(계산) (예: 250=2.50회, BLANK=null)",
        'BEANDAY_':  "콩류 섭취 횟수(계산) (예: 175=1.75회, BLANK=null)",
        'GRENDAY_':  "진한 녹색 채소 섭취 횟수(계산) (예: 300=3.00회, BLANK=null)",
        'ORNGDAY_':  "주황색 채소 섭취 횟수(계산) (예: 250=2.50회, BLANK=null)",
        'VEGEDA1_':  "기타 채소 섭취 횟수(계산) (예: 325=3.25회, BLANK=null)",
        '_FRUTSUM':  "전체 과일 섭취 횟수(계산) (예: 450=4.50회, BLANK=null)",
        '_VEGESUM':  "전체 채소 섭취 횟수(계산) (예: 1025=10.25회, BLANK=null)",
        '_FRTLT1':   "하루 과일 1회 이상 섭취 여부 (예: 1=Yes, 2=No, 7/9=UnkNown, BLANK=null)",
        '_VEGLT1':   "하루 채소 1회 이상 섭취 여부 (예: 1=Yes, 2=No, 7/9=UnkNown, BLANK=null)",
        'ACTIN11_':  "첫 번째 활동 운동 강도 (예: 0=Low, 1=Moderate, 2=Vigorous, BLANK=null)",
        'ACTIN21_':  "두 번째 활동 운동 강도 (예: 0=Low, 1=Moderate, 2=Vigorous, BLANK=null)",
        'PADUR1_':   "첫 번째 활동 주당 총 지속 시간(분) (예: 0=없음, 1–599=분, BLANK=null)",
        'PADUR2_':   "두 번째 활동 주당 총 지속 시간(분) (예: 0=없음, 1–599=분, BLANK=null)",
        'PAFREQ1_':  "첫 번째 활동 주당 빈도(계산) (예: 12000=12.000회, 99000=UnkNown, BLANK=null)",
        'PAFREQ2_':  "두 번째 활동 주당 빈도(계산) (예: 12000=12.000회, 99000=UnkNown, BLANK=null)",
        '_MINAC11':  "첫 번째 활동 주당 총 활동 시간(분) (예: 0=없음, 1–99999=분, BLANK=null)",
        '_MINAC21':  "두 번째 활동 주당 총 활동 시간(분) (예: 0=없음, 1–99999=분, BLANK=null)",
        'STRFREQ_':  "근력운동 주당 횟수(유산소 제외) (예: 01–99=월당 1–99회, 777/999=UnkNown, 888=Zero, BLANK=null)",
        'PAMIN11_':  "첫 번째 활동 MET 기준 유효 활동 시간(분) (예: 0–99999, BLANK=null)",
        'PAMIN21_':  "두 번째 활동 MET 기준 유효 활동 시간(분) (예: 0–99999, BLANK=null)",
        'PAVIG11_':  "첫 번째 활동 중 격렬 운동 시간(분) (예: 0–99999, BLANK=null)",
        'PAVIG21_':  "두 번째 활동 중 격렬 운동 시간(분) (예: 0–99999, BLANK=null)",
        '_PAINDX1':  "운동 권장 기준 충족 지표 (예: 1=운동충족, 2=미충족, 7/9=UnkNown)",
        '_PA300R2':  "주당 총 300분 이상 활동 여부 (예: 1=300분 이상, 2=1–299분, 3=0분, 9=UnkNown)",
        '_PASTRNG':  "근력운동 권장 기준 충족 여부 (예: 1=충족, 2=미충족, 7/9=UnkNown)",
        '_PAREC1':   "유산소+근력 운동 권장 준수 여부 (예: 1=Both, 2=AerobicOnly, 3=StrengthOnly, 4=Neither, 9=UnkNown)",
        '_BMI5CAT':  "BMI 체중 상태 (예: 1=Underweight, 2=Normal, 3=Overweight, 4=Obese, BLANK=null)",
        '_PA150R2':  "주당 150분 이상 중강도 운동 여부 (예: 1=150PLUS, 2=1to149, 3=None, 9=UnkNown)",
        'BPHIGH4':   "고혈압 진단 여부 (예: 빈칸=null, 1/2/4=Yes, 3=No, 7/9=UnkNown)",
        'CVDCRHD4':  "관상동맥 심장병 진단 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'CVDSTRK3':  "뇌졸중 진단 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'CHCKIDNY':  "신장 질환 진단 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'DIABETE3':  "당뇨병 진단 여부 (예: 빈칸=null, 1/2/4=Yes, 3=No, 7/9=UnkNown)",
        'SEX':       "성별 (예: 1=Male, 2=Female)",
        'PREGNANT':  "현재 임신 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'SMOKDAY2':  "흡연 여부 (예: 1='Daily', 2='Some', 3='Never', 7='UnkNown', 9='UnkNown', BLANK=null)",
        'ALCDAY5':   "지난 30일 음주 일 수 (예: 월간횟수, 777/999=UnkNown, 888=Zero, BLANK=null)",
        'AVEDRNK2':  "지난 30일 평균 일 음주량 (예: 1–7=잔 수, 77/99=UnkNown, BLANK=null)",
        'MAXDRNKS':  "지난 30일 최대 음주량 (예: 1–76=잔 수, 77/99=UnkNown, BLANK=null)",
        'FRUITJU1':  "지난 한 달 과일 주스 섭취 횟수 (예: 월간횟수, 300/555=Zero, 777/999=UnkNown, BLANK=null)",
        'FRUIT1':    "지난 한 달 과일 섭취 횟수 (예: 월간횟수, 300/555=Zero, 777/999=UnkNown, BLANK=null)",
        'FVBEANS':   "지난 한 달 콩류 섭취 횟수 (예: 월간횟수, 300/555=Zero, 777/999=UnkNown, BLANK=null)",
        'FVGREEN':   "지난 한 달 녹색 채소 섭취 횟수 (예: 월간횟수, 300/555=Zero, 777/999=UnkNown, BLANK=null)",
        'FVORANG':   "지난 한 달 주황색 채소 섭취 횟수 (예: 월간횟수, 300/555=Zero, 777/999=UnkNown, BLANK=null)",
        'VEGETAB1':  "지난 한 달 기타 채소 섭취 횟수 (예: 월간횟수, 300/555=Zero, 777/999=UnkNown, BLANK=null)",
        'EXERANY2':  "지난 30일 운동 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'STRENGTH':  "근력운동 횟수 (예: 월간횟수, 777/999=UnkNown, 888=Zero, BLANK=null)",
        '_RFHYPE5':  "고혈압 진단 여부 (예: 빈칸=null, 1=No, 2=Yes, 7/9=UnkNown)",
        '_MICHD':    "관상동맥 심장병(CHD) 또는 심근경색(MI) 여부 (예: 빈칸=null, 1=Yes, 2=No)",
        'HTM4':      "키(미터) (예: 91–244, 빈칸=null)",
        'WTKG3':     "몸무게(kg) (예: 30–200, 99999=UnkNown)",
        '_BMI5':     "체질량지수(BMI) (예: 18.5–40)",
        '_SMOKER3':  "현재 흡연 여부 (예: 1='Daily', 2='Some', 3='Former', 4='Never', 9='UnkNown')",
        'DRNKANY5':  "최근 30일간 음주 여부 (예: 빈칸=null, 1=Yes, 2=No, 7/9=UnkNown)",
        'DROCDY3_':  "최근 일주일간 평균 음주량 (예: 정수(3,7,10,71)=하루 평균 잔 수, 900=UnkNown, 빈칸=null)",
        '_FRUITEX':  "과일 섭취 여부 (예: 빈칸=null, 1=No, 2=Yes, 7/9=UnkNown)",
        '_VEGETEX':  "채소 섭취 여부 (예: 빈칸=null, 1=No, 2=Yes, 7/9=UnkNown)",
        '_PACAT1':   "신체 활동 수준 범주 (예: 1=VeryActive, 2=Active, 3=Insufficient, 4=Inactive, 9=UnkNown)",
        '_AGEG5YR':  "나이 5세 단위 그룹 (예: 1=18–24, 2=25–29, 3=30–34, 4=35–39, 5=40–44, 6=45–49, 7=50–54, 8=55–59, 9=60–64, 10=65–69, 11=70–74, 12=75–79, 13=80+)"
    }

    # B) 범주형 숫자->문자 매핑
    category_maps = {
    'GENHLTH': {1: 'Excellent', 2: 'VeryGood', 3: 'Good', 4: 'Fair', 5: 'Poor', 7: 'Unknown', 9: 'Unknown'},
    'PHYSHLTH': {**{i: str(i) for i in range(1, 31)}, 88: 'Zero', 77: 'Unknown', 99: 'Unknown'},
    'MENTHLTH': {**{i: str(i) for i in range(1, 31)}, 88: 'Zero', 77: 'Unknown', 99: 'Unknown'},
    'HLTHPLN1': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'POORHLTH': {**{i: str(i) for i in range(1, 31)}, 88: 'Zero', 77: 'Unknown', 99: 'Unknown'},
    'PERSDOC2': {1: 'YesOne', 2: 'YesMulti', 3: 'No', 7: 'Unknown', 9: 'Unknown'},
    'MEDCOST': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'CHECKUP1': {1: 'Within1Y', 2: 'Within2Y', 3: 'Within5Y', 4: 'Over5Y', 7: 'Unknown', 8: 'Never', 9: 'Unknown'},
    'BPMEDS': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'BLOODCHO': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'CHOLCHK': {1: 'Within1Y', 2:'Within2Y', 3:'Within5Y', 4:'Over5Y', 8:'Never', 7:'Unknown', 9:'Unknown', '':'Unknown'},
    'TOLDHI2': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'CVDINFR4': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'ASTHMA3': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'ASTHNOW': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'CHCSCNCR': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'CHCOCNCR': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'CHCCOPD1': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'HAVARTH3': {1: 'Yes', 2: 'No', 7: 'Unknown',9: 'Unknown'},   
    'ADDEPEV2': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'MARITAL': {1: 'Mar', 2: 'Div', 3: 'Wid', 4: 'Sep', 5: 'Nev', 6: 'UnP', 9: 'Unknown'},
    'EDUCA': {1: 'None', 2: 'Elem', 3: 'MidHS', 4: 'HS', 5: 'SomeCol', 6: 'ColGrad', 9: 'Unknown'},
    'RENTHOM1': {1: 'Own', 2: 'Rent', 3: 'Other', 7: 'Unknown', 9: 'Unknown'},
    'VETERAN3': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'EMPLOY1': {1: 'Employed', 2: 'SelfEmp', 3: 'Unemp1Y+', 4: 'Unemp<1Y', 5: 'Homemaker', 6: 'Student', 7: 'Retired', 8: 'Unable', 9: 'Unknown'},
    'INCOME2': {1: '<10K', 2: '10-15K', 3: '15-20K', 4: '20-25K', 5: '25-35K', 6: '35-50K', 7: '50-75K', 8: '75K+', 77: 'Unknown', 99: 'Unknown'},
    'INTERNET': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'QLACTLM2': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'USEEQUIP': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'DECIDE': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'DIFFWALK': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'DIFFDRES': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'DIFFALON': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'SMOKE100': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'STOPSMK2': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'LASTSMK2': {1: 'Within1M', 2: 'Within3M', 3: 'Within6M', 4: 'Within1Y', 5: 'Within5Y', 6: 'Within10Y', 7: 'Y10plus', 8: 'Never', 9: 'Unknown', 99: 'Unknown'},
    'USENOW3': {1: 'Daily', 2: 'Some', 3: 'Never', 7: 'Unknown', 9: 'Unknown'},
    'LMTJOIN3': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'ARTHDIS2': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'ARTHDIS2': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'ARTHSOCL': {1: 'High', 2: 'Moderate', 3: 'No', 7: 'Unknown', 9: 'Unknown'},
    'JOINPAIN': {**{i: str(i) for i in range(0, 11)}, 77: 'Unknown', 99: 'Unknown'},
    'FLUSHOT6': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'PNEUVAC3': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'HIVTST6': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'HIVTSTD3': {**{i: str(i) for i in range(11985, 122017)}, 777: 'Unknown', 999: 'Unknown'},
    'WHRTST10': {1: 'Private', 2: 'Center', 3: 'Inpatient', 4: 'Clinic', 5: 'Prison', 6: 'DrugTx', 7: 'Home', 8: 'Other', 9: 'ER', 77: 'Unknown', 99: 'Unknown'},
    'PDIABTST': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'PREDIAB1': {1: 'Yes', 2: 'Yes', 3: 'No', 7: 'Unknown', 9: 'Unknown'},
    'INSULIN': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'BLDSUGAR': {**{i: str(i) for i in range(0, 1000)}, 888: 'Zero', 999: 'Unknown'},
    'FEETCHK2': {**{i: str(i) for i in range(0, 1000)}, 888: 'Zero', 999: 'Unknown'},
    'DOCTDIAB': {**{i: str(i) for i in range(1, 77)}, 88: 'Zero', 77: 'Unknown', 99: 'Unknown'},
    'CHKHEMO3': {**{i: str(i) for i in range(1, 77)}, 88: 'Zero', 77: 'Unknown', 98: 'Unknown', 99: 'Unknown'},
    'FEETCHK':  {**{i: str(i) for i in range(1, 77)}, 88: 'Zero', 77: 'Unknown', 99: 'Unknown'},
    'DIABEYE': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'CIMEMLOS': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'SXORIENT': {1: 'Hetero', 2: 'LG', 3: 'Bi', 4: 'Other', 9: 'Unknown'},
    'TRNSGNDR': {1: 'Yes', 2: 'Yes', 3: 'Yes', 4: 'No',  9: 'Unknown'},
    'MSCODE': {1: 'Central', 2: 'Fringe', 3: 'Suburban', 5: 'NonMSA'},
    '_RFHLTH': {1: 'Yes', 2: 'No', 9: 'Unknown'},
    '_HCVU651': {1: 'Yes', 2: 'No', 9: 'Unknown'},
    '_CHOLCHK': {1: 'Yes', 2: 'No', 9: 'Unknown'},
    '_RFCHOL': {1: 'No', 2: 'Yes', 9: 'Unknown'},
    '_LTASTH1': {1: 'No', 2: 'Yes', 9: 'Unknown'},
    '_CASTHM1': {1: 'No', 2: 'Yes', 9: 'Unknown'},
    '_ASTHMS1': {1: 'Current', 2: 'Former', 3: 'Never', 9: 'Unknown'},
    '_DRDXAR1': {1: 'Yes', 2: 'No', 9: 'Unknown'},
    '_MRACE1': {1: 'White', 2: 'Black', 3: 'Native', 4: 'Asian', 5: 'Pacific', 6: 'Other', 7: 'Multi', 77: 'Unknown', 99: 'Unknown'},
    '_HISPANC': {1: 'Yes', 2: 'No', 9: 'Unknown'},
    '_RACE': {1: 'White', 2: 'Black', 3: 'Native', 4: 'Asian', 5: 'PacificI', 6: 'Other', 7: 'Multi', 8: 'Hispani', 9: 'Unknown'},
    '_RACEG21': {1: 'Yes', 2: 'No', 9: 'Unknown'},
    '_INCOMG': {1: '<15K', 2: '15-25K', 3: '25-35K', 4: '35-50K', 5: '50K+', 9: 'Unknown'},
    'FC60_': {**{i: str(i) for i in range(0, 8591)}, 99900: 'Unknown'},
    '_PASTAE1': {1: 'Yes', 2: 'No', 9: 'Unknown'},
    '_FLSHOT6': {1: 'Yes', 2: 'No', 9: 'Unknown'},
    '_PNEUMO2': {1: 'Yes', 2: 'No', 9: 'Unknown'},
    '_AIDTST3': {1: 'Yes', 2: 'No', 9: 'Unknown'},
    'EXRACT11': {'': 'NULL', 1: 'Active Gaming', 2: 'Aerobics', 3: 'Backpacking', 4: 'Badminton', 5: 'Basketball', 6: 'Stationary Bicycle', 7: 'Bicycling', 8: 'Canoeing/Kayaking', 9: 'Bowling', 10: 'Boxing', 11: 'Calisthenics', 12: 'Canoe Racing', 13: 'Carpentry', 14: 'Dancing', 15: 'Elliptical/EFX', 16: 'Fishing', 17: 'Frisbee', 18: 'Gardening', 19: 'Golf (cart)', 20: 'Golf (walking)', 21: 'Handball', 22: 'Hiking', 23: 'Hockey', 24: 'Horseback Riding', 25: 'Large Game Hunting', 26: 'Small Game Hunting', 27: 'Inline Skating', 28: 'Jogging', 29: 'Lacrosse', 30: 'Mountain Climbing', 31: 'Lawn Mowing', 32: 'Paddleball', 33: 'House Painting', 34: 'Pilates', 35: 'Racquetball', 36: 'Raking Leaves', 37: 'Running', 38: 'Rock Climbing', 39: 'Jump Rope', 40: 'Rowing Machine', 41: 'Rugby', 42: 'Scuba Diving', 43: 'Skateboarding', 44: 'Skating', 45: 'Sledding', 46: 'Snorkeling', 47: 'Snow Shoveling (machine)', 48: 'Snow Shoveling (manual)', 49: 'Skiing', 50: 'Snowshoeing', 51: 'Soccer', 52: 'Softball/Baseball', 53: 'Squash', 54: 'Stair Climbing', 55: 'Wading Fishing', 56: 'Surfing', 57: 'Swimming', 58: 'Lane Swimming', 59: 'Table Tennis', 60: 'Tai Chi', 61: 'Tennis', 62: 'Touch Football', 63: 'Volleyball', 64: 'Walking', 66: 'Water Skiing', 67: 'Weight Training', 68: 'Wrestling', 69: 'Yoga', 71: 'Caregiving Activities', 72: 'Farm/Ranch Work', 73: 'Household Activities', 74: 'Martial Arts/Karate', 75: 'Upper Body Ergometer', 76: 'Yard Work', 77: 'Unknown', 88: 'Zero', 98: 'Other', 99: 'Unknown'},
    'EXEROFT1': {**{i: str(i) for i in range(0, 300)}, 777: 'Unknown', 999: 'Unknown'},
    'EXERHMM1': {**{i: str(i) for i in range(0, 700)}, 777: 'Unknown', 888: 'Zero', 999: 'Unknown'},
    'EXRACT21': {**{i: str(i) for i in range(0, 77)}, 77: 'Unknown', 88: 'Zero', 98: 'Other', 99: 'Unknown'},
    'EXEROFT2': {**{i: str(i) for i in range(0, 300)}, 777: 'Unknown', 999: 'Unknown'},
    'EXERHMM2': {**{i: str(i) for i in range(0, 700)}, 777: 'Unknown', 888: 'Zero', 999: 'Unknown'},
    'DRNK3GE5': {**{i: str(i) for i in range(1, 77)}, 77: 'Unknown', 88: 'Zero', 99: 'Unknown'},
    '_RFBING5': {1: 'No', 2: 'Yes', 7: 'Unknown', 9: 'Unknown'},
    '_DRNKWEK': {**{i: str(i) for i in range(0, 99000)}, 99900: 'Unknown'},
    '_RFDRHV5': {1: 'No', 2: 'Yes', 7: 'Unknown', 9: 'Unknown'},
    'FTJUDA1_': {**{i: str(i) for i in range(0, 10000)}, '': 'Unknown'},
    'FRUTDA1_': {**{i: str(i) for i in range(0, 10000)}, '': 'Unknown'},
    'BEANDAY_': {**{i: str(i) for i in range(0, 10000)}, '': 'Unknown'},
    'GRENDAY_': {**{i: str(i) for i in range(0, 10000)}, '': 'Unknown'},
    'ORNGDAY_': {**{i: str(i) for i in range(0, 10000)}, '': 'Unknown'},
    'VEGEDA1_': {**{i: str(i) for i in range(0, 10000)}, '': 'Unknown'},
    '_FRUTSUM': {i: str(i) for i in range(0, 99999)},
    '_VEGESUM': {i: str(i) for i in range(0, 99999)},
    '_FRTLT1': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    '_VEGLT1': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'ACTIN11_': {0: 'Low', 1: 'Moderate', 2: 'Vigorous'},
    'ACTIN21_': {0: 'Low', 1: 'Moderate', 2: 'Vigorous'},
    'PADUR1_': {i: str(i) for i in range(0, 600)},
    'PADUR2_': {i: str(i) for i in range(0, 600)},
    'PAFREQ1_': {**{i: str(i) for i in range(0, 99000)}, 99000: 'Unknown'},
    'PAFREQ2_': {**{i: str(i) for i in range(0, 99000)}, 99000: 'Unknown'},
    '_MINAC11': {i: str(i) for i in range(0, 100000)},
    '_MINAC21': {i: str(i) for i in range(0, 100000)},
    'STRFREQ_': {**{i: str(i) for i in range(0, 99000)}, 99000: 'Unknown'},
    'PAMIN11_': {i: str(i) for i in range(0, 100000)},
    'PAMIN21_': {i: str(i) for i in range(0, 100000)},
    'PAVIG11_': {i: str(i) for i in range(0, 100000)},
    'PAVIG21_': {i: str(i) for i in range(0, 100000)},
    '_PAINDX1': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    '_PA300R2': {1: '300plus', 2: '1to299', 3: 'None', 9: 'Unknown'},
    '_PASTRNG': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    '_PAREC1': {1: 'BothMet', 2: 'AerobicOnly', 3: 'StrengthOnly', 4: 'Neither', 9: 'Unknown'},
    '_BMI5CAT': {1:'Underweight', 2:'Normal', 3:'Overweight', 4:'Obese', '':'Unknown'},
    '_PA150R2': {1: '150plus', 2: '1to149', 3: 'None', 9: 'Unknown'},
    'BPHIGH4': {1: 'Yes', 2: 'Yes', '4': 'Yes', 3: 'No', 7: 'Unknown', 9: 'Unknown'},
    'CVDCRHD4': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'CVDSTRK3': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'CHCKIDNY': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'DIABETE3':  {1: 'Yes', 2: 'Yes', '4': 'Yes', 3: 'No', 7: 'Unknown', 9: 'Unknown'},
    'SEX': {1: 'male', 2: 'female'},
    'PREGNANT': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'SMOKDAY2': {1: 'Daily', 2: 'Some', 3: 'Never', 7: 'Unknown', 9: 'Unknown'},
    'ALCDAY5': {**{i: str(i) for i in range(1, 700)}, **{str(i): str(i % 100) for i in range(201, 300)}, 888: 'Zero', 777: 'Unknown', 999: 'Unknown'},
    'AVEDRNK2': {**{str(i): str(i) for i in range(1, 77)}, 88: 'Zero', 77: 'Unknown', 99: 'Unknown'},
    'MAXDRNKS': {**{str(i): str(i) for i in range(1, 77)}, 88: 'Zero', 77: 'Unknown', 99: 'Unknown'},
    'FRUITJU1': {**{i: str(i) for i in range(1, 250)}, 300: 'Zero', 555: 'Zero', 777: 'Unknown', 999: 'Unknown'},
    'FRUIT1': {**{i: str(i) for i in range(1, 250)}, 300: 'Zero', 555: 'Zero', 777: 'Unknown', 999: 'Unknown'},
    'FVBEANS': {**{i: str(i) for i in range(1, 250)}, 300: 'Zero', 555: 'Zero', 777: 'Unknown', 999: 'Unknown'},
    'FVGREEN': {**{i: str(i) for i in range(1, 250)}, 300: 'Zero', 555: 'Zero', 777: 'Unknown', 999: 'Unknown'},
    'FVORANG': {**{i: str(i) for i in range(1, 250)}, 300: 'Zero', 555: 'Zero', 777: 'Unknown', 999: 'Unknown'},
    'VEGETAB1': {**{i: str(i) for i in range(1, 250)}, 300: 'Zero', 555: 'Zero', 777: 'Unknown', 999: 'Unknown'},
    'EXERANY2': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'STRENGTH': {**{i: str(i) for i in range(1, 700)}, 888: 'Zero', 777: 'Unknown', 999: 'Unknown'},
    '_RFHYPE5': {1: 'No', 2: 'Yes', 7:'UnkNown', 9:'UnkNown'},
    '_MICHD': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'HTM4': {i: str(i) for i in range(0, 1000)},
    'WTKG3': {**{i: str(i) for i in range(2300, 29501)}, 99999: 'Unknown'},
    '_BMI5': {**{i: str(i) for i in range(0, 10000)}, '': 'Unknown'},
    '_SMOKER3': {1: 'Daily', 2: 'Some', 3: 'Former', 4: 'Never', 9: 'Unknown'},
    'DRNKANY5': {1: 'Yes', 2: 'No', 7: 'Unknown', 9: 'Unknown'},
    'DROCDY3_': {**{i: str(i) for i in range(0, 900)}, '900': 'Unknown'},
    '_FRUITEX': {1: 'No', 2: 'Yes', 7: 'Unknown', 9: 'Unknown'},
    '_VEGETEX': {1: 'No', 2: 'Yes', 7: 'Unknown', 9: 'Unknown'},
    '_PACAT1': {1: 'VeryActive', 2: 'Active', 3: 'Insufficient', 4: 'Inactive', 9: 'Unknown'},
    '_AGEG5YR': {1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79', 13: '80+'}
} 

    # # C) 특수값 매핑 규칙
    # special_value_maps = {
    #     'PHYSHLTH': {88:0,77:-1,99:-1},
    #     'MENTHLTH': {88:0,77:-1,99:-1},
    #     'POORHLTH': {88:0,77:-1,99:-1},
    #     # 예: _PA300R2 은 lambda로 처리
    #     '_PA300R2': lambda x: 1 if x>=300 else 2 if x>=1 else 3 if x==0 else -1
    #     # 나머지 숫자형 컬럼은 필요 시 dict 추가
    # }

    # 범주형 카테고리 정보
    categorical_info = {}
    if obj_cols and encoder is not None:
        for idx, col in enumerate(obj_cols):
            categorical_info[col] = list(encoder.categories_[idx])

    # 숫자형 범위/예시 정보
    numeric_info = {}
    for col in feature_cols:
        if col not in obj_cols:
            col_num = pd.to_numeric(df[col], errors='coerce')
            if col_num.notna().any():
                numeric_info[col] = (int(col_num.min()), int(col_num.max()), int((col_num.min()+col_num.max())/2))
            else:
                numeric_info[col] = (None,None,None)

    # 입력 루프
        # ────────────────────────────────────────────────────────────────────
    # 9) 사용자 입력 루프: “모든 컬럼”과 “설명”을 출력한 뒤, 값을 차례대로 입력받음
    # ────────────────────────────────────────────────────────────────────
    new_data = {}

    for col in feature_cols:
        desc = feature_desc.get(col, "설명 없음")
        prompt = f"- [{col}] {desc}\n  → "
        val = input(prompt).strip()

    # ⚠️ "Unknown" 문자열 입력 → NaN 처리
        if val.lower() in ['unknown', '모름', '응답거부', ''] or val in ['77', '99', 'NaN']:
            num = pd.NA
        else:
            try:
                num = int(val)
            except:
                num = pd.NA

    # 사람이 볼 수 있는 label만 출력
        if col in category_maps:
            mapped_val = category_maps[col].get(num, 'Unknown')
            print(f"  ↪️ 입력한 {num} 은 '{mapped_val}' 로 매핑됩니다.")

    # # special_value_map 처리
    #     if col in special_value_maps and num is not pd.NA:
    #         rule = special_value_maps[col]
    #         if callable(rule):
    #             num = rule(num)
    #         else:
    #             num = rule.get(num, num)

    # ⚠️ 숫자형이거나 NaN으로 저장
        new_data[col] = num

        try:
            num = int(val)
        except:
            num = -1

# 🟡 ⚠️ 반드시 encoder.transform() **삭제!!**
# for col in obj_cols:
#     new_df[col] = encoder.transform(new_df[col])

# 🟡 숫자형으로 강제 변환 (LightGBM은 숫자만 받음)
    # 입력받은 new_data를 DataFrame으로 변환
    new_df = pd.DataFrame([new_data])

# 숫자형으로 강제 변환
    for col in feature_cols:
        try:
            new_df[col] = pd.to_numeric(new_df[col])
        except:
            new_df[col] = pd.NA

    #if new_df.isna().any().any():
        #new_df = new_df.fillna(-1)

# 예측 및 결과 출력
    probas_list = model.predict_proba(new_df)
    print("\n=== 예측 결과 ===")
    for idx, col in enumerate(target_cols):
        p = probas_list[idx][0][1] * 100
        print(f"[{col}] 양성 확률: {p:.2f}%")


if __name__ == '__main__':
    main()