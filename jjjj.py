import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    df = pd.read_csv('data/BRFSS_2015ver17.csv', low_memory=False)
    target_cols = ['BPHIGH4', 'CVDCRHD4', 'CVDSTRK3', 'CHCKIDNY', 'DIABETE3']
    df = df.dropna(subset=target_cols)
    for col in target_cols:
        df = df[df[col].isin([0, 1])]

    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols].copy()
    y = df[target_cols].astype(int)

    obj_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoder = None
    if obj_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[obj_cols] = encoder.fit_transform(X[obj_cols].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y['BPHIGH4']
    )

    category_feature = ['GENHLTH', 'PERSDOC2', 'CHECKUP1', 'CHOLCHK', 'MARITAL', 'EDUCA',
        'RENTHOM1', 'EMPLOY1', 'INCOME2', 'USENOW3', 'ARTHSOCL', 'WHRTST10',
        'SXORIENT', '_ASTHMS1', '_MRACE1', '_RACE', '_INCOMG', 'ACTIN11_',
        'ACTIN21_', '_PA300R2', '_PAREC1', '_BMI5CAT', '_PA150R2',
        'SEX', 'SMOKDAY2', '_SMOKER3', '_PACAT1', '_AGEG5YR', 'LASTSMK2', 'MSCODE']
    
    X_train[category_feature] = X_train[category_feature].astype('category')
    X_test[category_feature] = X_test[category_feature].astype('category')

    # 최적 파라미터 또는 실험용 하이퍼파라미터를 넣은 예
    model = MultiOutputClassifier(LGBMClassifier(
    # 다중 이진 분류
    objective='binary',
    # 재현성
    random_state=42,
    # 학습률 / 필수 요소
    learning_rate=0.05,
    # 트리개수 / 필수 요소
    n_estimators=300,
    # 각 트리의 최대 깊이 / 선택적 요소
    max_depth=7,
    # 하나의 트리에서 사용할 수 있는 리프 노드의 수 / 필수 요소
    num_leaves=60,
    # 리프 노드가 각져야 하는 최소 데이터 수 / 필수 요소
    min_child_samples=15,
    # 학습에 사용할 데이터 샘플 비율 / 선택적 요소
    subsample=0.9,
    # 하나의 트리 학습 시 사용할 피처 비율 / 선택적 요소
    colsample_bytree=0.8,
    # L1 정규화 계수 / 선택적 요소
    reg_alpha=2.0,
    # L2 정규화 계수 / 선택적 요소
    reg_lambda=1.0,
    # 출력 제어용 / 선택적 요소
    verbose=-1
    ), n_jobs=-1)

    model.fit(X_train, y_train,
                categorical_feature = category_feature)


    print("=== 최종 테스트 성능 ===")
    y_pred = model.predict(X_test)
    for idx, col in enumerate(target_cols):
        acc = accuracy_score(y_test[col], y_pred[:, idx])
        print(f"\n[{col}] 정확도: {acc:.4f}")
        print(classification_report(
            y_test[col],
            y_pred[:, idx],
            labels=[0, 1],
            target_names=["음성(0)", "양성(1)"],
            digits=4
        ))

    print("\n\n--- 사용자 입력 예측 ---")
    print(f"총 피처 수: {len(feature_cols)}")
    print("값 입력 없이 엔터를 누르게 되면 Null(빈칸)으로 처리가 됩니다.\n\n")

    feature_desc = {
        'GENHLTH':   "전반적인 건강 상태에 대한 자가 평가 (예: 1: 'Excellent', 2: 'VeryGood', 3: 'Good', 4: 'Fair', 5: 'Poor', -1='Unknown')",
        'PHYSHLTH':  "지난 30일간 신체 건강이 좋지 않았던 일수 (예: 0~30 사이 정수, -1=Unknown)",
        'MENTHLTH':  "지난 30일간 정신 건강(스트레스, 우울, 감정 문제 포함한 활동을 하지 못함) 일수 (예: 0~30 사이 정수, -1=Unknown)", 
        'POORHLTH': "지난 30일 동안 신체적 또는 정신적 건강 문제로 평소 하던 활동 (자가관리, 업무, 여가 등)을 하지 못한 일수 (예: 0~30 사이 정수, -1=Unknown)",
        'HLTHPLN1':  "건강 보험, HMO(선불 건강관리), Medicare, 인디언 보건 서비스 등 건강 보장 제도를 갖고 있는가? (예: 1=Yes, 2=No, -1=Unknown)",
        'PERSDOC2':  "개인 주치의 또는 건강관리 제공자가 있다고 생각하는지 (예: 1=한명, 2=두명이상, 3=없음, -1=Unknown)",
        'MEDCOST':   "지난 12개월 동안 비용 문제로 의사의 진료를 받지 못한 적이 있는지 (예: 1=Yes, 2=No, -1=Unknown)",
        'CHECKUP1':  "마지막으로 정기 건강검진을 받은 시점 (예: 1=1년 이내, 2=2년 이내, 3=5년 이내, 4=5년 이상, -1=Unknown)",
        'BPMEDS':    "현재 고혈압 약을 복용 중인지 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'BLOODCHO':  "혈중 콜레스테롤 수치를 측정한 적이 있는가? (예: 1=Yes, 2=No, -1=Unknown)", 
        'CHOLCHK':   "마지막으로 혈중 콜레스테롤 수치를 측정한 시점은 언제인가? (예: 1=1년 이내, 2=2년 이내, 3=5년 이내, 4=5년, -1=Unknown)",
        'TOLDHI2':   "보건 전문가로부터 혈중 콜레스테롤 수치가 높다고 들은 적이 있는가? (예:1=Yes, 2=No, -1=Unknown)",
        'CVDINFR4':  "보건 전문가로부터 심근경색(심장마비)을 진단받은 적이 있는가? (예: 1=Yes, 2=No, -1=Unknown)",
        'ASTHMA3':   "보건 전문가로부터 천식 진단을 받은 적이 있는가? (예: 1=Yes, 2=No, -1=Unknown)",
        'ASTHNOW':   "현재 천식이 있는지 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'CHCSCNCR':  "보건 전문가로부터 피부암(스킨 캔서) 진단을 받은 적이 있는가? (예: 1=Yes, 2=No, -1=Unknown)",
        'CHCOCNCR':  "보건 전문가로부터 피부암을 제외한 다른 암을 진단받은 적이 있는가? (예: 1=Yes, 2=No, -1=Unknown)",
        'CHCCOPD1':  "보건 전문가로부터 COPD, 폐기종, 기관지염 등 진단을 받은 적이 있는가? (예: 1=Yes, 2=No, -1=Unknown)",
        'HAVARTH3':  "보건 전문가로부터 관절염 진단을 받은 적이 있는가? (예:  1=Yes, 2=No, -1=Unknown)",
        'ADDEPEV2':  "보건 전문가로부터 우울 장애 진단을 받은 적이 있는가? (예: 1=Yes, 2=No, -1=Unknown)",
        'MARITAL':   "혼인 상태 (예: 1=결혼, 2=이혼, 3=과부, 4=분가, 5=결혼한적없음, 6=미혼부부, -1=Unknown)",
        'EDUCA':     "최고로 수료한 학년 또는 학력 수준 (예: 1=학교다닌적없음, 2=초등, 3=중등, 4=고등, 5=대학, 6=대학졸업, -1=Unknown)",
        'RENTHOM1':  "주택 소유 형태 (예: 1=Own, 2=Rent, 3=Other; -1=Unknown)",
        'VETERAN3':  "군 복무 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'EMPLOY1':   "현재 고용 상태 (예: 1=회사, 2=자영업자, 3=1년 이상 실직 상태, 4=1년 미만 실직 상태, 5=주부, 6=학생, 7=은퇴, 8=근무 불가능, -1=Unknown)",
        'INCOME2':   "가구 연간 소득 (예: 1=$10,000미만, 2=$10,000이상 $15,000미만, 3=$15,000이상 $20,000미만, 4=$20,000이상 $25,000미만, 5=$25,000이상 $35,000미만, 6=$35,000이상 $50,000미만, 7=$50,000이상 $75,000미만, 8=$75,000이상, -1=Unknown)",
        'INTERNET':  "지난 30일 이내 인터넷 사용 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'QLACTLM2':  "신체적/정신적/감정적 문제로 활동 제한 여부 (예:1=Yes, 2=No, -1=Unknown)",
        'USEEQUIP':  "지팡이/휠체어/특수장비 사용 필요 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'DECIDE':    "집중/기억/의사결정 어려움 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'DIFFWALK':  "걷기/계단 오르기 어려움 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'DIFFDRES':  "옷 입기/목욕하기 어려움 여부 (예:1=Yes, 2=No, -1=Unknown)",
        'DIFFALON':  "병원 방문/쇼핑 등 혼자 볼일 보기 어려움 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'SMOKE100':  "생애 동안 담배 100개비 이상 흡연 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'STOPSMK2':  "지난 12개월 금연 시도 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'LASTSMK2':  "마지막 흡연 시기 (예: 1=1달 이내, 2=3달 이내, 3=6달 이내, 4=1년 이내 5=10년 이내, 7=10년 이상, 8=피운적 없음, -1=Unknown, BLANK=null)",
        'USENOW3':   "현재 씹는 담배/스너프/스누스 사용 여부 (예: 1=Daily, 2=Some, 3=Never, -1=Unknown)",
        'LMTJOIN3':  "현재 관절염/관절 증상으로 활동 제한 여부 (예:1=Yes, 2=No, -1=Unknown)",
        'ARTHDIS2':  "관절염/관절 증상이 취업/업무에 미치는 영향 여부 (예:1=Yes, 2=No, -1=Unknown)",
        'ARTHSOCL':  "지난 30일 관절염/관절 증상으로 쇼핑/영화/종교/사교 활동에 방해 정도 (예: 1=높음, 2=중간, 3=없음, -1=Unknown)",
        'JOINPAIN':  "지난 30일 관절 통증 정도 (예: 0~10 정수, -1=Unknown)",
        'FLUSHOT6':  "지난 12개월 독감 예방접종 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'PNEUVAC3':  "폐렴 백신 접종 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'HIVTST6':   "HIV 검사 받은 적 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'HIVTSTD3':  "마지막 HIV 검사 시기 (월/년, 예: 2015년 5월=52015, -1=Unknown)",
        'WHRTST10':  "마지막 HIV 검사 장소 (예: 1=개인 의사 또는 HMO, 2=상담 및 검사 센터, 3=병원 입원 중, 4=클리닉, 5=교도소 등 수감 시설, 6=약물 치료 시설, 7=집, 8=기타, 9=응급실, -1=Unknown)",
        'PDIABTST':  "지난 3년 내 고혈당/당뇨병 검사 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'PREDIAB1':  "당뇨 전단계 진단 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'INSULIN':   "현재 인슐린 사용 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'DOCTDIAB':  "지난 12개월 당뇨병 진료 횟수 (예: 0~76 정수, -1=Unknown)",
        'CHKHEMO3':  "수최근 12개월당화혈색소(A1C) 검사 횟수 (예: 0~76 정수, -1=Unknown, )",
        'FEETCHK':   "지난 12개월 발 검사 횟수 (예: 0~76 정수, -1=Unknown)",
        'DIABEYE':   "당뇨 합병증(망막병증) 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'CIMEMLOS':  "지난 12개월 기억력 저하/혼란 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'SXORIENT':  "성적 지향 (예: 1=이성애자, 2=레즈비언/게이, 3=양성애자, 4=기타, -1=Unknown)",
        'TRNSGNDR':  "트랜스젠더 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'MSCODE':    "거주 지역 대도시권 상태 (예: 1=MSA 중심도시, 2=MSA 주변, 3=MSA 교외, 5=비MSA)",
        '_RFHLTH':   "전반적 건강 상태 (예: 1=좋음/매우좋음/보통, 2=나쁨/매우나쁨, -1=Unknown)",
        '_HCVU651':  "18~64세 응답자 건강보험 여부 (예: 1=있음, 2=없음, -1=Unknown)",
        '_CHOLCHK':  "최근 5년 내 콜레스테롤 검사 여부 (예: 1=예, 2=아니요, -1=Unknown)",
        '_RFCHOL':   "콜레스테롤 수치 높음 진단 여부 (예: 1=아니요, 2=예, -1=Unknown)",
        '_LTASTH1':  "성인 후 천식 진단 여부 (예: 1=아니요, 2=예, -1=Unknown)",
        '_CASTHM1':  "현재 천식 여부 (예: 1=아니요, 2=예, -1=Unknown)",
        '_ASTHMS1':  "현재/과거 천식 여부 (예: 1=현재, 2=과거, 3=없음, -1=Unknown)",
        '_DRDXAR1':  "관절염 진단 여부 (예: 1=진단받음, 2=진단받지않음)",
        '_MRACE1':   "다인종 인종 분류 (예: 1=White, 2=Black, 3=Native, 4=Asian, 5=Pacific, 6=Other, 7=Multi, -1=Unknown)",
        '_HISPANC':  "히스패닉/라티노 여부 (예: 1=예, 2=아니요, -1=Unknown)",
        '_RACE':     "인종/민족 범주 (예: 1=White, 2=Black, 3=Native, 4=Asian, 5=PacificI, 6=Other, 7=Multi, 8=Hispanic, -1=Unknown)",
        '_RACEG21':  "비히스패닉 백인 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        '_INCOMG':   "연간 가구 소득 (예: 1=$15,000미만, 2=$15,000이상 $25,000미만, 3=$25,000이상 $35,000미만, 4=$35,000이상 $50,000미만, 5=$50,000 이상, -1=Unknown)",
        '_PASTAE1':  "운동 권장 기준 충족 여부 (예: BLANK=null, 1=Yes, 2=No, -1=Unknown)",
        '_FLSHOT6':  "65세 이상 독감 예방접종 여부 (예: BLANK=null, 1=Yes, 2=No, -1=Unknown)",
        '_PNEUMO2':  "65세 이상 폐렴 백신 접종 여부 (예: BLANK=null, 1=Yes, 2=No, -1=Unknown)",
        '_AIDTST3':  "HIV 검사 여부 (예: BLANK=null, 1=Yes, 2=No, -1=Unknown)",
        'EXRACT11':  "첫 번째 주 활동 유형 코드 (예: 1=액티브 게이밍, 2=에어로빅, 3=배낭여행, 4=배드민턴, 5=농구, 6=실내 자전거, 7=자전거, 8=카누/카약, 9=볼링, 10=복싱, 11=맨몸 체조, 12=카누 경기, 13=목공, 14=춤, 15=일립티컬/EFX, 16=낚시, 17=프리스비, 18=원예, 19=골프(전동), 20=골프(도보), 21=핸드볼, 22=하이킹, 23=하키, 24=승마, 25=대형 사냥, 26=소형 사냥, 27=인라인 스케이트, 28=조깅, 29=라크로스, 30=등산, 31=잔디 깎기, 32=패들볼, 33=집 페인트칠, 34=필라테스, 35=라켓볼, 36=낙엽 긁기, 37=달리기, 38=암벽등반, 39=줄넘기, 40=로잉 머신, 41=럭비, 42=스쿠버 다이빙, 43=스케이트보드, 44=스케이트, 45=썰매 탑승, 46=스노클링, 47=눈 치우기(기계), 48=눈 치우기(삽), 49=스키, 50=스노슈잉, 51=축구, 52=소프트볼/야구, 53=스쿼시, 54=계단 오르기, 55=낚시(장화), 56=서핑, 57=수영, 58=레인 수영, 59=탁구, 60=태극권, 61=테니스, 62=터치 풋볼, 63=배구, 64=걷기, 66=수상스키, 67=웨이트 트레이닝, 68=레슬링, 69=요가, 71=돌봄 활동, 72=농장/목장 일, 73=가사 활동, 74=무술/가라데, 75=상체 자전거(에르고), 76=마당일, -1=Unknown)",
        'EXEROFT1':  "지난 한 달 동안 이 활동(걷기, 달리기, 조깅, 수영)을 주당 또는 월당 몇 회 했습니까? (예: 월간횟수, -1=Unknown)",
        'EXERHMM1':  "EXRACT11에서 선택한 운동 한 번 할 때 지속 시간 (예: 분 단위, -1=Unknown)",
        'EXRACT21':  "지난 한 달 동안 두 번째로 많이 한 신체 활동 유형 코드 (예: 1=액티브 게이밍, 2=에어로빅, 3=배낭여행, 4=배드민턴, 5=농구, 6=실내 자전거, 7=자전거, 8=카누/카약, 9=볼링, 10=복싱, 11=맨몸 체조, 12=카누 경기, 13=목공, 14=춤, 15=일립티컬/EFX, 16=낚시, 17=프리스비, 18=원예, 19=골프(전동), 20=골프(도보), 21=핸드볼, 22=하이킹, 23=하키, 24=승마, 25=대형 사냥, 26=소형 사냥, 27=인라인 스케이트, 28=조깅, 29=라크로스, 30=등산, 31=잔디 깎기, 32=패들볼, 33=집 페인트칠, 34=필라테스, 35=라켓볼, 36=낙엽 긁기, 37=달리기, 38=암벽등반, 39=줄넘기, 40=로잉 머신, 41=럭비, 42=스쿠버 다이빙, 43=스케이트보드, 44=스케이트, 45=썰매 탑승, 46=스노클링, 47=눈 치우기(제설기), 48=눈 치우기(삽), 49=스키, 50=스노슈잉, 51=축구, 52=소프트볼/야구, 53=스쿼시, 54=계단 오르기, 55=낚시(물속), 56=서핑, 57=수영, 58=레인 수영, 59=탁구, 60=태극권, 61=테니스, 62=터치 풋볼, 63=배구, 64=걷기, 66=수상스키, 67=웨이트 트레이닝, 68=레슬링, 69=요가, 71=돌봄 활동, 72=농장/목장 일, 73=가사 활동, 74=가라데/무술, 75=상체 자전거(에르고), 76=마당일, 0=다른 활동 없음, -1=Unknown)",
        'EXEROFT2':  "지난 한 달 동안 두 번째 활동을 주당/월당 몇 회 했는가? (예: 월간횟수, -1=Unknown)",
        'EXERHMM2':  "두 번째 활동 한 번 할 때 지속 시간 (예: 분 단위, -1=Unknown)",
        'DRNK3GE5':  "지난 30일 폭음 횟수 (남성: 한 번에 5잔 이상, 여성: 4잔 이상) (예: 0–76=횟수 -1=Unknown)",
        '_RFBING5':  "폭음 여부 (예: 1=아니요, 2=예, -1=Unknown)",
        '_DRNKWEK':  "주당 음주 횟수 (예: 0=음주하지 않음, 1–98999=횟수, -1=Unknown)",
        '_RFDRHV5':  "고위험 음주 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'FTJUDA1_':  "과일 주스 1일 섭취 횟수 (예: 125=1.25회)",
        'FRUTDA1_':  "과일 섭취 횟수(계산) (예: 250=2.50회)",
        'BEANDAY_':  "콩류 섭취 횟수(계산) (예: 175=1.75회)",
        'GRENDAY_':  "진한 녹색 채소 섭취 횟수(계산) (예: 300=3.00회)",
        'ORNGDAY_':  "주황색 채소 섭취 횟수(계산) (예: 250=2.50회)",
        'VEGEDA1_':  "기타 채소 섭취 횟수(계산) (예: 325=3.25회)",
        '_FRUTSUM':  "전체 과일 섭취 횟수(계산) (예: 450=4.50회)",
        '_VEGESUM':  "전체 채소 섭취 횟수(계산) (예: 1025=10.25회)",
        '_FRTLT1':   "하루 과일 1회 이상 섭취 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        '_VEGLT1':   "하루 채소 1회 이상 섭취 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'ACTIN11_':  "첫 번째 활동 운동 강도 (예: 1=평범한 정도 , 2=강한 정도)",
        'ACTIN21_':  "두 번째 활동 운동 강도 (예: 1=평범한 정도, 2=강한 정도)",
        'PADUR1_':   "첫 번째 활동 주당 총 지속 시간(분) (예: 0–599=분)",
        'PADUR2_':   "두 번째 활동 주당 총 지속 시간(분) (예: 0–599=분)",
        'PAFREQ1_':  "첫 번째 활동 주당 빈도(계산) (예: 12000=12.000회, -1=Unknown)",
        'PAFREQ2_':  "두 번째 활동 주당 빈도(계산) (예: 12000=12.000회, -1=Unknown)",
        '_MINAC11':  "첫 번째 활동 주당 총 활동 시간(분) (예: 0–99999=분)",
        '_MINAC21':  "두 번째 활동 주당 총 활동 시간(분) (예: 0–99999=분)",
        'STRFREQ_':  "근력운동 주당 횟수(유산소 제외) (예: 1000=1회/2000=2회 , -1=Unknown, BLANK=null)",
        'PAMIN11_':  "첫 번째 활동 MET 기준 유효 활동 시간(분) (예: 0–99999,)",
        'PAMIN21_':  "두 번째 활동 MET 기준 유효 활동 시간(분) (예: 0–99999)",
        'PAVIG11_':  "첫 번째 활동 중 격렬 운동 시간(분) (예: 0–99999)",
        'PAVIG21_':  "두 번째 활동 중 격렬 운동 시간(분) (예: 0–99999)",
        '_PAINDX1':  "운동 권장 기준 충족 지표 (예: 1=운동충족, 2=미충족, -1=Unknown)",
        '_PA300R2':  "주당 총 300분 이상 활동 여부 (예: 1=300분 이상, 2=1~299분, 3=0분, -1=Unknown)",
        '_PASTRNG':  "근력운동 권장 기준 충족 여부 (예: 1=충족, 2=미충족, -1=Unknown)",
        '_PAREC1':   "유산소+근력 운동 권장 준수 여부 (예: 1=두 기준 모두 충족, 2=유산소 운동 기준만 충족, 3=근력 운동 기준만 충족, 4=둘 다 충족하지 않음, -1=Unknown)",
        '_BMI5CAT':  "BMI 체중 상태 (예: 1=저체중, 2=정상 체중, 3=과체중, 4=비만, BLANK=null)",
        '_PA150R2':  "주당 150분 이상 중강도 운동 여부 (예: 1=150분 이상 운동, 2=1~149분 운동, 3=안함, -1=Unknown)",
        'SEX':       "성별 (예: 1=Male, 2=Female)",
        'PREGNANT':  "현재 임신 여부 (예: BLANK=null, 1=Yes, 2=No, -1=Unknown)",
        'SMOKDAY2':  "흡연 여부 (예: 1=매일, 2=가끔, 3=없음, -1='Unknown')",
        'ALCDAY5':   "지난 30일 음주 일 수 (예: 월간횟수, -1=Unknown, 0=Zero,)",
        'AVEDRNK2':  "지난 30일 평균 일 음주량 (예: 1–76=잔 수, -1=Unknown, 'ALCDAY5'에서 0입력=엔터키)",
        'MAXDRNKS':  "지난 30일 최대 음주량 (예: 1–76=잔 수, -1=Unknown, 'ALCDAY5'에서 0입력=엔터키)",
        'FRUITJU1':  "지난 한 달 과일 주스 섭취 횟수 (예: 월간횟수, 0=Zero, -1=Unknown)",
        'FRUIT1':    "지난 한 달 과일 섭취 횟수 (예: 월간횟수, 0=Zero, -1=Unknown)",
        'FVBEANS':   "지난 한 달 콩류 섭취 횟수 (예: 월간횟수, 0=Zero, -1=Unknown)",
        'FVGREEN':   "지난 한 달 녹색 채소 섭취 횟수 (예: 월간횟수, 0=Zero, -1=Unknown,)",
        'FVORANG':   "지난 한 달 주황색 채소 섭취 횟수 (예: 월간횟수, 0=Zero, -1=Unknown)",
        'VEGETAB1':  "지난 한 달 기타 채소 섭취 횟수 (예: 월간횟수, 0=Zero, -1=Unknown)",
        'EXERANY2':  "지난 30일 운동 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'STRENGTH':  "근력운동 횟수 (예: 월간횟수, -1=Unknown, 0=Zero)",
        '_RFHYPE5':  "고혈압 진단 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        '_MICHD':    "관상동맥 심장병(CHD) 또는 심근경색(MI) 여부 (예: 1=Yes, 2=No)",
        'HTM4':      "키(미터) (예: 91–244)",
        'WTKG3':     "몸무게(kg) (예: 2300 - 29500: 소수점 2자리 포함(2300=23.00 으로 생각하면 됨, -1=Unknown)",
        '_BMI5':     "체질량지수(BMI) (예: 1 - 9999: 소수점 2자리 포함(9999=99.99라고 생각하면됨)",
        '_SMOKER3':  "현재 흡연 여부 (예: 1='Daily', 2='Some', 3='Former', 4='Never', -1='Unknown')",
        'DRNKANY5':  "최근 30일간 음주 여부 (예: 1=Yes, 2=No, -1=Unknown)",
        'DROCDY3_':  "최근 일주일간 평균 음주량 (예: 하루 평균 잔 수, -1=Unknown)",
        '_PACAT1':   "신체 활동 수준 범주 (예: 1=매우 활동적, 2=활동적, 3=다소 부족함, 4=거의 운동 안함, -1=Unknown)",
        '_AGEG5YR':  "나이 5세 단위 그룹 (예: 1=18–24, 2=25–29, 3=30–34, 4=35–39, 5=40–44, 6=45–49, 7=50–54, 8=55–59, 9=60–64, 10=65–69, 11=70–74, 12=75–79, 13=80+)"
    }

    numeric_info = {}
    for col in feature_cols:
        if col not in obj_cols:
            col_num = pd.to_numeric(df[col], errors='coerce').dropna()
            if not col_num.empty:
                numeric_info[col] = (
                    int(col_num.min()),
                    int(col_num.max()),
                    int(col_num.mean())
                )

    categorical_info = {
        col: list(encoder.categories_[i]) for i, col in enumerate(obj_cols)
    } if encoder else {}

    new_data = {}
    for col in feature_cols:
        if col in feature_desc:
            desc = feature_desc[col]
        elif col in categorical_info:
            desc = f"(예: {', '.join(map(str, categorical_info[col]))})"
        elif col in numeric_info:
            min_val, max_val, mid_val = numeric_info[col]
            desc = f"(예: {mid_val}, 범위: {min_val}~{max_val})"
        else:
            desc = "(입력 예시 없음)"


        val = input(f"[{col}] {desc} → ").strip()
        if val == '' or val.lower() == 'nan':
            new_data[col] = np.nan
        else:
            try:
                new_data[col] = float(val)
            except:
                new_data[col] = np.nan

    new_df = pd.DataFrame([new_data])
    if encoder and obj_cols:
        new_df[obj_cols] = encoder.transform(new_df[obj_cols].astype(str))
        # 범주형 컬럼 지정 (학습 시 사용한 컬럼과 동일해야 함)
    new_df[category_feature] = new_df[category_feature].astype('category')


    probas_list = model.predict_proba(new_df)
    print("\n=== 예측 결과 ===")
    for idx, col in enumerate(target_cols):
        try:
            prob = probas_list[idx][0][1] * 100 if len(probas_list[idx][0]) > 1 else 0
            print(f"[{col}] 양성 확률: {prob:.2f}%")
        except Exception as e:
            print(f"[{col}] 예측 실패: {e}")

if __name__ == '__main__':
    main()