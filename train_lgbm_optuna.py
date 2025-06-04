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
        # --- 주요 Demographic·건강 상태 관련 컬럼 ---
        'AGE':      "응답자 나이(만 나이, 18~110 사이 숫자)",
        'SEX':      "성별 (Male / Female)",
        'EDUCAG':   "최종 학력 (Never / Elementary / Some school / High school / College)",
        'INCOME2':  "가구 소득 범위 (예: Less than $10,000 / $10,000 to less than $15,000 / ... / $75,000 or more)",
        'MARITAL':  "결혼 상태 (Married / Divorced / Widowed / Never married / Unmarried couple)",
        'GENHLTH':  "전반적 건강 상태 (Excellent / Very good / Good / Fair / Poor)",
        'PHYSHLTH': "지난 30일 중 신체 건강 문제로 활동하지 못한 날 수 (0~30 숫자)",
        'MENTHLTH': "지난 30일 중 정신 건강 문제로 활동하지 못한 날 수 (0~30 숫자)",
        'POORHLTH': "지난 30일 중 신체 또는 정신적 건강이 좋지 않았던 날 수 (0~30 숫자)",
        'HLTHPLN1': "건강 보험이 있나요? (Yes / No / Don’t know / Refused)",
        'PERSDOC2': "의사나 개인 주치의가 있나요? (Yes / No / 기타)",
        'MEDCOST':  "지난 12개월간, 돈이 없어서 의료 서비스를 못 받았나요? (Yes / No / 기타)",
        'CHECKUP1': "마지막 암 정기 검진 시기 (Within past year / Within past 2 years / …)",

        # --- 생활습관·행동 위험 요소 관련 컬럼 ---
        'BPMEDS':    "고혈압 약 복용 여부 (Yes / No / Don’t know / Refused)",
        'BLOODCHO':  "고콜레스테롤 진단 여부 (Yes / No / Don’t know / Refused)",
        'SMOKE100':  "평생 100개비 이상 흡연 여부 (Yes / No)",
        '_SMOKER3':  "현재 흡연 상태 코드 (1=Current every day smoker / 2=Current some day smoker / 3=Former smoker / 4=Never smoked)",
        'DRNKANY5':  "지난 30일간 음주(한 잔 이상)한 날 수 (0~30 숫자)",
        '_DRNKDY3_': "지난 30일간 음주한 날 수 (코드화된 숫자)",
        '_RFDRHV7_': "과도음주(Risky drinking) 여부 코드 (1=Yes / 2=No / 7=Don’t know/Refused)",
        'EXERANY2':  "지난 한 달 간 30분 이상 중강도 신체 활동 여부 (Yes / No)",
        '_PACAT1':   "유산소 운동 빈도 코드 (예: 저활동 / 보통활동 / 고활동)",
        'STRENGTH':  "근력 운동 빈도(주당) (0=안 함 / 1=1회 / 2=2회 / 3=3회 이상)",
        'FRUIT1':    "지난 하루간 과일 섭취 횟수 (0~3 이상 숫자)",
        'FVORANG':   "지난 하루간 오렌지/오렌지 주스 섭취 횟수 (0~숫자)",
        'VEGETAB1':  "지난 하루간 채소 섭취 횟수 (0~3 이상 숫자)",
        '_FRUITEX':  "권장 과일 섭취 충족 여부 (1=Yes / 2=No / 기타)",
        '_VEGETEX':  "권장 채소 섭취 충족 여부 (1=Yes / 2=No / 기타)",

        # --- 신체 측정·지표 관련 컬럼 ---
        'HTM4':      "키(cm) 숫자",
        'WTKG3':     "체중(kg) 숫자",
        '_BMI5':     "체질량지수(BMI) 숫자 (WTKG3 / (HTM4/100)^2)",
        '_MICHD':    "관상동맥질환 여부 파생 코드",
        '_RFHYPE5':  "고혈압 고위험군 여부 코드",
        '_RFSMOK3':  "흡연 고위험군 여부 코드",
        '_RFBMI5':   "비만 고위험군 여부 코드 (BMI ≥ 30kg/㎡)",
        '_RFRISK5':  "종합 만성질환 위험도 지수 코드",

        # --- 정신건강·사회적 환경 관련 컬럼 ---
        '_MENTHLTH': "정신 건강 문제 일수 코드 (코드화된 범주형)",
        '_RFSMOK2':  "현재 흡연 형태 코드 (예: 매일흡연 / 가끔흡연 / 금연자 등)",
        '_RFDRISK2': "행동 위험도 종합 지표 코드",
        '_AGEG5YR':  "연령대 그룹(5년 단위) 코드 (1=18~24 / 2=25~29 / … / 13=80 이상)",

        # --- 기타 대표 변수 ---
        'EMPLOY1':    "직업 상태 (예: Employed / Unemployed / 기타)",
        'INTERNET2':  "집에서 인터넷 사용 여부 (Yes / No)",
        'RENTHOM1':   "주거 형태 (Home owner / Renter / 기타)",
        'TOLDHI2':    "고혈압 진단 여부 (Yes / No / 기타)",
        'DOWNHOPE2':  "지난 30일간 우울감으로 활동 못한 날 수 (숫자)",
        'SAFETY2':    "집/주변 환경 안전 느낌 (Yes / No)",
        'DRNK3GE5':   "과도음주(Heavy drinking) 여부 코드",
        'CVDINFR4':   "심근경색(Heart attack) 진단 여부 (Yes / No)",
        # 그 외 컬럼은 'BRFSS 2015 CodeBook 참조'로 표시될 예정
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
