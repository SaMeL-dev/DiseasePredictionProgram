# 사용자 입력 기반 예측 함수 파일
# src/diabetes/predict.py

import pandas as pd
from src.diabetes.model import load_model

def predict_diabetes(user_input: dict):
    """
    사용자 입력값을 기반으로 당뇨병 발병 확률을 예측한다.

    Args:
        user_input (dict): 사용자 입력값 딕셔너리.
            예시:
            {
                'gender': 0,
                'age': 45,
                'hypertension': 0,
                'heart_disease': 0,
                'smoking_history': 2,
                'bmi': 28.7,
                'HbA1c_level': 6.2,
                'blood_glucose_level': 145
            }

    Returns:
        float: 당뇨병 발병 확률 (0.0 ~ 1.0)
    """
    model = load_model()

    # 기본 입력을 DataFrame으로 변환
    input_df = pd.DataFrame([user_input])

    # smoking_history One-hot 인코딩
    for i in range(4):
        col = f"smoking_history_{i}"
        input_df[col] = 1 if user_input['smoking_history'] == i else 0
    # 누락된 컬럼은 0으로 채움
    input_df['smoking_history_4'] = 0
    input_df['smoking_history_5'] = 0

    # gender One-hot 인코딩
    gender = user_input['gender']
    input_df['gender_Female'] = 1 if gender == 0 else 0
    input_df['gender_Male'] = 1 if gender == 1 else 0
    input_df['gender_Other'] = 0  # 기타 선택 없음

    # 수치형 컬럼 추가
    input_df['heart_disease'] = user_input.get('heart_disease', 0)
    input_df['blood_glucose_level'] = user_input.get('blood_glucose_level', 0)

    # 불필요한 컬럼 제거
    input_df = input_df.drop(columns=['smoking_history', 'gender'])

    # 최종 컬럼 순서 맞추기
    required_columns = [
        'age', 'bmi', 'HbA1c_level', 'hypertension', 'alcohol_history',
        'gender_Female', 'gender_Male', 'gender_Other',
        'heart_disease', 'blood_glucose_level',
        'smoking_history_0', 'smoking_history_1', 'smoking_history_2',
        'smoking_history_3', 'smoking_history_4', 'smoking_history_5'
    ]

    input_df = input_df.reindex(columns=required_columns, fill_value=0)
    
    # 예측
    probability = model.predict_proba(input_df)[0][1]
    return probability
