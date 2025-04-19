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

    # 입력 데이터를 DataFrame으로 변환
    input_df = pd.DataFrame([user_input])

    # smoking_history 컬럼 One-hot 인코딩 (0~5)
    for i in range(6):
        col = f"smoking_history_{i}"
        input_df[col] = 1 if user_input['smoking_history'] == i else 0

    input_df = input_df.drop(columns=['smoking_history'])

    # 예측
    probability = model.predict_proba(input_df)[0][1]
    return probability
