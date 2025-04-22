# 사용자 입력 기반 예측 함수 파일
# src/diabetes/predict.py

import pandas as pd
from src.diabetes.model import load_model

def predict_diabetes(user_input: dict) -> float:
    import pandas as pd
    import joblib
    from src.diabetes.model import load_model

    model = load_model()
    encoder = joblib.load("encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")

    # ✅ 예측 입력 정제
    valid_gender = ['Male', 'Female', 'Other']
    valid_smoking = ['never', 'former', 'current', 'ever', 'not current', 'No Info']

    # 기본값 설정
    if user_input.get('gender') not in valid_gender:
        user_input['gender'] = 'Male'
    if user_input.get('smoking_history') not in valid_smoking:
        user_input['smoking_history'] = 'never'

    # 강제로 문자열 변환
    user_input['gender'] = str(user_input['gender'])
    user_input['smoking_history'] = str(user_input['smoking_history'])

    df = pd.DataFrame([user_input])
    cat_cols = ['gender', 'smoking_history']

    encoded = encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    encoded_df.reset_index(drop=True, inplace=True)

    numeric_df = df.drop(columns=cat_cols).reset_index(drop=True)
    input_df = pd.concat([numeric_df, encoded_df], axis=1)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    probability = model.predict_proba(input_df)[0][1]
    return probability
