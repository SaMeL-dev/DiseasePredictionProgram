# 전처리 함수 파일
# src/diabetes/preprocess.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def preprocess_diabetes_dataset(csv_path: str):
    """
    당뇨병 데이터셋을 전처리하여 훈련 및 테스트용 데이터셋으로 분할한다.

    Args:
        csv_path (str): 데이터셋 CSV 파일 경로

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            - X_train (DataFrame): 훈련 입력 데이터
            - X_test (DataFrame): 테스트 입력 데이터
            - y_train (Series): 훈련 라벨
            - y_test (Series): 테스트 라벨
    """
    df = pd.read_csv(csv_path)

    # 흡연 이력 인코딩
    mapping = {
        'never': 0,
        'former': 1,
        'current': 2,
        'ever': 3,
        'not current': 3,   # 맥락상 의미가 유사한 'ever'와 통합합
        'No Info': 5        # 전체 데이터 중 35% 정도가 No Info로 작성되어 있어 drop 하지 않고 안 쓰는 번호인 5번으로 분류
    }
    # 기존 숫자 인코딩
    #df['smoking_history'] = df['smoking_history'].map(mapping)

    # 타겟/피처 분리
    y = df['diabetes']
    X = df.drop(columns=['diabetes'])

    # One-hot 인코딩
    # 🔄 모든 object형 (문자열) 열을 자동 탐지해서 인코딩
    cat_cols = X.select_dtypes(include='object').columns

    # One-hot 인코딩 수행
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    encoded = encoder.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    encoded_df.reset_index(drop=True, inplace=True)

    # 수치형 피처만 남기기
    numeric_X = X.drop(columns=cat_cols).reset_index(drop=True)
    X_processed = pd.concat([numeric_X, encoded_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, encoder, X_processed.columns.tolist()

