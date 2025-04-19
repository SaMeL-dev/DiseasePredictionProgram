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
        'No Info': 5
    }
    df['smoking_history'] = df['smoking_history'].map(mapping)

    # 타겟/피처 분리
    y = df['diabetes']
    X = df.drop(columns=['diabetes'])

    # One-hot 인코딩 (smoking_history만)
    cat_cols = ['smoking_history']
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(X[cat_cols]).toarray()  # ⬅ 핵심
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    encoded_df.reset_index(drop=True, inplace=True)

    # 수치형 피처만 따로 뽑기 + 병합
    numeric_X = X.drop(columns=cat_cols).reset_index(drop=True)
    X_processed = pd.concat([numeric_X, encoded_df], axis=1)

    # 결측값 제거
    X_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_processed.dropna(inplace=True)
    y = y[X_processed.index]

    # 데이터 분할
    return train_test_split(X_processed, y, test_size=0.2, random_state=42)