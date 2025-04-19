# 학습, 저장 로직 파일
# src/diabetes/model.py

import os
import joblib
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "diabetes_model.pkl"))

def train_model(X, y):
    """
    주어진 데이터를 사용해 랜덤 포레스트 모델을 학습한다.

    Args:
        X (pd.DataFrame): 입력 특성 데이터
        y (pd.Series): 라벨 데이터 (0 또는 1)

    Returns:
        model (RandomForestClassifier): 학습된 모델 객체
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def save_model(model, path=MODEL_PATH):
    """
    학습된 모델을 지정된 경로에 저장한다.

    Args:
        model (sklearn.base.BaseEstimator): 학습된 모델 객체
        path (str): 저장 경로 (.pkl 파일)
    """
    joblib.dump(model, path)

def load_model(path=MODEL_PATH):
    """
    저장된 모델 파일을 로드한다.

    Args:
        path (str): 모델 파일 경로 (.pkl)

    Returns:
        model (sklearn.base.BaseEstimator): 로드된 모델 객체
    """
    return joblib.load(path)