import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1) CSV 로드 & NaN/비이진(0,1) 제거
    df = pd.read_csv('BRFSS_2015ver14.csv', low_memory=False)
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
    if obj_cols:
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        X[obj_cols] = encoder.fit_transform(X[obj_cols])

    # 4) Train/Test 분할 (80:20), stratify는 1차원으로 지정
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y['BPHIGH4']
    )

    # 5) 분할 결과 출력
    print(f"훈련 데이터셋 크기: {X_train.shape}")
    print(f"테스트 데이터셋 크기: {X_test.shape}\n")

    # 6) 모델 학습
    base = LGBMClassifier(objective='binary', random_state=42)
    model = MultiOutputClassifier(base, n_jobs=-1)
    model.fit(X_train, y_train)

    # 7) 예측 & 평가
    y_pred = model.predict(X_test)
    print("=== 최종 테스트 성능 ===")
    for idx, col in enumerate(target_cols):
        acc = accuracy_score(y_test.iloc[:, idx], y_pred[:, idx])
        print(f"\n{col} 정확도: {acc:.4f}")
        print(classification_report(
            y_test.iloc[:, idx],
            y_pred[:, idx],
            target_names=['음성', '양성'],
            digits=4
        ))

if __name__ == "__main__":
    main()
