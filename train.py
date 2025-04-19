# 전체 모델 학습 실행 스크립트
# diabetes 모델 전용 버전

from src.diabetes.preprocess import preprocess_diabetes_dataset
from src.diabetes.model import train_model, save_model
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    """
    테스트 데이터를 활용해 모델의 성능을 평가한다.

    Args:
        model: 학습된 모델 객체
        X_test (DataFrame): 테스트 입력 데이터
        y_test (Series): 테스트 정답 레이블
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    print("\n📊 모델 성능 평가 (Test set):\n")
    print(report)

def main():
    """
    당뇨병 예측 모델 학습 및 저장을 위한 메인 실행 함수.
    """
    # 1. 데이터 로드 및 전처리
    data_path = "data/diabetes/diabetes_prediction_dataset.csv"
    X_train, X_test, y_train, y_test = preprocess_diabetes_dataset(data_path)
    print(f"✅ 데이터 로드 완료: {X_train.shape[0] + X_test.shape[0]}개 샘플, {X_train.shape[1]}개 특성")

    # 2. 모델 학습
    model = train_model(X_train, y_train)
    print("🎯 모델 학습 완료")

    # 3. 모델 저장
    save_model(model)
    print("💾 모델 저장 완료 (diabetes_model.pkl)")

    # 4. 학습한 모델 평가
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()