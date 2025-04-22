from src.diabetes.preprocess import preprocess_diabetes_dataset
from src.diabetes.model import train_model, save_model
from sklearn.metrics import classification_report
import joblib

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    print("\n📊 모델 성능 평가 (Test set):\n")
    print(report)

def main():
    # 1. 데이터 로드 및 전처리
    data_path = "data/diabetes/diabetes_prediction_dataset.csv"
    
    # ✅ 6개 리턴값 받기
    X_train, X_test, y_train, y_test, encoder, feature_names = preprocess_diabetes_dataset(data_path)
    print(f"✅ 데이터 로드 완료: {X_train.shape[0] + X_test.shape[0]}개 샘플, {X_train.shape[1]}개 특성")

    # 2. 모델 학습
    model = train_model(X_train, y_train)
    print("🎯 모델 학습 완료")

    # 3. 모델 저장
    save_model(model)
    print("💾 모델 저장 완료 (diabetes_model.pkl)")

    # ✅ 4. 전처리 도구 저장
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(feature_names, "feature_names.pkl")
    print("📦 전처리 정보 저장 완료 (encoder.pkl, feature_names.pkl)")

    # 5. 평가
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
