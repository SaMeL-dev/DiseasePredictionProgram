# 사용자 입력 및 결과 반환 앱(CLI or Web)
# 당뇨 예측 모델 콘솔 입력 버전
import sys
from src.diabetes.predict import predict_diabetes

def get_input(prompt, type_=str):
    while True:
        try:
            return type_(input(prompt))
        except ValueError:
            print("❗ 잘못된 입력입니다. 다시 시도해주세요.")

def main():
    print("🩺 [당뇨병 예측 시스템]")
    print("다음 정보를 입력해주세요.\n")

    gender = get_input("성별 (0: 여성, 1: 남성): ", int)
    age = get_input("나이: ", int)
    hypertension = get_input("고혈압 유무 (0: 없음, 1: 있음): ", int)
    heart_disease = get_input("심장병 유무 (0: 없음, 1: 있음): ", int)
    smoking_history = get_input("흡연 척도 (0: 비흡연, 1: 5년 이상 금연, 2: 흡연 중  3: 5년 이내 금연): ", int)
    bmi = get_input("BMI 지수: ", float)
    hba1c_level = get_input("헤모글로빈 A1c 수치: ", float)
    blood_glucose_level = get_input("혈당 수치: ", float)

    user_input = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c_level,
        'blood_glucose_level': blood_glucose_level
    }

    
    result = predict_diabetes(user_input)
    print(f"\n🧾 예측된 당뇨병 발병 확률: {result * 100:.2f}%")

    if result >= 0.7:
        print("⚠️ 고위험군입니다. 가까운 병원에 내원하는 것을 권장합니다.")
    elif result >= 0.4:
        print("🟡 주의가 필요합니다. 건강관리 습관 개선이 필요한 시점입니다다.")
    else:
        print("🟢 현재는 낮은 위험도입니다.")

if __name__ == "__main__":
    main()