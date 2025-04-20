# Flask 메인 서버 코드
from flask import Flask, render_template, request
import pandas as pd
from src.diabetes.predict import predict_diabetes

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form/<disease>")
def form(disease):
    return render_template("form.html", disease=disease)

@app.route("/result", methods=["POST"])
def result():
    disease = request.form["disease"]
    
    user_input = {
        "gender": int(request.form["gender"]),
        "age": int(request.form["age"]),
        "hypertension": int(request.form["hypertension"]),
        "heart_disease": int(request.form["heart_disease"]),
        "smoking_history": int(request.form["smoking_history"]),
        "bmi": float(request.form["bmi"]),
        "HbA1c_level": float(request.form["hba1c"]),
        "blood_glucose_level": float(request.form["glucose"])
    }

    # 현재는 당뇨 모델만 연결
    if disease == "diabetes":
        probability = predict_diabetes(user_input)
    else:
        probability = 0.0  # 추후 다른 질병 확장 예정

    return render_template("result.html", disease=disease, prob=round(probability * 100, 2))

@app.route("/predict", methods=["POST"])
def predict():
    # 입력값 수집
    age = int(request.form["age"])
    gender = int(request.form["gender"])
    bmi = float(request.form["bmi"])
    smoking = int(request.form["smoking_history"])
    alcohol = int(request.form["alcohol_history"])
    hypertension = int(request.form["hypertension"])
    hba1c = float(request.form["hba1c"])

    # 예측 모델 사용
    user_input = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "smoking_history": smoking,
        "alcohol_history": alcohol,
        "hypertension": hypertension,
        "HbA1c_level": hba1c
    }

    # 예측 결과 가져오기
    result = predict_diabetes(user_input)
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)