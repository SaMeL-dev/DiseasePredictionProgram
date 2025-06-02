import pandas as pd

df = pd.read_csv("data/diabetes/diabetes_prediction_dataset.csv")

# 값 확인
print("🧪 unique 값:", df['smoking_history'].unique())
print("📊 데이터 수:", len(df))

# 인코딩이 안되어 있다면 매핑 먼저
mapping = {
    'never': 0,
    'former': 1,
    'current': 2,
    'ever': 3,
    'not current': 4,
    'No Info': 5
}
df['smoking_history'] = df['smoking_history'].map(mapping)

# 재계산
count_per_label = df['smoking_history'].value_counts().sort_index()
missing_count = count_per_label.get(5, 0)
total = len(df)
missing_rate = missing_count / total * 100

print("\n🚬 smoking_history 값별 개수:")
print(count_per_label)
print(f"\n❓ 'No Info'(5번) 비율: {missing_rate:.2f}%")

# 깃 연동 테스트 2