import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.features.feature_engineering import build_feature_vector

with open("data/labeled_news.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X = []
y = []
label_map = {"매수": 0, "관망": 1, "매도": 2}

for item in data:
    feature = build_feature_vector(item)
    X.append(feature)
    y.append(label_map[item["label"]])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "models/risk_model.pkl")
print("✅ 모델 학습 및 저장 완료")
