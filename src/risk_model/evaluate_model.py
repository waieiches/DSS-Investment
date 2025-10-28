import json
from sklearn.metrics import classification_report
from src.risk_model.predict_risk import predict_with_fallback, predict_with_explanation

# 평가용 데이터셋 불러오기
with open("data/test_labeled_news.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

y_true = []
y_pred = []

for item in test_data:
    actual_label = item["label"]
    predicted_label = predict_with_explanation(item)

    y_true.append(actual_label)
    y_pred.append(predicted_label)

print("🔍 평가 결과:")
print(classification_report(y_true, y_pred, digits=3))