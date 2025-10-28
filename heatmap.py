import json
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 파일 불러오기
with open("evaluation_results_with_explanation.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# 2. 정답과 예측 값 리스트 추출
y_true = [item["true_label"] for item in results]
y_pred = [item["predicted_label"] for item in results]

# 3. 정확도 계산
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ 정확도: {accuracy:.2%}")

# 4. Confusion Matrix 생성
labels = ["매수", "관망", "매도"]  # 클래스 순서 고정
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 5. 히트맵 시각화
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (예측 vs 실제)")
plt.tight_layout()
plt.show()
