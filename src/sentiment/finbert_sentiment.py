#감성 분석 모듈
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 모델 로딩 (한 번만)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def analyze_sentiment_finbert(text: str) -> tuple[str, float]:
    """
    FinBERT로 감성 분석 실행 (negative, neutral, positive 중 분류)
    :param text: 분석할 본문 텍스트
    :return: (label, score) 튜플
    """
    if not text or len(text.strip()) < 50:
        return "neutral", 0.0  # 너무 짧은 경우 중립 처리

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

    label_id = torch.argmax(probs).item()
    score = probs[label_id].item()
    labels = ["negative", "neutral", "positive"]
    return labels[label_id], round(score, 4)