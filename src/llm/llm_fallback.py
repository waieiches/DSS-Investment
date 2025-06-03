import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gpt_risk_decision(news_item):
    prompt = f"""
너는 투자 전문가고, 다음은 기사 요약, 감성 분석, 키워드야.
요소들을 종합해서 이 뉴스에 대해 '매수', '관망', '매도' 중 하나로 판단해줘.

요약: {news_item['summary']}
감성 분석 결과: {news_item['sentiment']}
키워드: {', '.join(news_item['keywords'])}

판단:
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def gpt_risk_decision_with_reason(news_item):
    prompt = f"""
다음 뉴스 기사를 바탕으로 투자 판단을 내려줘. 판단 기준은 다음 셋 중 하나야: 매수, 관망, 매도.

기사 요약: {news_item.get('summary', '')}
감성 분석: {news_item.get('sentiment', '')} ({news_item.get('sentiment_score', 0.0)})
본문: {news_item.get('full_content', '')}

형식:
- label: 판단 결과 (매수 / 관망 / 매도)
- reason: 판단 이유
- keywords: 관련 핵심 키워드 3~5개

응답은 반드시 JSON 형식으로 해줘.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 fallback 값 반환
        print("❌ GPT 응답이 JSON 형식이 아님:", content.strip())
        return {
            "label": "예측 실패",
            "reason": content.strip(),  # GPT 응답 그대로 반환
            "keywords": []
        }

    return result