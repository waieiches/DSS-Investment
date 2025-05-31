import os
from openai import OpenAI
from dotenv import load_dotenv

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
