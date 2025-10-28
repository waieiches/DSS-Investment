import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def summarize_article(text, model="gpt-3.5-turbo", max_tokens=300) -> str:
    if not text or len(text.strip()) < 20:
        return "내용이 부족하여 요약할 수 없습니다."

    prompt = f"""다음 뉴스 기사를 간결하게 요약해 주세요. 핵심 내용을 3문장 이내로 정리해 주세요:\n\n{text}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=max_tokens,
        )
        summary = response.choices[0].message.content.strip()
        if summary:
            print("📝 요약 내용:", summary)
            return summary
        else:
            print("⚠️ 요약 결과가 비어 있습니다.")
            return None
    except Exception as e:
        print(f"❌ 요약 실패: {e}")
        return None
    
if __name__ == "__main__":
    # 테스트용 기사 본문
    test_text = """
    Apple has introduced a new 'Viral' playlist to both Apple Music and Shazam. 
    This new playlist ranks the top 50 songs discovered by users around the world 
    through the Shazam service. The company aims to surface songs gaining traction 
    organically across different regions, allowing users to explore trends in real-time.
    """
    summary = summarize_article(test_text)
    print(summary)