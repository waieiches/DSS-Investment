import os
import json
import faiss
from sentence_transformers import SentenceTransformer


def main():
    # 1) 모델 로드
    MODEL_NAME = "baconnier/Finance2_embedding_small_en-V1.5"
    model = SentenceTransformer(MODEL_NAME)

    # 2) FAISS 인덱스 로드 
    index_path = os.path.join("vectorstore", "financial_news_index.faiss")
    metadata_path = os.path.join("vectorstore", "financial_news_metadata.json")

    assert os.path.exists(index_path), f"Index file not found: {index_path}"
    assert os.path.exists(metadata_path), f"Metadata file not found: {metadata_path}"

    index = faiss.read_index(index_path)

    # 3) 메타데이터 로드
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    # 4) 차원 일치 확인
    embed_dim = model.get_sentence_embedding_dimension()
    assert index.d == embed_dim, f"Index dimension ({index.d}) != Model dimension ({embed_dim})"
    print(f"✅ Index dimension matches model: {embed_dim}")

    # 5) 간단한 검색 테스트
    queries = [
        "금리 인상 여파",
        "기업 실적 발표",
        "미국 주가 전망"
    ]
    k = 3
    for q in queries:
        q_emb = model.encode([q], convert_to_numpy=True)
        D, I = index.search(q_emb, k)
        print(f"\n🔎 Query: {q}")
        for distance, idx in zip(D[0], I[0]):
            meta = metadatas[idx]
            title = meta.get("title", "(제목 없음)")
            source = meta.get("source", "(출처 없음)")
            published = meta.get("publishedAt", "(날짜 없음)")
            print(f"  - {title} | {source} | {published} (distance: {distance:.4f})")

    print("\n🎉 테스트 완료: 모든 검색 및 인덱스 로드가 성공적으로 수행되었습니다.")


if __name__ == "__main__":
    main()

# 실행 예시:
# (venv) PS C:\Users\toast\genai> python src/search/test.py
      
# C:\Users\toast\genai\venv\lib\site-packages\sentence_transformers\SentenceTransformer.py:196: FutureWarning: The `use_auth_token` argument is deprecated and will be removed in v4 of SentenceTransformers.
#   warnings.warn(
# ✅ Index dimension matches model: 384

# 🔎 Query: 금리 인상 여파
#   - quri-parts-ionq 0.22.0 | Pypi.org | 2025-05-12T07:30:58Z (distance: 1.1329)
#   - pulumi-okta 4.19.0a1746771698 | Pypi.org | 2025-05-09T06:30:44Z (distance: 1.1329)
#   - pulumi-okta 4.19.0a1746734672 | Pypi.org | 2025-05-08T20:16:28Z (distance: 1.1329)

# 🔎 Query: 기업 실적 발표
#   - Chimps' rhythmic drumming and complex calls hint at origins of human language | NPR | 2025-05-12T10:00:00Z (distance: 1.2740)
#   - quri-parts-ionq 0.22.0 | Pypi.org | 2025-05-12T07:30:58Z (distance: 1.2801)
#   - pulumi-okta 4.19.0a1746771698 | Pypi.org | 2025-05-09T06:30:44Z (distance: 1.2801)

# 🔎 Query: 미국 주가 전망
#   - quri-parts-ionq 0.22.0 | Pypi.org | 2025-05-12T07:30:58Z (distance: 1.1091)
#   - pulumi-okta 4.19.0a1746771698 | Pypi.org | 2025-05-09T06:30:44Z (distance: 1.1091)
#   - pulumi-okta 4.19.0a1746734672 | Pypi.org | 2025-05-08T20:16:28Z (distance: 1.1091)

# 🎉 테스트 완료: 모든 검색 및 인덱스 로드가 성공적으로 수행되었습니다.
# (venv) PS C:\Users\toast\genai> 