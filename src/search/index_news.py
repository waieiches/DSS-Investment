#뉴스 코퍼스 전체를 한 번에 읽어서 임베딩하고 FAISS 인덱스를 생성해 파일로 저장하는 배치 스크립트트

import os
import json
import faiss
from sentence_transformers import SentenceTransformer


def main():
    # 1) 금융 특화 SBERT 모델 로드
    MODEL_NAME = "baconnier/Finance2_embedding_small_en-V1.5"
    model = SentenceTransformer(MODEL_NAME)

    # 2) 프로젝트 루트 기준 뉴스 데이터 경로
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data", "news")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"뉴스 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")

    # 3) 뉴스 문서와 메타데이터 수집
    documents = []
    metadatas = []
    for fn in os.listdir(data_dir):
        if not fn.endswith(".json"): 
            continue
        path = os.path.join(data_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        for art in articles:
            raw = art.get("full_content")
            if not raw:
                continue
            text = raw.strip()
            if len(text) < 30:
                continue
            documents.append(text)
            metadatas.append({
                "title": art.get("title"),
                "source": art.get("source", {}).get("name"),
                "publishedAt": art.get("publishedAt"),
                "filename": fn
            })

    print(f"▶ 임베딩 문서 수: {len(documents)}")

    # 4) 배치 임베딩 수행
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # 5) FAISS 인덱스 생성
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 6) 인덱스 및 메타데이터 저장 경로
    vs_dir = os.path.join(project_root, "vectorstore")
    os.makedirs(vs_dir, exist_ok=True)
    index_path = os.path.join(vs_dir, "financial_news_index.faiss")
    meta_path = os.path.join(vs_dir, "financial_news_metadata.json")

    # 7) 파일로 저장
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    print(f"✅ FAISS 인덱스 저장 완료: {index_path}")


if __name__ == "__main__":
    main()

# (venv) PS C:\Users\toast\genai> python src/search/index_news.py
# C:\Users\toast\genai\venv\lib\site-packages\sentence_transformers\SentenceTransformer.py:196: FutureWarning: The `use_auth_token` argument is deprecated and will be removed in v4 of SentenceTransformers.
#   warnings.warn(
# ▶ 임베딩 문서 수: 332
# Batches: 100%|██████████| 11/11 [01:40<00:00,  9.18s/it]
# ✅ FAISS 인덱스 저장 완료: C:\Users\toast\genai\vectorstore\financial_news_index.faiss