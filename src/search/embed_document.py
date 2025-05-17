#단일 문서 임베딩을 위한 코드
import os
import json
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

# 1. 금융 특화 SBERT 모델 로딩
MODEL_NAME = "baconnier/Finance2_embedding_small_en-V1.5"
model = SentenceTransformer(MODEL_NAME,
                            use_auth_token=True)

# 2. 데이터 경로 설정
data_dir = "data/news"

# 3. 문서와 메타데이터 수집
documents = []
metadatas = []
for filename in os.listdir(data_dir):
    if not filename.endswith(".json"):
        continue
    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
        articles = json.load(f)
    for art in articles:
        txt = art.get("full_content", "").strip()
        if len(txt) < 30:
            continue
        documents.append(txt)
        metadatas.append({
            "title": art.get("title"),
            "source": art.get("source", {}).get("name"),
            "publishedAt": art.get("publishedAt"),
            "filename": filename
        })

print(f"▶ 임베딩할 문서 수: {len(documents)}")

# 4. 배치 임베딩 수행
# convert_to_numpy=True 로 넘파이 배열 반환, show_progress_bar=True 로 진행 표시
embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# 5. FAISS 인덱스 생성 및 저장
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

os.makedirs("vectorstore", exist_ok=True)
faiss.write_index(index, "vectorstore/financial_news_index.faiss")

with open("vectorstore/financial_news_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadatas, f, ensure_ascii=False, indent=2)

print("✅ 금융 특화 FAISS 인덱스 및 메타데이터 저장 완료")
