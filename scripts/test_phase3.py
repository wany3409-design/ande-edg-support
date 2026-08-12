#!/usr/bin/env python
"""
Phase 3 完整测试: ChromaDB 入库 + 语义检索 + 10题测试
"""

import sys, os, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    KNOWLEDGE_DOCS_DIR, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
    RETRIEVAL_TOP_K
)
from src.rag.ingestion import IngestionPipeline

print("=" * 60)
print("  Phase 3: ChromaDB 入库 + 检索测试")
print("=" * 60)

# ===== Step 1: 加载模型 =====
print("\n[1/5] 加载 Embedding 模型...")
t0 = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("shibing624/text2vec-base-chinese", device="cpu")
print(f"  加载完成 ({time.time() - t0:.1f}s)")

# ===== Step 2: 获取 chunks =====
print("\n[2/5] 解析文档并生成 chunks...")
pipeline = IngestionPipeline()
all_chunks = pipeline.ingest_directory(str(KNOWLEDGE_DOCS_DIR))
print(f"  总计: {len(all_chunks)} chunks")

# ===== Step 3: ChromaDB 初始化并入库 =====
print("\n[3/5] 初始化 ChromaDB 并写入向量...")

import chromadb
from chromadb.config import Settings

# 清空旧数据（如果存在）
import shutil
if os.path.exists(CHROMA_PERSIST_DIR):
    for item in os.listdir(CHROMA_PERSIST_DIR):
        item_path = os.path.join(CHROMA_PERSIST_DIR, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False),
)

# 如果 collection 存在则删除重建
try:
    client.delete_collection(CHROMA_COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"description": "安得EDG产品知识库"},
)

# 分批写入（避免一次写入过多）
batch_size = 100
texts = [c.text for c in all_chunks]
metadatas = [c.to_metadata() for c in all_chunks]
ids = [c.chunk_id for c in all_chunks]

t1 = time.time()
total_embedded = 0
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]
    batch_metas = metadatas[i:i + batch_size]

    # Embed
    batch_embeddings = model.encode(
        batch_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    collection.add(
        embeddings=batch_embeddings,
        documents=batch_texts,
        metadatas=batch_metas,
        ids=batch_ids,
    )
    total_embedded += len(batch_texts)
    if (i // batch_size + 1) % 2 == 0:
        print(f"  已写入: {total_embedded}/{len(texts)}")

t2 = time.time()
print(f"  写入完成: {total_embedded} chunks")
print(f"  总耗时: {t2 - t1:.1f}s")
print(f"  平均: {(t2 - t1) / len(texts) * 1000:.1f}ms/chunk")

# ===== Step 4: 磁盘占用 =====
print("\n[4/5] 磁盘占用:")
chroma_size = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(CHROMA_PERSIST_DIR)
    for f in files
)
usage = shutil.disk_usage("C:/")
print(f"  ChromaDB: {chroma_size / 1024 / 1024:.1f} MB")
print(f"  C盘剩余: {usage.free / 1024**3:.1f} GB")

# ===== Step 5: 10 个测试问题检索 =====
print("\n[5/5] 检索测试...")
print("=" * 70)

test_questions = [
    ("Q1", "文件为什么会自动加密？"),
    ("Q2", "文件解密后为什么又自动加密？"),
    ("Q3", "策略配置了为什么没有生效？"),
    ("Q4", "例外目录应该如何配置？"),
    ("Q5", "客户端安装后没有人员怎么办？"),
    ("Q6", "客户端连接不上服务端怎么办？"),
    ("Q7", "AD域同步怎么配置？"),
    ("Q8", "文件外发后为什么仍然打不开？"),
    ("Q9", "压缩包解压后的文件为什么又被加密？"),
    ("Q10", "Word打开提示'你的组织策略阻止了我们为你完成此操作'怎么排查？"),
]

for qid, question in test_questions:
    print(f"\n{'─' * 70}")
    print(f"【{qid}】{question}")

    # Embed query
    q_embedding = model.encode(
        [question], normalize_embeddings=True
    ).tolist()

    # 检索
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=5,
        include=["metadatas", "documents", "distances"],
    )

    print(f"  Top-5 检索结果:")
    for rank in range(5):
        meta = results["metadatas"][0][rank]
        dist = results["distances"][0][rank]
        sim = 1 - dist  # cosine distance -> similarity
        doc_preview = results["documents"][0][rank][:120].replace("\n", " ")

        print(f"  #{rank + 1} | sim={sim:.3f}")
        print(f"      文件: {meta.get('source_file', 'N/A')}")
        print(f"      页码: {meta.get('page', 'N/A')}")
        print(f"      章节: {meta.get('section', '(无)')}")
        print(f"      主题: {meta.get('topic', '(无)')}")
        print(f"      类型: {meta.get('doc_type', 'N/A')}")
        print(f"      预览: {doc_preview}...")

print(f"\n{'=' * 70}")
print("  Phase 3 检索测试完成")
print(f"{'=' * 70}")

# 输出简单的评分统计
print("\n质量评估 (基于相似度):")
for qid, question in test_questions:
    q_embedding = model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_embedding, n_results=5,
        include=["metadatas", "distances"],
    )
    top_sim = 1 - results["distances"][0][0]
    avg_sim = 1 - sum(results["distances"][0]) / 5
    flag = "✅" if top_sim > 0.55 else ("⚠️" if top_sim > 0.45 else "❌")
    print(f"  {flag} [{qid}] top={top_sim:.3f} avg={avg_sim:.3f} | {question[:40]}")
