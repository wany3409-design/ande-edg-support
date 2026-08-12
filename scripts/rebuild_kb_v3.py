#!/usr/bin/env python
"""
Phase 4.5: 重建知识库

- 新 HTML 解析器 (按 slide 分组)
- V3 splitter (400 字符, 扩展 topic 关键词)
- bge-small-zh-v1.5 embedding
- 输出: ande_edg_v3 collection
"""

import sys, os, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import KNOWLEDGE_DOCS_DIR, CHROMA_PERSIST_DIR
from src.rag.ingestion import IngestionPipeline
from src.text_processor.splitter import TextSplitter

COLLECTION_NAME = "ande_edg_v3"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# V2-style chunk size (dense retrieval) + V3 improvements (HTML parser, topics)
CHUNK_SIZE = 250
CHUNK_OVERLAP = 40


def main():
    print("=" * 60)
    print("  Phase 4.5: 重建知识库 (V3)")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Chunk: {CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP}")
    print(f"  Embedding: {EMBEDDING_MODEL}")
    print("=" * 60)

    # Step 1: 解析 + 切分
    print("\n[Step 1] 解析文档 + V3 切分")
    t0 = time.time()
    pipeline = IngestionPipeline()
    pipeline.splitter = TextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = pipeline.ingest_directory(str(KNOWLEDGE_DOCS_DIR))
    elapsed = time.time() - t0
    print(f"  Total: {len(chunks)} chunks ({elapsed:.1f}s)")

    # 统计
    avg_len = sum(len(c.text) for c in chunks) / max(1, len(chunks))
    print(f"  Avg length: {avg_len:.0f} chars, Min: {min(len(c.text) for c in chunks)}, Max: {max(len(c.text) for c in chunks)}")

    # 按来源统计
    from collections import Counter
    src_counts = Counter(c.source_file for c in chunks)
    for src, cnt in src_counts.most_common():
        print(f"    {src}: {cnt} chunks")

    # 按 topic 统计
    topic_counts = Counter()
    no_topic = 0
    for c in chunks:
        if c.topic:
            for t in c.topic.split("; "):
                topic_counts[t.strip()] += 1
        else:
            no_topic += 1
    print(f"  Chunks without topic: {no_topic}/{len(chunks)}")
    print(f"  Top topics: {topic_counts.most_common(15)}")

    # Step 2: 加载 embedding 模型
    print(f"\n[Step 2] 加载 embedding 模型")
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    print(f"  Model loaded ({time.time() - t0:.1f}s), dim={model.get_embedding_dimension()}")

    # Step 3: 写入 ChromaDB
    print(f"\n[Step 3] 写入 ChromaDB collection: {COLLECTION_NAME}")
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # 删除旧 collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted old collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "安得EDG知识库 v3 (400-char chunks, bge-small)"},
    )

    texts = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]
    ids = [c.chunk_id for c in chunks]

    batch_size = 200
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]

        embeddings = model.encode(
            batch_texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

        collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas,
            ids=batch_ids,
        )
        print(f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: {len(batch_texts)} chunks")

    elapsed = time.time() - t0
    print(f"  Write complete: {len(texts)} chunks in {elapsed:.1f}s ({elapsed/len(texts)*1000:.1f}ms/chunk)")

    # 磁盘
    chroma_size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(CHROMA_PERSIST_DIR) for f in files
    )
    usage = shutil.disk_usage("C:/")
    print(f"\n  ChromaDB size: {chroma_size / 1024 / 1024:.1f} MB")
    print(f"  C盘剩余: {usage.free / 1024**3:.1f} GB")

    # Step 4: 快速验证
    print(f"\n[Step 4] 快速验证")
    sample_queries = [
        "安得EDG怎么做POC测试？",
        "安得EDG支持哪些文件类型的加密？",
        "客户端连接不上服务端怎么排查？",
    ]
    for q in sample_queries:
        q_emb = model.encode([q], normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=q_emb, n_results=3, include=["metadatas", "documents", "distances"])
        print(f"\n  Query: {q}")
        for rank in range(3):
            meta = res["metadatas"][0][rank]
            sim = round(1 - res["distances"][0][rank], 4)
            src = meta.get("source_file", "?")[:40]
            section = meta.get("section", "")[:40]
            topic = meta.get("topic", "")
            print(f"    [{rank+1}] sim={sim:.4f} src={src} section={section} topic={topic}")
            snippet = res['documents'][0][rank][:120].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            print(f"         text: {snippet}")

    print(f"\n{'=' * 60}")
    print(f"  知识库重建完成: {COLLECTION_NAME}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
