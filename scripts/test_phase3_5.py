#!/usr/bin/env python
"""
Phase 3.5: 检索优化 + Embedding A/B 测试

- v2 chunk 策略（200字符、无metadata header）
- 20 道标注测试题
- text2vec-base-chinese vs bge-small-zh-v1.5
"""

import sys, os, time, shutil, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import KNOWLEDGE_DOCS_DIR, CHROMA_PERSIST_DIR
from src.rag.ingestion import IngestionPipeline

# ====================================================================
# 测试问题集（20题，含人工标注）
# 标注依据：仅基于已有的 3 份安得文档内容
# ====================================================================
TEST_QUESTIONS = [
    # (id, question, expected_source_keywords, expected_topic, answerable)
    ("Q01", "安得EDG支持哪些文件类型的加密？",
     ["产品使用手册", "策略配置"], "文件类型策略", True),

    ("Q02", "文件为什么会自动加密？",
     ["产品使用手册", "技术培训"], "透明加密", True),

    ("Q03", "如何对单个文件进行手动解密？",
     ["产品使用手册", "手动解密"], "文件解密", True),

    ("Q04", "文件解密后为什么又自动加密了？",
     ["技术培训"], "透明加密", True),

    ("Q05", "例外目录应该如何配置？",
     ["产品使用手册", "策略配置", "例外"], "例外目录", True),

    ("Q06", "文件类型策略在哪里配置？",
     ["产品使用手册", "策略配置"], "文件类型策略", True),

    ("Q07", "策略配置了为什么没有生效？",
     ["产品使用手册", "策略配置"], "策略配置", True),

    ("Q08", "AD域同步怎么配置？",
     ["产品使用手册", "用户同步", "AD"], "AD域同步", True),

    ("Q09", "客户端安装后没有人员怎么办？",
     [], "", False),  # 知识库中未发现相关内容

    ("Q10", "客户端连接不上服务端怎么排查？",
     ["技术培训", "端口", "产品使用手册"], "客户端服务端连接", True),

    ("Q11", "安得EDG服务端有哪些端口？",
     ["技术培训", "端口映射"], "端口配置", True),

    ("Q12", "文件外发后对方为什么打不开？",
     ["产品使用手册", "外发", "技术培训"], "文件外发", True),

    ("Q13", "压缩包解压后的文件为什么又被加密了？",
     ["技术培训", "加密逻辑"], "透明加密", True),

    ("Q14", "Word打开提示'你的组织策略阻止了我们为你完成此操作'怎么排查？",
     ["技术培训", "caseviews"], "故障排查", True),

    ("Q15", "PDF文件打不开怎么办？",
     ["产品使用手册", "技术培训"], "故障排查", True),

    ("Q16", "POC测试应该怎么开展？",
     ["技术培训"], "POC测试", True),

    ("Q17", "安得EDG的水印功能支持哪些类型？",
     ["产品使用手册", "技术培训", "水印"], "水印", True),

    ("Q18", "客户端日志在哪里查看？如何收集日志排查问题？",
     ["技术培训", "产品使用手册", "日志"], "日志排查", True),

    ("Q19", "服务端如何部署？支持哪些操作系统？",
     ["技术培训", "产品介绍"], "安装部署", True),

    ("Q20", "安得EDG是否支持与钉钉集成实现审批？",
     [], "", False),  # 知识库中未发现相关内容
]


def load_model(model_name: str, device: str = "cpu"):
    """加载 embedding 模型"""
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    print(f"  加载 {model_name} ...")
    model = SentenceTransformer(model_name, device=device)
    elapsed = time.time() - t0
    dim = model.get_sentence_embedding_dimension() if hasattr(model, 'get_sentence_embedding_dimension') else model.get_embedding_dimension()
    max_seq = model.max_seq_length
    print(f"  -> 维度={dim}, max_seq={max_seq}, 耗时={elapsed:.1f}s")
    return model, dim, max_seq


def build_v2_chunks():
    """使用 v2 splitter 重新生成 chunks"""
    print("\n[构建 v2 chunks]")
    # 强制使用新的 splitter 参数
    from src.text_processor.splitter import TextSplitter, V2_CHUNK_SIZE, V2_CHUNK_OVERLAP
    pipeline = IngestionPipeline()
    pipeline.splitter = TextSplitter(chunk_size=V2_CHUNK_SIZE, chunk_overlap=V2_CHUNK_OVERLAP)

    chunks = pipeline.ingest_directory(str(KNOWLEDGE_DOCS_DIR))
    print(f"  总计: {len(chunks)} chunks (chunk_size={V2_CHUNK_SIZE}, overlap={V2_CHUNK_OVERLAP})")

    # 检查 chunk 平均长度
    avg_len = sum(len(c.text) for c in chunks) / max(1, len(chunks))
    max_len = max(len(c.text) for c in chunks)
    min_len = min(len(c.text) for c in chunks)
    print(f"  平均长度: {avg_len:.0f} 字符, 最小: {min_len}, 最大: {max_len}")
    return chunks


def build_collection(client, collection_name: str, model, chunks):
    """将 chunks 写入 ChromaDB collection"""
    import chromadb
    from chromadb.config import Settings

    # 删除旧 collection
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "安得EDG知识库 v2"},
    )

    texts = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]
    ids = [c.chunk_id for c in chunks]

    batch_size = 200
    start = time.time()
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

    elapsed = time.time() - start
    print(f"  写入耗时: {elapsed:.1f}s ({elapsed/len(texts)*1000:.1f}ms/chunk)")
    return collection


def run_retrieval_test(collection, model, label: str):
    """运行 20 题检索测试"""
    print(f"\n{'=' * 70}")
    print(f"  [{label}] 20 题检索测试")
    print(f"{'=' * 70}")

    results = []
    for qid, question, expected_sources, expected_topic, answerable in TEST_QUESTIONS:
        q_emb = model.encode([question], normalize_embeddings=True).tolist()
        res = collection.query(
            query_embeddings=q_emb, n_results=5,
            include=["metadatas", "documents", "distances"],
        )

        hits = []
        for rank in range(5):
            meta = res["metadatas"][0][rank]
            dist = res["distances"][0][rank]
            sim = round(1 - dist, 4)
            src = meta.get("source_file", "")
            topic = meta.get("topic", "")
            section = meta.get("section", "")
            page = meta.get("page", 0)
            doc_preview = res["documents"][0][rank][:100].replace("\n", " ")

            # 判断是否命中期望来源
            hit_source = any(
                kw in src for kw in expected_sources
            ) if expected_sources else False
            hit_topic = (expected_topic.lower() in topic.lower()) if expected_topic else False

            hits.append({
                "rank": rank + 1,
                "sim": sim,
                "source": src,
                "page": page,
                "section": section,
                "topic": topic,
                "preview": doc_preview,
                "hit_source": hit_source,
                "hit_topic": hit_topic,
            })

        # 计算是否命中（Top-1/3/5）
        top1_hit = hits[0]["hit_source"] or hits[0]["hit_topic"] if answerable else None
        top3_hit = any(h["hit_source"] or h["hit_topic"] for h in hits[:3]) if answerable else None
        top5_hit = any(h["hit_source"] or h["hit_topic"] for h in hits[:5]) if answerable else None

        results.append({
            "qid": qid,
            "question": question,
            "answerable": answerable,
            "expected_sources": expected_sources,
            "expected_topic": expected_topic,
            "hits": hits,
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "top5_hit": top5_hit,
        })

    return results


def print_results(results, label: str):
    """打印测试结果"""
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    top1 = sum(1 for r in answerable if r["top1_hit"]) / max(1, len(answerable))
    top3 = sum(1 for r in answerable if r["top3_hit"]) / max(1, len(answerable))
    top5 = sum(1 for r in answerable if r["top5_hit"]) / max(1, len(answerable))

    # MRR
    mrr_sum = 0
    for r in answerable:
        for h in r["hits"]:
            if h["hit_source"] or h["hit_topic"]:
                mrr_sum += 1.0 / h["rank"]
                break
    mrr = mrr_sum / max(1, len(answerable))

    print(f"\n  [{label}] 可回答问题 ({len(answerable)}/{len(results)}) 命中率:")
    print(f"    Top-1: {top1:.1%}  ({sum(1 for r in answerable if r['top1_hit'])}/{len(answerable)})")
    print(f"    Top-3: {top3:.1%}  ({sum(1 for r in answerable if r['top3_hit'])}/{len(answerable)})")
    print(f"    Top-5: {top5:.1%}  ({sum(1 for r in answerable if r['top5_hit'])}/{len(answerable)})")
    print(f"    MRR:   {mrr:.3f}")

    # 不可回答的问题
    if unanswerable:
        print(f"\n  不可回答问题 ({len(unanswerable)}):")
        for r in unanswerable:
            top_sim = r["hits"][0]["sim"]
            flag = "✅正确拒答" if top_sim < 0.45 else "⚠️相似度偏高"
            print(f"    [{r['qid']}] {r['question'][:50]}... | top1_sim={top_sim:.3f} {flag}")

    # 逐题详情
    print(f"\n  逐题详情:")
    for r in results:
        flag = "✅" if r["answerable"] and r["top1_hit"] else ("⚠️" if r["answerable"] and r["top3_hit"] else ("❌" if r["answerable"] else "—"))
        top_src = r["hits"][0]["source"][:30] if r["hits"] else "N/A"
        top_topic = r["hits"][0]["topic"] if r["hits"] else "N/A"
        print(f"    {flag} [{r['qid']}] top1_sim={r['hits'][0]['sim']:.3f} src={top_src}... topic={top_topic}")

    return {"top1": top1, "top3": top3, "top5": top5, "mrr": mrr}


def main():
    import chromadb
    from chromadb.config import Settings

    print("=" * 60)
    print("  Phase 3.5: 检索优化 + Embedding A/B 测试")
    print("=" * 60)

    # ===== Step 1: 构建 v2 chunks =====
    print("\n[Step 1] 构建 v2 chunks (200字符, 无metadata前缀)")
    chunks = build_v2_chunks()

    # 展示一个 chunk 样例
    print("\n  Chunk 样例:")
    sample = chunks[50]
    print(f"    text ({len(sample.text)}字): {sample.text[:150]}...")
    print(f"    metadata: source={sample.source_file}, page={sample.page}, topic={sample.topic}")

    # ===== Step 2: 初始化 ChromaDB =====
    print("\n[Step 2] 初始化 ChromaDB")
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # ===== Model A: text2vec-base-chinese =====
    print("\n[Step 3] Model A: text2vec-base-chinese")
    model_a, dim_a, maxseq_a = load_model("shibing624/text2vec-base-chinese")

    collection_a = build_collection(client, "ande_edg_v2_text2vec", model_a, chunks)

    # 磁盘占用
    chroma_size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(CHROMA_PERSIST_DIR) for f in files
    )
    print(f"  ChromaDB 总占用: {chroma_size / 1024 / 1024:.1f} MB")

    results_a = run_retrieval_test(collection_a, model_a, "text2vec-base-chinese")
    metrics_a = print_results(results_a, "text2vec-base-chinese")

    # 检索速度
    t_a = time.time()
    for _ in range(5):
        q_emb = model_a.encode(["测试"], normalize_embeddings=True).tolist()
        collection_a.query(query_embeddings=q_emb, n_results=5)
    t_a = (time.time() - t_a) / 5 * 1000
    print(f"\n  平均检索耗时: {t_a:.0f}ms")

    # ===== Model B: bge-small-zh-v1.5 =====
    print(f"\n{'─' * 70}")
    print("\n[Step 4] Model B: BAAI/bge-small-zh-v1.5")
    print("  下载中 (~48MB, 轻量级)...")
    model_b, dim_b, maxseq_b = load_model("BAAI/bge-small-zh-v1.5")

    collection_b = build_collection(client, "ande_edg_v2_bge_small", model_b, chunks)

    chroma_size2 = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(CHROMA_PERSIST_DIR) for f in files
    )
    print(f"  ChromaDB 总占用: {chroma_size2 / 1024 / 1024:.1f} MB")

    results_b = run_retrieval_test(collection_b, model_b, "bge-small-zh-v1.5")
    metrics_b = print_results(results_b, "bge-small-zh-v1.5")

    # 检索速度
    t_b = time.time()
    for _ in range(5):
        q_emb = model_b.encode(["测试"], normalize_embeddings=True).tolist()
        collection_b.query(query_embeddings=q_emb, n_results=5)
    t_b = (time.time() - t_b) / 5 * 1000
    print(f"\n  平均检索耗时: {t_b:.0f}ms")

    # ===== Step 5: 磁盘检查 =====
    usage = shutil.disk_usage("C:/")
    # 模型缓存大小
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_sizes = {}
    for name in ["text2vec-base-chinese", "bge-small-zh-v1.5"]:
        for root, dirs, files in os.walk(cache_dir):
            if name in root and "snapshots" in root:
                total = sum(os.path.getsize(os.path.join(root, f)) for f in files)
                model_sizes[name] = total
                break

    # ===== 对比报告 =====
    print(f"\n{'=' * 70}")
    print(f"  A/B 对比报告")
    print(f"{'=' * 70}")
    print(f"  {'指标':<25} {'text2vec-base':<20} {'bge-small-zh':<20}")
    print(f"  {'─' * 65}")
    print(f"  {'向量维度':<25} {dim_a:<20} {dim_b:<20}")
    print(f"  {'max_seq_length':<25} {maxseq_a:<20} {maxseq_b:<20}")
    print(f"  {'Top-1 命中率':<25} {metrics_a['top1']:.1%}                  {metrics_b['top1']:.1%}")
    print(f"  {'Top-3 命中率':<25} {metrics_a['top3']:.1%}                  {metrics_b['top3']:.1%}")
    print(f"  {'Top-5 命中率':<25} {metrics_a['top5']:.1%}                  {metrics_b['top5']:.1%}")
    print(f"  {'MRR':<25} {metrics_a['mrr']:.3f}                {metrics_b['mrr']:.3f}")
    print(f"  {'平均检索耗时':<25} {t_a:.0f}ms                  {t_b:.0f}ms")
    a_size = model_sizes.get("text2vec-base-chinese", 0) / 1024 / 1024
    b_size = model_sizes.get("bge-small-zh-v1.5", 0) / 1024 / 1024
    print(f"  {'模型磁盘占用':<25} {a_size:.0f}MB                  {b_size:.0f}MB")
    print(f"  {'C盘剩余':<25} {usage.free / 1024**3:.1f}GB")
    print(f"  {'ChromaDB占用':<25} {chroma_size2 / 1024 / 1024:.1f}MB")
    print(f"  {'v2 chunks数':<25} {len(chunks)}")

    # 逐题对比
    print(f"\n  逐题 Top-1 对比:")
    print(f"  {'ID':<5} {'问题':<40} {'text2vec':<10} {'bge-small':<10} {'期望来源'}")
    print(f"  {'─' * 90}")
    for ra, rb in zip(results_a, results_b):
        a_src = ra["hits"][0]["source"][:25] if ra["hits"] else "N/A"
        b_src = rb["hits"][0]["source"][:25] if rb["hits"] else "N/A"
        a_hit = "✅" if ra["answerable"] and ra["top1_hit"] else ("—" if not ra["answerable"] else "❌")
        b_hit = "✅" if rb["answerable"] and rb["top1_hit"] else ("—" if not rb["answerable"] else "❌")
        exp = ra["expected_sources"][0][:20] if ra["expected_sources"] else "(知识库外)"
        print(f"  {ra['qid']:<5} {ra['question'][:38]:<40} {a_hit} {a_src:<8} {b_hit} {b_src:<8} {exp}")

    # 推荐
    print(f"\n{'─' * 70}")
    if metrics_b["top1"] >= metrics_a["top1"] and metrics_b["top3"] >= metrics_a["top3"]:
        winner = "B: bge-small-zh-v1.5"
        reason = "Top-1/Top-3 命中率均不低于 text2vec，且 max_seq=512、模型仅 48MB"
    elif metrics_a["top1"] > metrics_b["top1"]:
        winner = "A: text2vec-base-chinese"
        reason = "Top-1 命中率更高"
    else:
        winner = "需进一步分析"
        reason = "两者各有优势"

    print(f"  推荐: {winner}")
    print(f"  原因: {reason}")
    print(f"\n{'=' * 70}")
    print(f"  Phase 3.5 完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
