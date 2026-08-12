#!/usr/bin/env python
"""
Phase 4: RAG 回答全链路测试

20 题测试集 — 完整链路：
问题分类 → Chroma 召回 (Top 10) → Rerank (Top 3-5) → DeepSeek → 回答 + 引用

测试指标：
- 分类准确率
- 检索命中率 (Top-1/3/5)
- Rerank 效果
- 可回答判断准确率
- 回答质量（引用来源是否准确）
- 不可回答问题是否触发拒答
- 链路耗时
"""

import sys
import os
import time
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CHROMA_PERSIST_DIR
from src.llm.answer_pipeline import AnswerPipeline, PHASE4_COLLECTION, PHASE4_EMBEDDING_MODEL

# ====================================================================
# 20 题测试集（与 Phase 3.5 一致）
# ====================================================================
TEST_QUESTIONS = [
    ("Q01", "安得EDG支持哪些文件类型的加密？",
     ["产品使用手册", "策略配置"], "文件类型策略", True, "knowledge"),

    ("Q02", "文件为什么会自动加密？",
     ["产品使用手册", "技术培训"], "透明加密", True, "knowledge"),

    ("Q03", "如何对单个文件进行手动解密？",
     ["产品使用手册", "手动解密"], "文件解密", True, "implementation"),

    ("Q04", "文件解密后为什么又自动加密了？",
     ["技术培训"], "透明加密", True, "troubleshooting"),

    ("Q05", "例外目录应该如何配置？",
     ["产品使用手册", "策略配置", "例外"], "例外目录", True, "implementation"),

    ("Q06", "文件类型策略在哪里配置？",
     ["产品使用手册", "策略配置"], "文件类型策略", True, "implementation"),

    ("Q07", "策略配置了为什么没有生效？",
     ["产品使用手册", "策略配置"], "策略配置", True, "troubleshooting"),

    ("Q08", "AD域同步怎么配置？",
     ["产品使用手册", "用户同步", "AD"], "AD域同步", True, "implementation"),

    ("Q09", "客户端安装后没有人员怎么办？",
     [], "", False, "troubleshooting"),  # 知识库外

    ("Q10", "客户端连接不上服务端怎么排查？",
     ["技术培训", "端口", "产品使用手册"], "客户端服务端连接", True, "troubleshooting"),

    ("Q11", "安得EDG服务端有哪些端口？",
     ["技术培训", "端口映射"], "端口配置", True, "knowledge"),

    ("Q12", "文件外发后对方为什么打不开？",
     ["产品使用手册", "外发", "技术培训"], "文件外发", True, "troubleshooting"),

    ("Q13", "压缩包解压后的文件为什么又被加密了？",
     ["技术培训", "加密逻辑"], "透明加密", True, "troubleshooting"),

    ("Q14", "Word打开提示'你的组织策略阻止了我们为你完成此操作'怎么排查？",
     ["技术培训", "caseviews"], "故障排查", True, "troubleshooting"),

    ("Q15", "PDF文件打不开怎么办？",
     ["产品使用手册", "技术培训"], "故障排查", True, "troubleshooting"),

    ("Q16", "POC测试应该怎么开展？",
     ["技术培训"], "POC测试", True, "implementation"),

    ("Q17", "安得EDG的水印功能支持哪些类型？",
     ["产品使用手册", "技术培训", "水印"], "水印", True, "knowledge"),

    ("Q18", "客户端日志在哪里查看？如何收集日志排查问题？",
     ["技术培训", "产品使用手册", "日志"], "日志排查", True, "troubleshooting"),

    ("Q19", "服务端如何部署？支持哪些操作系统？",
     ["技术培训", "产品介绍"], "安装部署", True, "implementation"),

    ("Q20", "安得EDG是否支持与钉钉集成实现审批？",
     [], "", False, "knowledge"),  # 知识库外
]


def print_separator(title: str = ""):
    print(f"\n{'=' * 70}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 70}")


def check_retrieval_hit(evidences, expected_sources, expected_topic):
    """检查检索是否命中期望来源/主题"""
    if not expected_sources:
        return False, False

    for ev in evidences:
        meta = ev.get("metadata", {})
        src = meta.get("source_file", "")
        topic = meta.get("topic", "")

        hit_src = any(kw in src for kw in expected_sources)
        hit_topic = expected_topic.lower() in topic.lower() if expected_topic else False

        if hit_src or hit_topic:
            return True, hit_src

    return False, False


def check_answer_quality(answer: str, expected_sources, is_answerable: bool) -> dict:
    """检查回答质量"""
    checks = {
        "has_citations": False,
        "cites_correct_source": False,
        "refuses_correctly": False,
        "no_hallucination_flag": True,
    }

    # 检查 LLM 是否自我拒答（即使 pipeline 判断为 is_answerable）
    refuse_markers = [
        "没有找到足够依据", "知识库中没有", "无法确认",
        "知识库资料不足", "不能确认", "没有找到足够",
    ]
    checks["llm_refused"] = any(m in answer for m in refuse_markers)

    if not is_answerable:
        checks["refuses_correctly"] = checks["llm_refused"]
        return checks

    # 可回答问题：检查引用
    checks["has_citations"] = "参考资料" in answer or "《" in answer
    checks["cites_correct_source"] = any(
        src in answer for src in expected_sources
    ) if expected_sources else False

    # 简单幻觉检测：不应包含明显的编造内容
    hallucination_markers = [
        "作为一个AI", "根据我的知识", "据我所知",
        "虽然知识库中没有", "我认为", "一般来说",
    ]
    checks["no_hallucination_flag"] = not any(
        m in answer for m in hallucination_markers
    )

    return checks


def print_result(i: int, qid: str, question: str, result, checks: dict):
    """打印单题结果"""
    answerable_str = "可答" if result.expect_answerable else "不可答"
    is_answerable_str = "已答" if result.is_answerable else "拒答"

    status = "OK"
    if result.expect_answerable and not result.is_answerable:
        status = "MISS"  # 该答但拒答
    elif not result.expect_answerable and result.is_answerable:
        if checks.get("llm_refused"):
            status = "OK-LLM"  # LLM 自我拒答
        else:
            status = "HALLU"  # 不该答但答了

    print(f"\n  --- [{qid}] {status} | 分类={result.category}({result.category_confidence}) "
          f"| {answerable_str}->{is_answerable_str} | "
          f"检索={result.retrieval_count}->Rerank={result.rerank_count} | "
          f"耗时={result.timing.get('total', 0):.1f}s ---")
    print(f"  问题: {question}")
    print(f"  分类: {result.category} (confidence={result.category_confidence})")

    # 检索证据
    if result.evidences:
        top1 = result.evidences[0]
        meta = top1.get("metadata", {})
        vec_sim = top1.get("similarity", 0)
        rerank_score = top1.get("rerank_score", 0)
        print(f"  Top1证据: src={meta.get('source_file', '?')[:40]} "
              f"page={meta.get('page', '?')} "
              f"vec_sim={vec_sim:.3f} rerank={rerank_score:.3f}")
    else:
        print(f"  Top1证据: (无)")

    # 回答摘要
    answer_preview = result.answer[:300].replace("\n", " / ")
    print(f"  回答: {answer_preview}...")

    # 质量检查
    if checks:
        print(f"  质量: {checks}")

    return status


def main():
    print_separator("Phase 4: RAG 回答全链路 — 20 题测试")
    print(f"  Collection: {PHASE4_COLLECTION}")
    print(f"  Embedding:  {PHASE4_EMBEDDING_MODEL}")
    print(f"  LLM:        deepseek-v4-pro")

    # 磁盘检查
    usage = shutil.disk_usage("C:/")
    print(f"  C盘剩余:    {usage.free / 1024**3:.1f} GB")

    # ===== 初始化管线 =====
    print_separator("Step 1: 初始化 RAG 管线")
    t0 = time.time()
    pipeline = AnswerPipeline()
    # 触发加载
    _ = pipeline.embedding_model
    _ = pipeline.collection
    print(f"  管线初始化完成 ({time.time() - t0:.1f}s)")
    print(f"  Collection count: {pipeline.collection.count()}")

    # ===== 逐题测试 =====
    print_separator("Step 2: 20 题全链路测试")

    all_results = []
    stats = {
        "total": 0,
        "answerable_total": 0,
        "unanswerable_total": 0,
        "category_match": 0,
        "retrieval_hit_top1": 0,
        "retrieval_hit_top3": 0,
        "retrieval_hit_top5": 0,
        "correctly_answered": 0,
        "correctly_refused": 0,
        "false_negative": 0,  # 该答但拒答
        "false_positive": 0,  # 不该答但答了
        "has_citations": 0,
        "cites_correct_source": 0,
        "no_hallucination_flag": 0,
    }

    for i, (qid, question, expected_sources, expected_topic, answerable, expected_category) in enumerate(TEST_QUESTIONS):
        stats["total"] += 1
        if answerable:
            stats["answerable_total"] += 1
        else:
            stats["unanswerable_total"] += 1

        t_start = time.time()

        # 完整管线
        result = pipeline.answer(question)

        # 附加期望值用于报告
        result.expect_answerable = answerable
        result.expected_sources = expected_sources
        result.expected_topic = expected_topic
        result.expected_category = expected_category
        result.wall_time = round(time.time() - t_start, 1)

        # 检索命中检查
        hit, _ = check_retrieval_hit(result.evidences[:1], expected_sources, expected_topic)
        hit3, _ = check_retrieval_hit(result.evidences[:3], expected_sources, expected_topic)
        hit5, _ = check_retrieval_hit(result.evidences[:5], expected_sources, expected_topic)

        if answerable and hit:
            stats["retrieval_hit_top1"] += 1
        if answerable and hit3:
            stats["retrieval_hit_top3"] += 1
        if answerable and hit5:
            stats["retrieval_hit_top5"] += 1

        # 分类一致性
        if result.category == expected_category:
            stats["category_match"] += 1

        # 回答质量
        checks = check_answer_quality(result.answer, expected_sources, answerable)

        # 统计（含 LLM 自我拒答修正）
        if answerable and result.is_answerable:
            stats["correctly_answered"] += 1
        if not answerable and not result.is_answerable:
            stats["correctly_refused"] += 1
        if not answerable and result.is_answerable and checks.get("llm_refused", False):
            # LLM 自我拒答成功 → 视为正确拒答
            stats["correctly_refused"] += 1
        elif not answerable and result.is_answerable:
            # 真正误答：不该答但 LLM 也没拒
            stats["false_positive"] += 1
        if answerable and not result.is_answerable:
            stats["false_negative"] += 1

        if checks.get("has_citations") and answerable:
            stats["has_citations"] += 1
        if checks.get("cites_correct_source") and answerable:
            stats["cites_correct_source"] += 1
        if checks.get("no_hallucination_flag"):
            stats["no_hallucination_flag"] += 1

        # 打印此轮结果
        status = print_result(i + 1, qid, question, result, checks)

        all_results.append({
            "qid": qid,
            "question": question,
            "status": status,
            "category": result.category,
            "category_confidence": result.category_confidence,
            "expected_category": expected_category,
            "answerable_expected": answerable,
            "is_answerable": result.is_answerable,
            "answer": result.answer,
            "citations": result.citations,
            "top1_sim": result.evidences[0].get("similarity", 0) if result.evidences else 0,
            "top1_rerank": result.evidences[0].get("rerank_score", 0) if result.evidences else 0,
            "retrieval_count": result.retrieval_count,
            "rerank_count": result.rerank_count,
            "timing": result.timing,
            "checks": checks,
        })

    # ===== 汇总报告 =====
    print_separator("Phase 4 测试报告")

    a_total = stats["answerable_total"]
    u_total = stats["unanswerable_total"]

    print(f"\n  [分类准确率]")
    print(f"    分类匹配: {stats['category_match']}/{stats['total']} = {stats['category_match']/stats['total']:.1%}")

    print(f"\n  [检索命中率] (可回答 {a_total} 题)")
    if a_total > 0:
        print(f"    Top-1: {stats['retrieval_hit_top1']}/{a_total} = {stats['retrieval_hit_top1']/a_total:.1%}")
        print(f"    Top-3: {stats['retrieval_hit_top3']}/{a_total} = {stats['retrieval_hit_top3']/a_total:.1%}")
        print(f"    Top-5: {stats['retrieval_hit_top5']}/{a_total} = {stats['retrieval_hit_top5']/a_total:.1%}")

    print(f"\n  [可回答判断]")
    print(f"    正确答案:  {stats['correctly_answered']}/{a_total} (应答|可答题)")
    print(f"    正确拒答:  {stats['correctly_refused']}/{u_total} (拒答|不可答题)")
    print(f"    [MISS] 漏答: {stats['false_negative']} (该答但拒答)")
    print(f"    [HALLU] 误答: {stats['false_positive']} (不该答但答了)")

    print(f"\n  [引用质量] (可回答 {a_total} 题)")
    if a_total > 0:
        print(f"    包含引用:     {stats['has_citations']}/{a_total} = {stats['has_citations']/a_total:.1%}")
        print(f"    引用正确来源: {stats['cites_correct_source']}/{a_total} = {stats['cites_correct_source']/a_total:.1%}")
    print(f"    无幻觉标志:   {stats['no_hallucination_flag']}/{stats['total']} = {stats['no_hallucination_flag']/stats['total']:.1%}")

    print(f"\n  [耗时统计]")
    timings = {
        "classification": [],
        "retrieval": [],
        "rerank": [],
        "llm": [],
        "total": [],
    }
    for r in all_results:
        for k in timings:
            t = r.get("timing", {}).get(k, 0)
            if t:
                timings[k].append(t)

    for k, vals in timings.items():
        if vals:
            avg = sum(vals) / len(vals)
            print(f"    {k:<15}: avg={avg:.2f}s, min={min(vals):.2f}s, max={max(vals):.2f}s")

    print(f"\n  [逐题状态]")
    for r in all_results:
        print(f"    [{r['qid']}] {r['status']:<6} | cat={r['category']:<18} "
              f"exp={r['answerable_expected']} got={r['is_answerable']} | "
              f"top1_sim={r['top1_sim']:.3f} rerank={r['top1_rerank']:.3f} | "
              f"t={r['timing'].get('total', 0):.1f}s")

    # 保存详细结果
    out_path = os.path.join(os.path.dirname(__file__), "phase4_results.json")
    # 只保留文本，不保存完整 answer 中的长文本（避免 JSON 过大）
    slim_results = []
    for r in all_results:
        slim_results.append({
            "qid": r["qid"],
            "question": r["question"],
            "status": r["status"],
            "category": r["category"],
            "is_answerable": r["is_answerable"],
            "answer_preview": r["answer"][:200],
            "citations": r["citations"],
            "top1_sim": r["top1_sim"],
            "top1_rerank": r["top1_rerank"],
            "timing": r["timing"],
            "checks": r["checks"],
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(slim_results, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果已保存至: {out_path}")

    # 磁盘
    usage = shutil.disk_usage("C:/")
    print(f"\n  C盘剩余: {usage.free / 1024**3:.1f} GB")

    print_separator("Phase 4 测试完成")
    return all_results, stats


if __name__ == "__main__":
    main()
