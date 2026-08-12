#!/usr/bin/env python
"""
Phase 4.5: 知识质量和引用准确性修复 — 20 题全链路测试

改进点:
- V3 collection (400-char chunks, improved HTML parser, slide-based parsing)
- Citation Grounding 检查
- 改进的问题分类
- POC/测试/部署 同义词 Reranker
- 结果对比 Phase 4 vs Phase 4.5
"""

import sys
import os
import time
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CHROMA_PERSIST_DIR
from src.llm.answer_pipeline import AnswerPipeline, PHASE4_COLLECTION, PHASE4_EMBEDDING_MODEL

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
     [], "", False, "troubleshooting"),

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

    ("Q16", "安得EDG怎么做POC测试？",
     ["技术培训"], "测试验证", True, "implementation"),

    ("Q17", "安得EDG的水印功能支持哪些类型？",
     ["产品使用手册", "技术培训", "水印"], "水印", True, "knowledge"),

    ("Q18", "客户端日志在哪里查看？如何收集日志排查问题？",
     ["技术培训", "产品使用手册", "日志"], "日志排查", True, "troubleshooting"),

    ("Q19", "服务端如何部署？支持哪些操作系统？",
     ["技术培训", "产品介绍"], "安装部署", True, "implementation"),

    ("Q20", "安得EDG是否支持与钉钉集成实现审批？",
     [], "", False, "knowledge"),
]


def print_separator(title: str = ""):
    print(f"\n{'=' * 70}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 70}")


def check_retrieval_hit(evidences, expected_sources, expected_topic):
    if not expected_sources and not expected_topic:
        return False, False
    for ev in evidences:
        meta = ev.get("metadata", {})
        src = meta.get("source_file", "")
        topic = meta.get("topic", "")
        hit_src = any(kw in src for kw in expected_sources) if expected_sources else False
        hit_topic = expected_topic.lower() in topic.lower() if expected_topic else False
        if hit_src or hit_topic:
            return True, hit_src
    return False, False


def check_answer_quality(answer: str, expected_sources, is_answerable: bool) -> dict:
    checks = {
        "has_citations": False,
        "cites_correct_source": False,
        "refuses_correctly": False,
        "no_hallucination_flag": True,
        "llm_refused": False,
    }

    refuse_markers = [
        "没有找到足够依据", "知识库中没有", "无法确认",
        "知识库资料不足", "不能确认", "没有找到足够",
    ]
    checks["llm_refused"] = any(m in answer for m in refuse_markers)

    if not is_answerable:
        checks["refuses_correctly"] = checks["llm_refused"]
        return checks

    checks["has_citations"] = "参考资料" in answer or "《" in answer
    checks["cites_correct_source"] = any(
        src in answer for src in expected_sources
    ) if expected_sources else False

    hallucination_markers = [
        "作为一个AI", "根据我的知识", "据我所知",
        "虽然知识库中没有", "我认为", "一般来说",
    ]
    checks["no_hallucination_flag"] = not any(
        m in answer for m in hallucination_markers
    )
    return checks


def main():
    print_separator("Phase 4.5: 知识质量修复 — 20 题全链路测试")
    print(f"  Collection: {PHASE4_COLLECTION}")
    print(f"  Embedding:  {PHASE4_EMBEDDING_MODEL}")
    print(f"  LLM:        deepseek-v4-pro")

    usage = shutil.disk_usage("C:/")
    print(f"  C盘剩余:    {usage.free / 1024**3:.1f} GB")

    # Init
    print_separator("Step 1: 初始化 RAG 管线")
    t0 = time.time()
    pipeline = AnswerPipeline()
    _ = pipeline.embedding_model
    _ = pipeline.collection
    print(f"  管线初始化完成 ({time.time() - t0:.1f}s)")
    print(f"  Collection count: {pipeline.collection.count()}")

    # Test
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
        "false_negative": 0,
        "false_positive": 0,
        "has_citations": 0,
        "cites_correct_source": 0,
        "no_hallucination_flag": 0,
        "grounding_ok": 0,
    }

    for i, (qid, question, expected_sources, expected_topic, answerable, expected_category) in enumerate(TEST_QUESTIONS):
        stats["total"] += 1
        if answerable:
            stats["answerable_total"] += 1
        else:
            stats["unanswerable_total"] += 1

        t_start = time.time()
        result = pipeline.answer(question)
        result.expect_answerable = answerable
        result.expected_sources = expected_sources
        result.expected_topic = expected_topic
        result.expected_category = expected_category
        result.wall_time = round(time.time() - t_start, 1)

        # Retrieval hit check
        hit, _ = check_retrieval_hit(result.evidences[:1], expected_sources, expected_topic)
        hit3, _ = check_retrieval_hit(result.evidences[:3], expected_sources, expected_topic)
        hit5, _ = check_retrieval_hit(result.evidences[:5], expected_sources, expected_topic)

        if answerable and hit: stats["retrieval_hit_top1"] += 1
        if answerable and hit3: stats["retrieval_hit_top3"] += 1
        if answerable and hit5: stats["retrieval_hit_top5"] += 1

        # Classification
        if result.category == expected_category:
            stats["category_match"] += 1

        # Answer quality
        checks = check_answer_quality(result.answer, expected_sources, answerable)

        # Stats
        if answerable and result.is_answerable:
            stats["correctly_answered"] += 1
        if not answerable and not result.is_answerable:
            stats["correctly_refused"] += 1
        if not answerable and result.is_answerable and checks.get("llm_refused", False):
            stats["correctly_refused"] += 1
        elif not answerable and result.is_answerable:
            stats["false_positive"] += 1
        if answerable and not result.is_answerable:
            stats["false_negative"] += 1

        if checks.get("has_citations") and answerable:
            stats["has_citations"] += 1
        if checks.get("cites_correct_source") and answerable:
            stats["cites_correct_source"] += 1
        if checks.get("no_hallucination_flag"):
            stats["no_hallucination_flag"] += 1
        if result.grounding.get("grounded", False) and answerable and result.is_answerable:
            stats["grounding_ok"] += 1

        # Display
        status = "OK"
        if result.expect_answerable and not result.is_answerable:
            status = "MISS"
        elif not result.expect_answerable and result.is_answerable:
            status = "OK-LLM" if checks.get("llm_refused") else "HALLU"

        top1_sim_print = result.evidences[0].get("similarity", 0) if result.evidences else 0
        top1_rerank_print = result.evidences[0].get("rerank_score", 0) if result.evidences else 0
        grounding_info = f"ground={'Y' if result.grounding.get('grounded', False) else 'N'}"
        print(f"\n  [{qid}] {status} | cat={result.category}({result.category_confidence}) "
              f"| exp={answerable} got={result.is_answerable} | "
              f"top1_sim={top1_sim_print:.3f} rerank={top1_rerank_print:.3f} | "
              f"{grounding_info} | t={result.wall_time:.1f}s")
        if result.evidences:
            meta = result.evidences[0].get("metadata", {})
            print(f"    Top1: {meta.get('source_file', '?')[:45]} p{meta.get('page', '?')} "
                  f"s={meta.get('section', '')[:40]}")
        print(f"    Answer: {result.answer[:200].replace(chr(10), ' / ')}...")

        all_results.append({
            "qid": qid, "question": question, "status": status,
            "category": result.category, "category_confidence": result.category_confidence,
            "answerable_expected": answerable, "is_answerable": result.is_answerable,
            "answer_preview": result.answer[:300],
            "top1_sim": top1_sim_print, "top1_rerank": top1_rerank_print,
            "retrieval_count": result.retrieval_count, "rerank_count": result.rerank_count,
            "timing": result.timing, "checks": checks,
            "grounding": result.grounding,
        })

    # ===== Report =====
    print_separator("Phase 4.5 测试报告")

    a_total = stats["answerable_total"]
    u_total = stats["unanswerable_total"]

    print(f"\n  [分类准确率]")
    print(f"    分类匹配: {stats['category_match']}/{stats['total']} = {stats['category_match']/stats['total']:.1%}")

    print(f"\n  [检索命中率] (可回答 {a_total} 题)")
    if a_total > 0:
        t1 = stats['retrieval_hit_top1']/a_total
        t3 = stats['retrieval_hit_top3']/a_total
        t5 = stats['retrieval_hit_top5']/a_total
        print(f"    Top-1: {stats['retrieval_hit_top1']}/{a_total} = {t1:.1%}")
        print(f"    Top-3: {stats['retrieval_hit_top3']}/{a_total} = {t3:.1%}")
        print(f"    Top-5: {stats['retrieval_hit_top5']}/{a_total} = {t5:.1%}")

    print(f"\n  [可回答判断]")
    print(f"    正确答案:    {stats['correctly_answered']}/{a_total} (应答|可答题)")
    print(f"    正确拒答:    {stats['correctly_refused']}/{u_total} (拒答|不可答题)")
    print(f"    [MISS] 漏答: {stats['false_negative']} (该答但拒答)")
    print(f"    [HALLU] 误答: {stats['false_positive']} (不该答但答了)")

    print(f"\n  [引用质量] (可回答 {a_total} 题)")
    if a_total > 0:
        print(f"    包含引用:     {stats['has_citations']}/{a_total} = {stats['has_citations']/a_total:.1%}")
        print(f"    引用正确来源: {stats['cites_correct_source']}/{a_total} = {stats['cites_correct_source']/a_total:.1%}")
    print(f"    无幻觉标志:   {stats['no_hallucination_flag']}/{stats['total']} = {stats['no_hallucination_flag']/stats['total']:.1%}")

    print(f"\n  [Citation Grounding]")
    if a_total > 0:
        answered = sum(1 for r in all_results if r["answerable_expected"] and r["is_answerable"])
        if answered > 0:
            print(f"    Grounding 通过: {stats['grounding_ok']}/{answered} = {stats['grounding_ok']/answered:.1%}")

    print(f"\n  [耗时统计]")
    timings = {"classification": [], "retrieval": [], "rerank": [], "llm": [], "grounding": [], "total": []}
    for r in all_results:
        for k in timings:
            t = r.get("timing", {}).get(k, 0)
            if t:
                timings[k].append(t)
    for k, vals in timings.items():
        if vals:
            avg = sum(vals) / len(vals)
            print(f"    {k:<15}: avg={avg:.2f}s")

    # ===== Phase 4 vs Phase 4.5 comparison =====
    print_separator("Phase 4 vs Phase 4.5 对比")
    # Phase 4 results (from previous run)
    p4 = {
        "category": 0.800,
        "top1": 0.667, "top3": 0.778, "top5": 0.944,
        "answered": "17/18", "refused": "2/2",
        "miss": 1, "hallu": 0,
        "citations": 0.944, "cite_correct": 0.778,
        "no_hallu": 1.000,
    }
    p5 = {
        "category": stats['category_match']/stats['total'],
        "top1": stats['retrieval_hit_top1']/a_total if a_total else 0,
        "top3": stats['retrieval_hit_top3']/a_total if a_total else 0,
        "top5": stats['retrieval_hit_top5']/a_total if a_total else 0,
        "answered": f"{stats['correctly_answered']}/{a_total}",
        "refused": f"{stats['correctly_refused']}/{u_total}",
        "miss": stats['false_negative'], "hallu": stats['false_positive'],
        "citations": stats['has_citations']/a_total if a_total else 0,
        "cite_correct": stats['cites_correct_source']/a_total if a_total else 0,
        "no_hallu": stats['no_hallucination_flag']/stats['total'],
    }

    print(f"\n  {'指标':<22} {'Phase 4':<16} {'Phase 4.5':<16} {'变化'}")
    print(f"  {'-'*60}")
    metrics = [
        ("分类准确率", "category", ".0%"),
        ("Top-1 命中率", "top1", ".1%"),
        ("Top-3 命中率", "top3", ".1%"),
        ("Top-5 命中率", "top5", ".1%"),
        ("正确回答", "answered", "s"),
        ("正确拒答", "refused", "s"),
        ("漏答 (MISS)", "miss", "d"),
        ("误答 (HALLU)", "hallu", "d"),
        ("引用率", "citations", ".1%"),
        ("引用正确率", "cite_correct", ".1%"),
        ("无幻觉率", "no_hallu", ".1%"),
    ]
    for name, key, fmt in metrics:
        v4 = p4[key]
        v5 = p5[key]
        if fmt == ".0%":
            s4, s5 = f"{v4:.0%}", f"{v5:.0%}"
        elif fmt == ".1%":
            s4, s5 = f"{v4:.1%}", f"{v5:.1%}"
        elif fmt == "d":
            s4, s5 = str(v4), str(v5)
        else:
            s4, s5 = str(v4), str(v5)

        # Change indicator
        if fmt in (".0%", ".1%"):
            diff = v5 - v4
            if diff > 0.01:
                change = f"+{diff:.1%}"
            elif diff < -0.01:
                change = f"{diff:.1%}"
            else:
                change = "-"
        elif fmt == "d":
            if v5 < v4:
                change = f"-{v4 - v5} (improved)"
            elif v5 > v4:
                change = f"+{v5 - v4}"
            else:
                change = "-"
        else:
            change = ""
        print(f"  {name:<22} {s4:<16} {s5:<16} {change}")

    # Phase 4.5 specific
    print(f"\n  [Phase 4.5 新增]")
    if a_total > 0:
        answered = sum(1 for r in all_results if r["answerable_expected"] and r["is_answerable"])
        print(f"    Citation Grounding: {stats['grounding_ok']}/{answered} = {stats['grounding_ok']/answered:.1%}" if answered else "    Citation Grounding: N/A")
    print(f"    Chunk size: 200 -> 400")
    print(f"    HTML parser: 50-line groups -> slide-based ({pipeline.collection.count()} total chunks)")

    usage = shutil.disk_usage("C:/")
    print(f"\n  C盘剩余: {usage.free / 1024**3:.1f} GB")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "phase4_5_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  详细结果已保存至: {out_path}")

    print_separator("Phase 4.5 测试完成")
    return all_results, stats


if __name__ == "__main__":
    main()
