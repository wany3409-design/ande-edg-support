#!/usr/bin/env python
"""
Phase 6: 真实售后问题压力测试运行器

用法:
  单条测试:  python scripts/test_real_support.py --query "问题文本"
  批量测试:  python scripts/test_real_support.py --batch

单条测试直接在命令行输出结果；批量测试从 data/real_support_test.jsonl
读取所有 human_evaluation 为空的问题，运行管线，回填结果。

测试完成后，人类评估标注填入 human_evaluation 字段后运行:
  python scripts/test_real_support.py --report
"""

import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.answer_pipeline import AnswerPipeline, PHASE4_COLLECTION, PHASE4_EMBEDDING_MODEL

TEST_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "real_support_test.jsonl",
)


def load_pipeline():
    print("加载管线...", end=" ", flush=True)
    p = AnswerPipeline()
    _ = p.embedding_model
    _ = p.collection
    print(f"OK (collection={p.collection.count()})")
    return p


def test_one(pipeline, query: str) -> dict:
    """运行单条测试，返回完整结果字典"""
    t0 = time.time()
    result = pipeline.answer(query)
    wall = round(time.time() - t0, 1)

    top1 = result.evidences[0] if result.evidences else {}
    meta = top1.get("metadata", {})

    return {
        "query": query,
        "category": f"{result.category}({result.category_confidence})",
        "is_answerable": result.is_answerable,
        "top1_sim": top1.get("similarity", 0),
        "top1_rerank": top1.get("rerank_score", 0),
        "top1_source": meta.get("source_file", "")[:50],
        "top1_section": meta.get("section", "")[:40],
        "retrieval_count": result.retrieval_count,
        "rerank_count": result.rerank_count,
        "wall_time": wall,
        "timing": result.timing,
        "answer": result.answer,
        "citations": result.citations,
        "grounding": result.grounding,
        # 预填行为标记
        "expected_behavior": "",
        "human_evaluation": "",
    }


def print_result(r: dict):
    """格式化打印单条测试结果"""
    print(f"\n{'=' * 70}")
    print(f"📋 Query: {r['query'][:100]}")
    print(f"📂 分类: {r['category']}")
    print(f"✅ 可回答: {r['is_answerable']}")
    print(f"📊 Top1: sim={r['top1_sim']:.4f} rerank={r['top1_rerank']:.4f}")
    print(f"📄 来源: {r['top1_source']} | {r['top1_section']}")
    print(f"⏱️  耗时: {r['wall_time']}s")
    print(f"📝 回答 ({len(r['answer'])}字符):")
    print(f"{r['answer'][:500]}")
    if len(r['answer']) > 500:
        print("    ...(截断)")
    print(f"\n📚 引用:")
    print(r['citations'][:300] if r['citations'] else "(无)")
    print(f"\n🔍 Grounding: {json.dumps(r['grounding'], ensure_ascii=False)}")
    print(f"{'=' * 70}")


def run_batch():
    """批量运行 JSONL 中未评估的问题"""
    pipeline = load_pipeline()

    # 读取现有记录
    records = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # 筛选需要运行的问题（human_evaluation 为空 且 answer 为空 或 query 有内容）
    pending = [r for r in records if not r.get("human_evaluation")]
    print(f"共 {len(records)} 条记录，{len(pending)} 条待测试\n")

    for i, rec in enumerate(pending):
        q = rec.get("query", "").strip()
        if not q:
            continue

        print(f"[{i + 1}/{len(pending)}] {q[:80]}...")
        result = test_one(pipeline, q)

        # 回填到记录
        rec["answer"] = result["answer"]
        rec["citations"] = result["citations"]
        rec["grounding"] = json.dumps(result["grounding"], ensure_ascii=False)
        rec["category"] = result["category"]
        if not rec.get("expected_behavior"):
            # 自动推断预期行为
            if result["is_answerable"]:
                # 检查 LLM 是否实际拒答
                refuse_markers = ["没有找到足够依据", "知识库中没有", "无法确认", "没有找到足够"]
                if any(m in result["answer"] for m in refuse_markers):
                    rec["expected_behavior"] = "INSUFFICIENT_EVIDENCE (LLM自拒)"
                else:
                    rec["expected_behavior"] = "ANSWERED"
            else:
                rec["expected_behavior"] = "INSUFFICIENT_EVIDENCE (管线拒答)"

        print(f"  → {'可回答' if result['is_answerable'] else '拒答'} | "
              f"sim={result['top1_sim']:.4f} rerank={result['top1_rerank']:.4f} | "
              f"{result['wall_time']}s")

    # 写回文件
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n批量测试完成，结果已写入 {TEST_FILE}")
    print("请对每条记录填写 human_evaluation 字段后运行 --report")


def run_report():
    """生成测试报告"""
    records = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("暂无测试记录")
        return

    total = len(records)
    answered = [r for r in records if r.get("expected_behavior") == "ANSWERED"]
    insufficient = [r for r in records if "INSUFFICIENT" in r.get("expected_behavior", "")]
    evaluated = [r for r in records if r.get("human_evaluation")]

    print(f"\n{'=' * 60}")
    print(f"  Phase 6: 真实售后问题测试报告")
    print(f"{'=' * 60}")

    print(f"\n  总问题数: {total}")
    print(f"  可回答:   {len(answered)}")
    print(f"  拒答:     {len(insufficient)}")

    if evaluated:
        correct = [r for r in evaluated if "CORRECT" in r.get("human_evaluation", "").upper()]
        wrong = [r for r in evaluated if "WRONG" in r.get("human_evaluation", "").upper()]
        partial = [r for r in evaluated if "PARTIAL" in r.get("human_evaluation", "").upper()]
        print(f"\n  人类评估 ({len(evaluated)}条):")
        print(f"    ✅ 正确:  {len(correct)}")
        print(f"    ⚠️ 部分:  {len(partial)}")
        print(f"    ❌ 错误:  {len(wrong)}")

    # 分类统计
    from collections import Counter
    cats = Counter(r.get("category", "").split("(")[0] for r in records)
    print(f"\n  问题类型分布:")
    for cat, cnt in cats.most_common():
        print(f"    {cat}: {cnt}")

    # 知识库缺口
    gaps = [r for r in records if "INSUFFICIENT" in r.get("expected_behavior", "")]
    if gaps:
        print(f"\n  知识库缺口 ({len(gaps)}条):")
        for g in gaps:
            print(f"    - {g.get('query', '')[:100]}")

    # 检索失败分析
    low_sim = [r for r in records if r.get("top1_sim", 0) < 0.25]
    if low_sim:
        print(f"\n  低相似度问题 (sim<0.25, {len(low_sim)}条):")
        for r in low_sim:
            print(f"    - sim={r.get('top1_sim', 0):.4f} | {r.get('query', '')[:100]}")

    # 高频问题类型
    print(f"\n  最常召回来源:")
    sources = Counter()
    for r in records:
        src = r.get("top1_source", "")
        if src:
            sources[src] += 1
    for src, cnt in sources.most_common(5):
        print(f"    {src}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 真实售后问题测试")
    parser.add_argument("--query", "-q", type=str, help="单条测试问题")
    parser.add_argument("--batch", "-b", action="store_true", help="批量运行 JSONL 中待测试问题")
    parser.add_argument("--report", "-r", action="store_true", help="生成测试报告")
    args = parser.parse_args()

    if args.report:
        run_report()
    elif args.batch:
        run_batch()
    elif args.query:
        pipeline = load_pipeline()
        result = test_one(pipeline, args.query)
        print_result(result)

        # 询问是否保存
        save = input("\n是否保存此结果到 JSONL？[y/N]: ").strip().lower()
        if save == "y":
            cat = input("预期行为类别 (knowledge/implementation/troubleshooting): ").strip()
            record = {
                "id": f"R{int(time.time())}",
                "query": args.query,
                "category": cat,
                "expected_behavior": "ANSWERED" if result["is_answerable"] else "INSUFFICIENT_EVIDENCE (管线拒答)",
                "answer": result["answer"],
                "citations": result["citations"],
                "grounding": json.dumps(result["grounding"], ensure_ascii=False),
                "human_evaluation": "",
            }
            with open(TEST_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"已保存至 {TEST_FILE}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
