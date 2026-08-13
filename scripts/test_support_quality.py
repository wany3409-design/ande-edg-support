#!/usr/bin/env python
"""
技术支持回答质量测试器

依据《docs/技术支持回答规范.md》，对当前 RAG 系统做「作为售后/实施技术支持助手」的
质量验收。目标不是刷 benchmark 分数，而是检验系统是否：
  1. 资料有明确依据时 → 具体可操作地回答（不把明确资料说成"可能/推测"）
  2. 资料只有部分依据时 → 已知说已知，缺失项明确标注"资料未说明"
  3. 资料完全没有依据时 → 直接拒答产品级结论
  4. 不编造菜单/按钮/参数，不用通用 IT 知识冒充 EDG 产品知识

每个问题输出四类标签之一：
  SUPPORTED      有明确依据，具体可操作
  PARTIAL        部分依据 + 明确标注缺失项
  UNSUPPORTED    无依据，正确拒答/标注无法确认
  HALLUCINATION  编造菜单/按钮/参数，或通用 IT 知识冒充产品知识（最严重，一票否决）

用法:
  python scripts/test_support_quality.py                 # 运行全部用例
  python scripts/test_support_quality.py --case anti_gpo # 只跑单个用例
  python scripts/test_support_quality.py --list          # 列出全部用例

重要：本测试不修改 Embedding / ChromaDB / Reranker / 阈值，只调用现有管线并做
规则式判定。判定器是启发式的：HALLUCINATION 检测（禁用词 + 编造菜单路径）是确定性
硬检查；SUPPORTED/PARTIAL/UNSUPPORTED 三分是启发式，用于透明报告而非调参。
"""

import sys
import os
import re
import time
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 下 stdout 默认 GBK，强制 UTF-8 避免中文乱码
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and _stream.encoding.lower() != "utf-8":
            _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.llm.answer_pipeline import AnswerPipeline
from src.llm.prompt_builder import classify_query

# ===== 四类判定标签 =====
LABEL_SUPPORTED = "SUPPORTED"
LABEL_PARTIAL = "PARTIAL"
LABEL_UNSUPPORTED = "UNSUPPORTED"
LABEL_HALLUCINATION = "HALLUCINATION"

# ===== 全局禁用词（通用 IT 知识，绝不允许冒充 EDG 产品知识）=====
# 判定规则：出现在回答中、且不在证据中、且不在用户问题原文中 → 视为编造/通用知识。
FORBIDDEN_GENERIC_IT = [
    # 组策略 / 注册表
    "gpedit", "组策略", "本地组策略", "组策略编辑器", "注册表", "regedit",
    "注册表编辑器",
    # Office 信任中心
    "信任中心", "受信任位置", "受信任文档",
    # 杀毒 / 安全软件
    "杀毒", "杀软", "防病毒", "火绒", "电脑管家", "腾讯管家", "毒霸",
    "360安全卫士", "360卫士",
    # 防火墙 / 系统
    "防火墙", "Defender", "安全模式",
    # 网盘
    "网盘", "百度网盘", "坚果云", "阿里云盘", "腾讯微云", "Dropbox",
    "OneDrive", "天翼云盘",
    # 其他通用"万能"建议
    "兼容模式", "管理员身份运行", "以管理员",
]

# ===== 拒答 / 资料未说明信号 =====
REFUSAL_SIGNALS = [
    "资料未说明", "资料无法确认", "无法确认", "资料不足", "未检索到",
    "证据不足", "没有找到", "不能确认", "暂不能确认", "没有明确给出",
    "不猜测", "无法从现有资料", "没有提供",
]

# ===== 强拒答信号（回答无法给出任何结论）=====
STRONG_REFUSAL = [
    "无法给出", "无法提供", "无法判断", "不能给出", "无法从现有资料",
    "证据不足以", "无法答复", "无法确定", "不能确认",
]

# ===== 正向依据信号（回答确实用知识库支撑了正向结论）=====
# 只保留系统提示词强制要求的三信息区分短语。拒答类回答也会提到"证据/来源"，
# 但那是用来解释"证据无关"，不能算正向依据，故不放进来。
_POSITIVE_PHRASES = [
    "资料明确支持", "资料明确说明", "明确支持", "明确说明",
    "跨文档整理", "根据多份资料整理",
]
# 「资料明确支持：无」这类是否定用法，不能算正向依据
_NEGATED_TAIL = re.compile(r"^[:：]?(无|没有|否|未)")


def _has_positive_evidence(answer: str) -> bool:
    """检测回答是否用知识库支撑了正向结论，排除「资料明确支持：无」这类否定用法。"""
    stripped = re.sub(r"[*\s　]", "", answer)
    for phrase in _POSITIVE_PHRASES:
        idx = 0
        while True:
            i = stripped.find(phrase, idx)
            if i == -1:
                break
            tail = stripped[i + len(phrase): i + len(phrase) + 4]
            if _NEGATED_TAIL.match(tail):
                idx = i + 1  # 否定用法，继续找下一处
                continue
            return True
    return False

# ===== 菜单路径（→ / -> / ➜ 链式结构）=====
_MENU_PATH_RE = re.compile(
    r"[一-鿿A-Za-z0-9]+(?:\s*(?:→|->|➜)\s*[一-鿿A-Za-z0-9]+)+"
)


@dataclass
class TestCase:
    """单个质量测试用例"""
    case_id: str
    category: str              # 测试类别（见规范 7 大类）
    question: str
    expect: Set[str]           # 可接受的标签集合
    forbidden: Tuple[str, ...] = ()  # 该用例额外禁用词


# ===== 测试用例 =====
# category 取值：full_config / partial_config / no_evidence / troubleshooting /
#               menu_path / multi_doc / anti_hallucination
TEST_CASES: List[TestCase] = [
    # 1. 有完整配置依据
    TestCase(
        "full_config_ad", "full_config",
        "安得EDG如何配置AD域同步？请给我具体操作步骤",
        expect={LABEL_SUPPORTED, LABEL_PARTIAL},
    ),
    # 2. 有部分配置依据（落地加密 —— 有明确参数但缺菜单路径）
    TestCase(
        "partial_landing_encrypt", "partial_config",
        "我要配置安得EDG的落地加密策略，怎么配置？请带我一步一步做",
        expect={LABEL_PARTIAL},
    ),
    TestCase(
        "partial_watermark", "partial_config",
        "安得EDG如何配置文档水印？",
        expect={LABEL_PARTIAL, LABEL_SUPPORTED},
    ),
    # 3. 完全没有产品依据
    TestCase(
        "no_evidence_dingtalk", "no_evidence",
        "安得EDG如何对接钉钉实现审批流程？",
        expect={LABEL_UNSUPPORTED},
    ),
    # 4. 故障排查
    TestCase(
        "troubleshoot_re_encrypt", "troubleshooting",
        "文件解密后为什么又自动加密了？",
        expect={LABEL_PARTIAL, LABEL_SUPPORTED},
        forbidden=("坚果云", "百度网盘", "安全模式", "网盘"),
    ),
    TestCase(
        "troubleshoot_conn_fail", "troubleshooting",
        "客户端打开文件提示连接不上服务器，怎么办？",
        expect={LABEL_PARTIAL, LABEL_UNSUPPORTED},
    ),
    # 5. 菜单路径
    TestCase(
        "menu_path_config", "menu_path",
        "安得EDG在哪里配置文件加密策略？给我具体菜单路径",
        expect={LABEL_SUPPORTED, LABEL_PARTIAL},
    ),
    # 6. 多文档综合
    TestCase(
        "multi_doc_overview", "multi_doc",
        "请完整介绍安得EDG的落地加解密功能：原理、配置方法、验证方式",
        expect={LABEL_SUPPORTED, LABEL_PARTIAL},
    ),
    # 7. 防止通用知识脑补
    TestCase(
        "anti_gpo", "anti_hallucination",
        "客户打不开 Word，提示“你的组织策略阻止了我们为你完成此操作”",
        expect={LABEL_UNSUPPORTED, LABEL_PARTIAL},
        forbidden=("gpedit", "组策略", "注册表", "信任中心", "regedit"),
    ),
    TestCase(
        "anti_virus", "anti_hallucination",
        "文件突然打不开了，是不是中了勒索病毒？怎么恢复？",
        expect={LABEL_UNSUPPORTED, LABEL_PARTIAL},
        forbidden=("杀毒", "防火墙", "安全模式", "火绒"),
    ),
]


def build_pool_text(docs: List[dict]) -> str:
    """把候选证据池拼成纯文本，用于 grounding 检查。"""
    parts = []
    for d in docs:
        meta = d.get("metadata", {}) or {}
        sec = meta.get("section", "") or ""
        txt = d.get("document", "") or d.get("text", "") or ""
        if sec:
            parts.append(f"{sec} {txt}")
        else:
            parts.append(txt)
    return " ".join(parts).lower()


def extract_menu_paths(answer: str) -> List[str]:
    """提取回答中的菜单路径链（→ / -> 连接）。"""
    return [m for m in _MENU_PATH_RE.findall(answer)]


def _is_fabricated_path(chain: str, pool_text: str) -> bool:
    """判断一条菜单路径链是否编造：多数段落不在证据池中。"""
    segs = [
        s.strip() for s in re.split(r"→|->|➜", chain)
        if len(s.strip()) >= 2
    ]
    if len(segs) < 2:
        return False
    absent = [s for s in segs if s.lower() not in pool_text]
    # 至少 2 段，且超过一半段落不在证据中 → 判定编造
    return len(absent) >= len(segs) / 2 and len(absent) >= 1


# ===== 否定语义豁免 =====
# 系统提示词会明确列出 gpedit.msc/注册表/信任中心 等禁用词，因此 LLM 在"正确拒答"时
# 可能以"不能使用 X 冒充产品知识"的方式提到它们。这类否定/禁止语境不应判为 HALLUCINATION。
_NEGATION_MARKERS = [
    "不能", "禁止", "不得", "不要", "避免", "不应", "不可", "不会", "不用",
    "无法", "不使用", "不提供", "不主动", "不做", "不得使用", "不可使用",
    "不能使用", "不要使用", "不建议", "而非", "而不是", "不是",
]

# 诊断性提问语境：禁用词出现在"是否…有关/相关"这类向用户求证的问题里，
# 是询问而非建议，同样不应判为编造。
# 例如"是否与 Office 自身组策略、信任中心、加载项、域策略有关"——这是在排查
# 该报错是否属于非 EDG 的通用 Windows/Office 问题，而非建议用户去改组策略。
_QUESTION_MARKERS = ["是否", "是不是", "有没有", "是否有", "是否与"]


def _term_used_positively(answer: str, term: str) -> bool:
    """判断禁用词是否被"正面使用"（作为解决方案建议），而非否定/禁止/诊断提问语境。

    按句子切分后，仅当 term 所在句子的 term 之前部分同时不含否定标记、也不含
    诊断性提问标记时，才视为正面使用。
    例如"不能使用 gpedit.msc、注册表…冒充"里 gpedit/注册表都被豁免；
    "是否与 Office 自身组策略…有关"里的组策略也被豁免（是询问不是建议）；
    而"可以在 gpedit.msc 里修改"里的 gpedit 会被判定为正面使用。
    """
    for sent in re.split(r"[。！？；\n]", answer):
        i = sent.find(term)
        if i == -1:
            continue
        before = sent[:i]
        if any(m in before for m in _NEGATION_MARKERS):
            continue  # 该句 term 前有否定语义 → 豁免
        if any(m in before for m in _QUESTION_MARKERS):
            continue  # 该句是向用户求证（是否…有关）→ 豁免
        return True  # 正面使用 → 判定编造
    return False


def judge(
    answer: str,
    pool_text: str,
    question: str = "",
    extra_forbidden: Tuple[str, ...] = (),
) -> Tuple[str, str]:
    """
    规则式判定回答质量标签。

    返回 (标签, 判定理由)。判定优先级：
      1. HALLUCINATION（禁用词 / 编造菜单路径）—— 确定性硬检查
      2. UNSUPPORTED（强拒答且无任何正向依据）
      3. PARTIAL（有依据 + 标注缺失项）
      4. SUPPORTED（有依据且未标注缺失）
    """
    a = (answer or "").strip()
    if not a:
        return LABEL_UNSUPPORTED, "回答为空"

    # 1. 禁用词检查（正面使用 → 编造；否定/禁止语境 → 豁免）
    for term in list(FORBIDDEN_GENERIC_IT) + list(extra_forbidden):
        if term in question or term in pool_text:
            continue  # 问题原文或证据中出现，不是编造
        if _term_used_positively(a, term):
            return LABEL_HALLUCINATION, f"出现通用 IT 知识/编造词:「{term}」"

    # 2. 编造菜单路径检查
    for chain in extract_menu_paths(a):
        if _is_fabricated_path(chain, pool_text):
            return LABEL_HALLUCINATION, f"编造菜单路径:「{chain}」(证据中无对应段落)"

    # 3. 拒答 / 依据信号
    refusal = any(s in a for s in REFUSAL_SIGNALS)
    strong_refusal = any(s in a for s in STRONG_REFUSAL)
    positive = _has_positive_evidence(a)

    if strong_refusal and not positive:
        return LABEL_UNSUPPORTED, "正确拒答：明确无法从资料确认，且未给出正向结论"
    if refusal:
        return LABEL_PARTIAL, "部分依据：已知说已知，缺失项标注「资料未说明」"
    if positive or len(a) >= 40:
        return LABEL_SUPPORTED, "有明确依据，回答具体"
    return LABEL_UNSUPPORTED, "回答过短且无依据"


def load_pipeline() -> AnswerPipeline:
    print("加载管线...", end=" ", flush=True)
    p = AnswerPipeline()
    _ = p.embedding_model
    _ = p.collection
    print(f"OK (collection={p.collection.count()})")
    return p


def run_one(pipeline: AnswerPipeline, tc: TestCase) -> dict:
    """运行单个用例，返回完整结果字典。"""
    t0 = time.time()
    result = pipeline.answer(tc.question)
    wall = round(time.time() - t0, 1)

    # 用完整候选池做 grounding（比 top-5 更严格、更诚实）
    cat = classify_query(tc.question).category
    try:
        pool = pipeline._multi_retrieve(
            tc.question, cat, n_results=pipeline.top_k_candidate
        )
    except Exception:
        pool = result.evidences
    pool_text = build_pool_text(pool)

    label, reason = judge(
        result.answer, pool_text, question=tc.question, extra_forbidden=tc.forbidden
    )

    top1 = result.evidences[0] if result.evidences else {}
    meta = top1.get("metadata", {})

    return {
        "case_id": tc.case_id,
        "category": tc.category,
        "question": tc.question,
        "label": label,
        "reason": reason,
        "expect": sorted(tc.expect),
        "pass": label in tc.expect,
        "pipeline_category": f"{result.category}({result.category_confidence})",
        "is_answerable": result.is_answerable,
        "top1_sim": top1.get("similarity", 0),
        "top1_rerank": top1.get("rerank_score", 0),
        "top1_source": meta.get("source_file", "")[:60],
        "wall_time": wall,
        "answer": result.answer,
        "citations": result.citations,
        "grounding": result.grounding,
    }


def print_one(r: dict, idx: int, total: int):
    """打印单条结果。"""
    mark = "✅" if r["pass"] else "❌"
    h = " 🔴" if r["label"] == LABEL_HALLUCINATION else ""
    print(f"\n{'─' * 70}")
    print(f"[{idx}/{total}] {mark} {r['case_id']}  ({r['category']}){h}")
    print(f"  问: {r['question']}")
    print(f"  判定: {r['label']}   (期望: {'/'.join(r['expect'])})")
    print(f"  理由: {r['reason']}")
    print(f"  管线: {r['pipeline_category']} | 可回答={r['is_answerable']} | "
          f"top1 sim={r['top1_sim']:.4f} rerank={r['top1_rerank']:.4f} | "
          f"{r['wall_time']}s | 来源={r['top1_source']}")
    if r["label"] == LABEL_HALLUCINATION or not r["pass"]:
        print(f"  回答摘录: {r['answer'][:300]}")
    if r["label"] == LABEL_HALLUCINATION:
        print(f"  引用: {r['citations'][:200]}")


def print_summary(results: List[dict]):
    """打印汇总。"""
    from collections import Counter
    labels = Counter(r["label"] for r in results)
    n = len(results)
    n_pass = sum(1 for r in results if r["pass"])
    n_hallu = labels.get(LABEL_HALLUCINATION, 0)

    print(f"\n{'=' * 70}")
    print(f"  技术支持回答质量测试 —— 汇总")
    print(f"{'=' * 70}")
    print(f"  总用例: {n}   通过: {n_pass}   失败: {n - n_pass}")
    print(f"\n  标签分布:")
    for lbl, name in [
        (LABEL_SUPPORTED, "SUPPORTED  (有明确依据)"),
        (LABEL_PARTIAL, "PARTIAL    (部分依据)"),
        (LABEL_UNSUPPORTED, "UNSUPPORTED(正确拒答)"),
        (LABEL_HALLUCINATION, "HALLUCINATION(编造)"),
    ]:
        cnt = labels.get(lbl, 0)
        bar = "█" * cnt
        print(f"    {name:22} {cnt:2}  {bar}")

    # 分类统计
    cats = Counter(r["category"] for r in results)
    print(f"\n  测试类别覆盖:")
    for cat, cnt in cats.most_common():
        print(f"    {cat}: {cnt}")

    # 结论
    print(f"\n{'=' * 70}")
    if n_hallu == 0:
        print(f"  ✅ 验收底线通过：HALLUCINATION = 0")
    else:
        print(f"  ❌ 验收底线未通过：存在 {n_hallu} 条 HALLUCINATION（编造/通用知识）")
    print(f"{'=' * 70}")


def save_results(results: List[dict], out_path: str):
    """把结果写入 JSON 文件，便于人工复查。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {out_path}")


def main():
    parser = argparse.ArgumentParser(description="技术支持回答质量测试")
    parser.add_argument("--case", "-c", type=str, help="只运行指定 case_id")
    parser.add_argument("--list", "-l", action="store_true", help="列出全部用例")
    args = parser.parse_args()

    if args.list:
        for tc in TEST_CASES:
            print(f"  {tc.case_id:24} [{tc.category}]  {tc.question}")
        return

    cases = TEST_CASES
    if args.case:
        cases = [tc for tc in TEST_CASES if tc.case_id == args.case]
        if not cases:
            print(f"未找到 case_id={args.case}，可用 --list 查看全部")
            return

    pipeline = load_pipeline()
    results = []
    for i, tc in enumerate(cases, 1):
        print(f"\n运行中 [{i}/{len(cases)}] {tc.case_id}...", flush=True)
        r = run_one(pipeline, tc)
        results.append(r)
        print_one(r, i, len(cases))

    print_summary(results)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "support_quality_results.json",
    )
    save_results(results, out_path)


if __name__ == "__main__":
    main()
