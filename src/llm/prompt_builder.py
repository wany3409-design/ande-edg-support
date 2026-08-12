"""
Prompt 构建器

负责：
1. 问题分类
2. 构建 System Prompt
3. 组装证据 + 用户问题 -> 最终 messages
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ClassifiedQuery:
    """分类后的问题"""
    query: str
    category: str  # knowledge / implementation / troubleshooting
    confidence: str  # high / medium / low


# ===== System Prompt =====
SYSTEM_PROMPT = """你是安得EDG（安得电子文档安全管理系统）的专业技术支持助手。

## 核心约束（必须遵守）
1. 你只能基于下方【参考资料】中提供的知识库内容来回答安得EDG产品的具体问题。
2. **禁止根据通用知识猜测**安得EDG的具体菜单名称、配置项、路径、端口号、策略行为或产品功能。
3. 如果【参考资料】中找不到足够依据，必须明确回答："当前知识库中没有找到足够依据确认该问题。" 然后可以提供"建议排查方向"，但必须标注"以下为推测/排查建议，非产品官方结论"。
4. 回答要专业、简洁、可执行。不使用空泛的AI套话。不编造内容。

## 回答格式

### 知识问答类
直接给出答案，条理清晰。
末尾列出【参考资料】。

### 实施配置类
【前置条件】
【操作步骤】
【验证方法】
【参考资料】

### 故障排查类
【问题判断】
【可能原因】
【排查步骤】
【解决方案】
【需要补充的信息】
【参考资料】

## 【参考资料】格式要求
每条引用必须包含：
- 文件名
- 页码（如有）
- 章节（如有）

示例：
> 1. 《安得卫士（AndSec）产品使用手册》第68页，1.2.2.1.1 策略配置
> 2. 《安得卫士6.0.8技术培训》第12页，常见问题

## 通用原则
- 如果用户提供了环境信息（操作系统、版本等），优先基于这些信息给出针对性建议
- 如果缺少关键信息，主动列出需要补充的内容
- 对于涉及系统修改的操作，提醒用户提前备份"""


def classify_query(query: str) -> ClassifiedQuery:
    """规则分类问题类型 (v2: 增加更多关键词 + 优先级逻辑)"""
    q = query.lower()

    # 故障排查关键词 (最高优先级 — 如果是排查类问题优先归类为 troubleshooting)
    troubleshooting_keywords = [
        "为什么", "怎么办", "打不开", "没生效", "不生效", "失败",
        "报错", "错误", "连接不上", "无法", "不能", "提示",
        "排查", "异常", "崩溃", "闪退", "卡住", "怎么排查",
        "如何排查", "没反应", "不行", "出问题", "显示",
        "弹窗", "警告", "阻止", "不工作",
    ]
    # 实施配置关键词
    implementation_keywords = [
        "怎么配置", "如何配置", "怎么部署", "如何部署", "怎么安装",
        "如何安装", "怎么设置", "如何设置", "怎么做", "如何做",
        "配置方法", "设置方法", "安装步骤", "部署步骤",
    ]
    # 知识问答关键词
    knowledge_keywords = [
        "支持哪些", "有哪些", "是什么", "功能", "版本",
        "区别", "介绍", "说明", "支持",
        "是否支持", "能不能", "可以", "兼容",
    ]

    trouble_score = sum(1 for kw in troubleshooting_keywords if kw in q)
    impl_score = sum(1 for kw in implementation_keywords if kw in q)
    know_score = sum(1 for kw in knowledge_keywords if kw in q)

    # 强故障信号：问题类疑问词 + 负面结果词
    strong_trouble = any(kw in q for kw in ["为什么", "怎么办", "打不开", "报错", "连接不上", "怎么排查", "如何排查", "阻止"])
    strong_impl = any(kw in q for kw in ["怎么配置", "如何配置", "怎么部署", "如何部署", "怎么安装", "如何安装", "如何设置", "怎么做", "如何做"])

    # 优先强信号
    if strong_trouble and trouble_score >= strong_impl:
        confidence = "high" if trouble_score >= 2 else "medium"
        return ClassifiedQuery(query=query, category="troubleshooting", confidence=confidence)
    if strong_impl and impl_score >= trouble_score:
        confidence = "high" if impl_score >= 2 else "medium"
        return ClassifiedQuery(query=query, category="implementation", confidence=confidence)

    scores = {
        "troubleshooting": trouble_score,
        "implementation": impl_score,
        "knowledge": know_score,
    }
    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score == 0:
        return ClassifiedQuery(query=query, category="knowledge", confidence="low")
    elif best_score >= 2:
        return ClassifiedQuery(query=query, category=best, confidence="high")
    else:
        return ClassifiedQuery(query=query, category=best, confidence="medium")


def build_evidence_text(evidences: List[Dict[str, Any]]) -> str:
    """将证据列表组装为纯文本"""
    parts = []
    for i, ev in enumerate(evidences, 1):
        meta = ev.get("metadata", {})
        text = ev.get("document", "") or ev.get("text", "")

        src = meta.get("source_file", "未知来源")
        page = meta.get("page", "")
        section = meta.get("section", "")
        source_level = meta.get("source_level", "")

        header = f"[证据 {i}]"
        if page:
            header += f" 第{page}页"
        if section:
            header += f" {section}"

        parts.append(f"{header}\n来源: {src}\n内容: {text}\n")

    return "\n".join(parts)


def build_citation_text(evidences: List[Dict[str, Any]]) -> str:
    """构建引用来源文本"""
    lines = ["\n【参考资料】"]
    seen = set()
    for i, ev in enumerate(evidences, 1):
        meta = ev.get("metadata", {})
        src = meta.get("source_file", "未知来源")
        page = meta.get("page", "")
        section = meta.get("section", "")

        key = f"{src}:{page}:{section}"
        if key in seen:
            continue
        seen.add(key)

        citation = f"{i}. 《{src}》"
        if page:
            citation += f" 第{page}页"
        if section:
            citation += f"，{section}"
        lines.append(citation)

    return "\n".join(lines)


def build_messages(
    query: str,
    evidences: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """构建发送给 DeepSeek 的完整 messages"""

    evidence_text = build_evidence_text(evidences)
    classification = classify_query(query)

    # 根据问题类型给出格式提示
    format_hint = ""
    if classification.category == "troubleshooting":
        format_hint = "\n请按照【问题判断】→【可能原因】→【排查步骤】→【解决方案】→【需要补充的信息】→【参考资料】格式回答。"
    elif classification.category == "implementation":
        format_hint = "\n请按照【前置条件】→【操作步骤】→【验证方法】→【参考资料】格式回答。"

    user_message = f"""## 参考资料
{evidence_text}

## 用户问题
{query}
{format_hint}

请根据以上参考资料回答。如证据不足，请明确说明。"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_message})
    return messages


def ground_citations(
    answer: str,
    evidences: List[Dict[str, Any]],
    min_overlap_ratio: float = 0.15,
) -> dict:
    """
    Citation Grounding 检查

    检查回答中的关键实体/短语是否在证据文本中出现。
    使用简单的关键词/实体匹配，不依赖大型模型。

    返回:
      {
        "grounded": bool,         # 是否有足够依据
        "claims_checked": int,    # 检查了几条事实声明
        "claims_grounded": int,   # 有几条能在证据中找到
        "unverified": list[str],  # 无法验证的声明
      }
    """
    # 提取证据中的所有文本
    evidence_texts = []
    for ev in evidences:
        text = ev.get("document", "") or ev.get("text", "")
        meta = ev.get("metadata", {})
        # 也加入元数据中的标题/section
        section = meta.get("section", "")
        if section:
            text = section + " " + text
        evidence_texts.append(text)

    combined_evidence = " ".join(evidence_texts).lower()

    # 从回答中提取关键声明行
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    claim_lines = []
    for line in lines:
        # 跳过纯格式行、引用标记、空泛描述
        if any(skip in line for skip in [
            "##", "###", "【", "参考资料", "> ", "---",
            "请按照", "如证据不足", "当前知识库",
            "建议排查方向", "以下为推测", "非产品官方结论",
            "非官方", "推测/排查建议",
        ]):
            continue
        if len(line) > 10:
            claim_lines.append(line)

    # 对每条声明做关键词匹配
    grounded_count = 0
    unverified = []
    claims_checked = 0

    for line in claim_lines:
        claims_checked += 1
        # 提取关键短语 (中文按短句, 英文按空格)
        phrases = extract_key_phrases(line)
        if not phrases:
            continue

        # 检查是否有足够短语在证据中
        matches = sum(1 for p in phrases if p.lower() in combined_evidence)
        ratio = matches / len(phrases) if phrases else 0

        if ratio >= min_overlap_ratio:
            grounded_count += 1
        else:
            unverified.append(line[:120])

    grounded = len(unverified) <= max(1, claims_checked * 0.3)  # 最多30%未验证

    return {
        "grounded": grounded,
        "claims_checked": claims_checked,
        "claims_grounded": grounded_count,
        "unverified": unverified[:5],  # 最多返回5条
    }


def extract_key_phrases(text: str) -> list:
    """提取关键短语用于匹配"""
    import re
    phrases = set()

    # 去掉标点
    cleaned = re.sub(r"[，。；：！？、\*\#\(\)\[\]\{\}]", " ", text)

    # 中文: 取2-4字的片段
    chinese = re.findall(r"[一-鿿]+", cleaned)
    for word in chinese:
        if len(word) >= 2:
            # 取2-gram和3-gram
            for step in [2, 3]:
                for i in range(0, len(word) - step + 1, step):
                    phrases.add(word[i:i + step])

    # 英文/数字词
    en_words = re.findall(r"[a-z0-9]+", cleaned.lower())
    for w in en_words:
        if len(w) >= 2:
            phrases.add(w)

    return list(phrases)

