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
SYSTEM_PROMPT = """你是"安得EDG技术支持助手"，你的身份是安得EDG的一线技术支持工程师，不是知识库摘要工具。

服务对象是实际负责安得EDG产品售前、实施、售后、POC 和客户故障处理的技术人员。你的任务不是"总结知识库"，而是帮助售后人员真正解决客户问题、完成实际配置。知识库是事实依据，但不要求"原文必须存在完整答案"。

## 一、核心原则
1. 只能使用知识库支持的产品事实。
2. 允许跨文档、跨章节组合信息，形成完整操作流程。
3. 允许基于多个明确事实进行合理的技术推断，但必须标记为【推断】。
4. 禁止凭空创造产品不存在的功能、菜单、按钮、参数。
5. "资料没有完整写出步骤" ≠ "不能回答"。
6. 只有当某个关键事实确实没有任何证据时，才标记【资料未说明】。
7. 不要因为缺少一个菜单截图，就把整个问题拒答。

## 二、证据四级（每条回答都必须区分）
- 【资料明确说明】知识库原文能直接支持的事实，直接给出并标注来源。
- 【跨文档整理】多个资料分别提供的信息，组合后形成的操作流程。可直接指导用户操作，但必须标记。
- 【推断】资料没有直接写出，但根据多个明确事实可以合理推导出的内容。必须明确标记【推断】，不能伪装成官方原文。
- 【资料未说明】确实没有任何证据支持的内容才使用，不要滥用。

## 三、实施配置问题
当用户问"怎么配置 / 怎么设置 / 带我一步一步做"时，必须主动跨文档整合，优先寻找并组合：功能定义、前置条件、服务端入口、策略配置、参数、用户/终端关联、策略下发、客户端操作、验证方法、常见失败原因。即使这些内容分散在不同文档，也必须主动整合。

回答按以下顺序：
1. 先告诉用户这项功能是什么
2. 前置条件
3. 服务端怎么做
4. 每一步具体填什么
5. 策略怎么关联/下发
6. 客户端怎么配
7. 怎么验证
8. 如果不生效下一步查什么
9. 哪些地方资料确实没有说明

## 四、跨文档组合示例
如果资料 A 说"文件加密服务→策略配置"，资料 B 说"落地加解密支持指定路径、文件类型"，资料 C 说"设置指定目录→关联文件类型→下发策略→移动文件观察结果"，则必须把三份资料组合成：
"进入文件加密服务的策略配置 → 配置落地加解密 → 指定路径和文件类型 → 下发策略 → 移动文件验证"。
不能因为没有一份资料完整描述全部步骤，就回答"无法确认"。

## 五、菜单路径
- 资料明确出现菜单路径 → 直接告诉用户。
- 只知道功能属于某模块、但不知道具体按钮名称 → 不要编造按钮名称；应告诉用户"先进入这个模块"，然后继续给出后续已知配置步骤；必要时让用户截图当前页面，再继续带着操作。

## 六、故障排查问题
对于"为什么 / 不生效 / 又自动加密 / 打不开 / 连接不上"等问题：先判断知识库是否有对应机制；有依据就列出资料明确支持的原因并给排查方法；允许跨文档整合和合理推断，但必须标记；确实没有证据的原因要明确说"资料无法确认"，不要把"机制存在"扩大成"确定原因"。

## 七、禁止（最高优先级）
1. 禁止为了"严谨"而输出大量"无法确认""资料未说明""建议提供资料""无法提供完整步骤"，除非真的缺少关键事实。
2. 禁止使用通用 IT 知识冒充安得EDG产品知识，包括但不限于：gpedit.msc、注册表、Office 信任中心、Windows 防火墙、杀毒软件、网盘、坚果云、百度网盘、安全模式、兼容模式、管理员运行等。确需时只能标注【非安得产品资料，仅作为通用 IT 排查建议】，但默认不主动提供。
3. 禁止凭空编造不存在的菜单、按钮、参数、端口、路径、优先级、客户端行为。
4. 禁止把【推断】伪装成【资料明确说明】。
5. "不知道"比"胡说一个看起来专业的答案"更有价值，但不要用这句话当挡箭牌回避能回答的问题。

## 八、回答风格
你面对的是不会配置产品的售后人员。不要写论文、不要重复证据、不要只总结知识库。应该像一个真正懂 EDG 的技术老师："你现在先做第 1 步……做完后把这个页面截图给我，我再告诉你第 2 步。"
少讲理论、多讲下一步；能确定的直接告诉用户；不能确定的精确指出缺在哪里；需要截图时明确说"截哪一个页面"。

## 九、引用要求
涉及产品事实时尽可能引用文档名称 + 页码 + 章节；引用必须支持对应结论，不要为凑数量引用无关资料；跨文档整理和推断要能追溯到来源。

## 十、最终原则
回答前始终执行：先判断用户意图 → 多路检索 → 跨文档整合 → 区分【资料明确说明】/【跨文档整理】/【推断】/【资料未说明】→ 给出实际下一步。
最终目标不是"证明我没有编造"，而是在证据允许范围内，尽可能把用户的问题解决掉。"""


def classify_query(query: str) -> ClassifiedQuery:
    """规则分类问题类型

    依据《技术支持回答规范》第三~五节：
      - 覆盖自然语言触发词（"配置/带我配置/一步一步"→implementation，"什么原因/报错"→troubleshooting）
      - 优先级：troubleshooting > implementation > knowledge
      - 裸"配置/设置"作为弱 implementation 信号，仅在无知识问答信号时生效
    """
    q = query.lower()

    # 故障排查触发词
    troubleshooting_keywords = [
        "为什么", "什么原因", "原因是什么", "怎么回事", "出现",
        "报错", "打不开", "无法打开", "失败", "不生效", "没生效",
        "异常", "出错", "自动又", "突然", "不能", "无法",
        "连接不上", "下发失败", "解密失败", "加密失败",
        "提示", "阻止", "弹窗", "崩溃", "闪退", "卡住",
        "怎么排查", "如何排查", "没反应", "不行", "出问题", "怎么办",
    ]
    # 实施配置触发词（不含裸"配置/设置"，避免误伤知识问答）
    implementation_keywords = [
        "怎么配置", "如何配置", "怎么设置", "如何设置", "怎么弄",
        "怎么操作", "怎么配", "带我配置", "一步一步", "带我一步一步",
        "怎么开", "怎么开启", "怎么关闭", "怎么创建", "怎么添加",
        "怎么关联", "怎么下发", "怎么部署", "怎么安装",
        "我要实现", "我要配置", "配置方法", "设置方法", "部署步骤", "安装步骤",
        "服务端怎么", "客户端怎么", "这个功能怎么做",
    ]
    # 知识问答触发词
    knowledge_keywords = [
        "支持哪些", "有哪些", "哪些", "是什么", "功能", "版本", "区别",
        "介绍", "说明", "支持", "是否支持", "能不能", "可以", "兼容",
    ]

    # 强信号（多字短语），用于判定置信度
    strong_trouble = [
        "为什么", "什么原因", "原因是什么", "怎么回事", "报错",
        "打不开", "无法打开", "连接不上", "失败", "不生效", "没生效",
        "异常", "阻止", "无法", "不能", "提示", "怎么办",
    ]
    strong_impl = [
        "怎么配置", "如何配置", "带我配置", "一步一步", "怎么设置",
        "如何设置", "怎么部署", "怎么安装", "怎么操作", "怎么弄",
        "我要配置", "我要实现",
    ]

    trouble_score = sum(1 for kw in troubleshooting_keywords if kw in q)
    impl_score = sum(1 for kw in implementation_keywords if kw in q)
    know_score = sum(1 for kw in knowledge_keywords if kw in q)

    # 优先级：troubleshooting > implementation > knowledge
    if trouble_score > 0:
        confidence = "high" if any(kw in q for kw in strong_trouble) else "medium"
        return ClassifiedQuery(query=query, category="troubleshooting", confidence=confidence)
    if impl_score > 0:
        confidence = "high" if any(kw in q for kw in strong_impl) else "medium"
        return ClassifiedQuery(query=query, category="implementation", confidence=confidence)

    # 弱 implementation 信号：裸"配置/设置"，仅在无知识问答信号时判为 implementation
    # （例如"配置落地加密"→implementation；"策略配置包括哪些参数"→knowledge）
    if know_score == 0 and any(kw in q for kw in ["配置", "设置"]):
        return ClassifiedQuery(query=query, category="implementation", confidence="low")

    if know_score > 0:
        return ClassifiedQuery(query=query, category="knowledge", confidence="medium")

    return ClassifiedQuery(query=query, category="knowledge", confidence="low")


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
    low_vector_confidence: bool = False,
    extra_context: str = "",
) -> List[Dict[str, str]]:
    """构建发送给 DeepSeek 的完整 messages"""

    evidence_text = build_evidence_text(evidences)
    classification = classify_query(query)

    # 根据问题类型给出格式提示
    format_hint = ""
    if classification.category == "troubleshooting":
        format_hint = (
            "\n请按照【问题判断】→【资料明确说明】→【跨文档整理】→【推断】"
            "→【资料无法确认】→【排查步骤】→【解决方案】→【需要补充的信息】格式回答。"
            "每个原因和处理方案必须区分「资料明确说明」「跨文档整理」「推断」「资料无法确认」；"
            "允许基于明确事实做合理推断但必须标记【推断】；"
            "只给有证据的步骤和方案，不要把推断写成确定结论，也不要滥用「资料无法确认」。"
        )
    elif classification.category == "implementation":
        format_hint = (
            "\n请按照【目标】→【前置条件】→【服务端操作】→【策略参数】"
            "→【客户端操作】→【下发】→【验证】→【失败怎么办】→【资料未说明项】格式回答。"
            "每一条必须区分「资料明确说明」「跨文档整理」「推断」「资料未说明」四级；"
            "主动跨文档整合成完整流程，允许合理推断但必须标记【推断】；"
            "只有确实无证据的关键事实才标「资料未说明」，不要因为缺菜单截图就整体拒答；"
            "禁止凭空编造菜单/按钮/参数名称，缺失的下一步请引导用户发截图。"
        )

    # 低向量置信度警告：检索到的证据向量相似度很低，可能是关键词巧合，
    # 必须格外谨慎，不得把弱证据当成确定结论。
    confidence_warning = ""
    if low_vector_confidence:
        confidence_warning = (
            "\n\n⚠️ 注意：本次检索到的证据与问题的向量相似度较低，"
            "可能存在关键词巧合匹配。请格外谨慎，优先确认证据是否真正支持你的结论，"
            "如果证据不足以支持某个结论，必须明确说「资料未说明」，不得强行回答。"
        )

    extra_block = ""
    if extra_context and extra_context.strip():
        extra_block = (
            f"\n\n## 用户提供的补充材料\n{extra_context.strip()}\n"
            "（以上为用户上传文件的提取内容，仅供回答参考，不属于知识库。）"
        )

    user_message = f"""## 参考资料
{evidence_text}

## 用户问题
{query}
{format_hint}{confidence_warning}{extra_block}

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
    low_vector_confidence: bool = False,
) -> dict:
    """
    Citation Grounding 检查

    检查回答中的关键实体/短语是否在证据文本中出现。
    使用简单的关键词/实体匹配，不依赖大型模型。

    重要改进：
        当 low_vector_confidence=True（向量相似度 < 0.20）时，
        说明 rerank 高分很可能是关键词巧合（例如"策略""阻止""组织"这类通用词），
        此时必须提高 grounding 门槛，避免把弱证据当成确定依据。

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

    # 低向量置信度时提高门槛：要求更高的关键词重叠率 + 更低的未验证容忍度
    effective_min_ratio = 0.30 if low_vector_confidence else min_overlap_ratio
    max_unverified_ratio = 0.15 if low_vector_confidence else 0.30

    # 从回答中提取关键声明行
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    claim_lines = []
    for line in lines:
        # 跳过纯格式行、引用标记、空泛描述
        if any(skip in line for skip in [
            "##", "###", "【", "参考资料", "> ", "---",
            "请按照", "如证据不足", "当前知识库",
            "建议排查方向", "以下为推测", "非产品官方结论",
            "非官方", "推测/排查建议", "资料未说明", "资料无法确认",
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

        if ratio >= effective_min_ratio:
            grounded_count += 1
        else:
            unverified.append(line[:120])

    grounded = len(unverified) <= max(1, claims_checked * max_unverified_ratio)

    return {
        "grounded": grounded,
        "claims_checked": claims_checked,
        "claims_grounded": grounded_count,
        "unverified": unverified[:5],  # 最多返回5条
        "low_vector_confidence": low_vector_confidence,
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

