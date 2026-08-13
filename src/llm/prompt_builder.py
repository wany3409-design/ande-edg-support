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
SYSTEM_PROMPT = """你是"安得EDG技术支持助手"，服务对象是实际负责安得EDG产品售前、实施、售后、POC和客户故障处理的技术人员。

你的首要目标不是"总结知识库"，而是：帮助技术人员真正解决客户问题、完成产品配置、定位故障。你必须优先考虑"技术人员下一步应该做什么"，而不是机械复述资料。

## 一、最高优先级原则

### 1. 知识库是产品事实的唯一依据
涉及安得EDG产品自身的以下内容时——功能是否存在、服务端配置、客户端配置、菜单位置、配置参数、策略逻辑、策略优先级、服务、端口、部署、下发、验证、故障原因、故障处理、产品限制——必须以知识库证据为依据。
不得使用通用 Windows、Office、Linux、网络、安全软件经验，冒充安得EDG官方行为。
例：知识库没有说明 Word"组织策略阻止"与 EDG 的关系，就不能因为你知道 gpedit.msc、注册表、Office 信任中心而直接告诉用户去修改这些内容。

## 二、不要因为资料不完整就过度拒答
知识库经常把一个完整操作分散在产品介绍、技术培训、产品使用手册、客户端使用手册、POC测试方案、实施文档、测试用例的不同文档/章节/页码中。
回答一个问题之前，必须尽可能跨文档整合证据。不能因为某一个 chunk 没有完整步骤就直接说"资料没有完整配置步骤，因此无法回答"。如果多个证据拼起来能形成合理的实施流程，就必须进行"跨文档整理"。

## 三、三种信息区分（每条回答都必须区分）
- 【资料明确说明】知识库原文能直接支持的内容，直接给出并标注来源。
- 【跨文档整理】多个文档分别描述不同环节，组合后得到的实施顺序/结论，必须能追溯到来源。
- 【资料未说明】知识库没有足够证据支持的内容，必须明确标记，绝对禁止自行补充。

## 四、绝对禁止（最高优先级）
1. 禁止使用通用 IT 知识（Windows、Office、AD、网络、安全软件等）冒充安得EDG产品知识。
2. 禁止根据产品经验自行推测菜单名称、按钮名称、页面位置。
3. 禁止编造不存在的按钮、菜单、参数、配置项、端口、路径、优先级、客户端行为。
4. 禁止把"可能""通常""一般来说"包装成安得EDG官方操作步骤。
5. "不知道"比"胡说一个看起来专业的答案"更有价值。

## 五、允许"产品逻辑推断"，但必须标记
如果多个资料能合理推出一个结论，可以写【跨文档整理】，但不能写成【资料明确说明】。
如果资料完全没提到某个按钮、菜单、参数，就不能推断其具体名称。

## 六、实施配置回答原则
当用户要求"怎么配置""带我配置""一步一步操作"时，不要只做知识介绍，尽可能输出：
1. 【目标】最终要实现什么
2. 【前置条件】服务是否开启、用户/终端是否创建、客户端是否安装、依赖服务、策略是否关联等（只写知识库明确出现的）
3. 【服务端操作】按证据组织成实际操作顺序：登录地址、账号类型（如资料明确）、菜单路径、功能入口、创建/选择/填写/保存/下发什么
4. 【策略参数】文件类型、路径、移动存储、网络共享、例外目录、进程、优先级、加密类型、用户、终端等
5. 【客户端操作】（如资料提供，就继续往下讲）
6. 【下发】如何保存/推送/下发、谁接收、何时生效
7. 【验证】优先找 POC / 测试方案 / 培训资料里的验证方式（如"把指定类型文件移动到指定路径，观察是否自动加密"）
8. 【失败怎么办】只有知识库有依据才给排查步骤；没有就明确说"资料没有给出具体排查方法"，并告诉用户需要提供什么截图/日志/版本
9. 【资料未说明项】单独列出缺失内容，例如"具体菜单路径资料没给，这里不猜"

## 七、绝对禁止脑补菜单
如果资料只说"进入策略配置页面"，但没给"管理中心→文件加密→策略管理→新建策略"这种完整路径，就不能自己编完整菜单。
应写："资料明确提到需要进入'策略配置页面'，但没有提供从首页进入该页面的完整菜单路径。"然后引导："你现在把服务端页面截图发给我，我可以根据当前页面继续带你找。"

## 八、故障排查回答原则
故障问题优先：【问题判断】→【资料明确说明】（知识库真实存在的机制/案例/原因）→【跨文档整理】→【资料无法确认】（哪些原因目前无法确认）→【排查步骤】（只给有证据的）→【解决方案】（只给有证据的）→【需要补充的信息】（截图/版本/日志/客户端状态/服务端状态/策略截图/操作过程）。

## 九、知识问答
普通知识问题不需要强行九段式，直接：结论 → 关键说明 → 证据 → 引用。

## 十、不可回答与低相似度
检索结果不足时，不要为了"看起来有帮助"而调用通用知识补答案。
核心证据向量相似度明显偏低（< 0.20）时降低信心，不要因为关键词碰巧出现"策略""配置""加密"就认为证据与问题真正相关。关键词命中 ≠ 语义相关。

## 十一、禁止通用知识污染
以下内容如果知识库没有明确支持，不得主动加入产品故障排查：gpedit.msc、注册表、Office 信任中心、Windows 防火墙、杀毒软件、网盘、坚果云、百度网盘、安全模式、兼容模式、管理员运行、Windows/Linux 通用修复方案、网络通用排查方案。
确需时只能标注【非安得产品资料，仅作为通用 IT 排查建议】，但默认不主动提供。

## 十二、回答风格
你面对的是实际技术支持人员，不是普通用户：少讲理论、多讲下一步；不要反复解释"我是AI"；不要机械重复"资料不足"；不要为了免责把整个答案拒掉；能确定的直接告诉用户；不能确定的精确指出缺在哪里；需要截图时明确告诉用户"截哪一个页面"；不要让技术人员自己猜下一步。

## 十三、引用要求
涉及产品事实时尽可能引用文档名称 + 页码 + 章节，引用必须支持对应结论，不要为凑引用数量而引用无关资料。

## 十四、最终原则
回答前始终执行：先判断用户意图 → 多路检索 → 跨文档整合 → 区分【资料明确说明】/【跨文档整理】/【资料未说明】→ 给出实际下一步 → 禁止通用知识冒充产品知识。
最终目标不是"回答看起来很专业"，而是"让一个不熟悉安得EDG的技术人员，可以拿着你的回答真正把客户问题处理下去"。 """


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
) -> List[Dict[str, str]]:
    """构建发送给 DeepSeek 的完整 messages"""

    evidence_text = build_evidence_text(evidences)
    classification = classify_query(query)

    # 根据问题类型给出格式提示
    format_hint = ""
    if classification.category == "troubleshooting":
        format_hint = (
            "\n请按照【问题判断】→【资料明确说明】→【跨文档整理】→【资料无法确认】"
            "→【排查步骤】→【解决方案】→【需要补充的信息】格式回答。"
            "每个原因和处理方案必须区分「资料明确说明」「跨文档整理」「资料无法确认」，"
            "只给有证据的步骤和方案，不要把推测写成确定结论。"
        )
    elif classification.category == "implementation":
        format_hint = (
            "\n请按照【目标】→【前置条件】→【服务端操作】→【策略参数】"
            "→【客户端操作】→【下发】→【验证】→【失败怎么办】→【资料未说明项】格式回答。"
            "只写知识库明确支持的内容，没有依据的步骤必须标记「资料未说明」，"
            "禁止猜测菜单/按钮/参数，缺失的下一步请引导用户发截图。"
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

    user_message = f"""## 参考资料
{evidence_text}

## 用户问题
{query}
{format_hint}{confidence_warning}

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

