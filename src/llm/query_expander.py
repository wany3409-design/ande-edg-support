"""
查询扩展器 (Query Expander)

问题背景：
    对于 implementation / troubleshooting 类问题，单次 embedding 检索容易被
    用户自然语言中的填充词稀释（例如"我要配置...怎么配置...请带我一步一步做"），
    导致核心知识点无法召回。实测："落地加密"单独检索能命中技术培训 p19 (sim=0.87)，
    但完整问句检索却完全召回不到。

设计原则：
    1. 查询扩展只用于"找资料"，绝不用于生成产品答案。
    2. 不使用 LLM 生成子查询（避免引入新模型调用、避免幻觉）。
    3. 基于规则 + jieba 关键词提取（jieba 已是既有依赖）。

输出：
    一组检索子查询，用于多路向量召回后合并去重。
"""

import re
from typing import List

import jieba
import jieba.analyse

# ===== 领域术语扩展表 =====
# 键为知识库中出现的核心技术词，值为同义/近义/上下位扩展词。
# 用于解决"落地加密" vs "落地加解密" vs "落地解密"这类表述不一致问题。
_DOMAIN_TERM_EXPANSIONS = {
    "落地加密": ["落地加解密", "落地解密", "落地加密策略"],
    "落地加解密": ["落地加密", "落地解密"],
    "落地解密": ["落地加密", "落地加解密"],
    "透明加密": ["透明加解密", "自动加密"],
    "透明加解密": ["透明加密", "透明解密", "自动加密"],
    "自动加密": ["透明加密", "落地加密"],
    "文件加密": ["文件加解密", "文件加密服务", "文件解密"],
    "文件解密": ["文件加解密", "文件加密", "解密流程"],
    "文件加解密": ["文件加密", "文件解密"],
    "策略配置": ["策略下发", "策略关联", "策略组"],
    "策略下发": ["策略配置", "策略关联", "下发策略"],
    "加解密类型": ["加密类型", "加解密策略"],
    "外发": ["文件外发", "外发审批", "外发流程"],
    "水印": ["屏幕水印", "文档水印", "打印水印"],
    "安全网关": ["加解密网关", "网关策略"],
    "AD域": ["AD域同步", "域同步", "组织架构"],
}

# ===== implementation 类问题的通用检索维度后缀 =====
# 一个实施/配置问题通常涉及多个维度，单一查询难以同时覆盖。
# 用核心词 + 这些维度分别检索，扩大召回面。
_IMPL_DIMENSION_SUFFIXES = [
    "服务端 策略配置",
    "文件类型 例外目录 进程列表 优先级",
    "策略下发 验证",
    "客户端 验证",
]

# ===== troubleshooting 类问题的通用检索维度后缀 =====
_TROUBLE_DIMENSION_SUFFIXES = [
    "排查 步骤",
    "可能原因 解决方案",
    "日志 报错",
]

# ===== 填充/干扰词（会稀释 embedding 的口语化表达）=====
_STOPWORDS = {
    "我要", "我想", "需要", "帮我", "帮我一下", "请", "请带我", "带我",
    "怎么", "如何", "怎么样", "为什么", "什么", "原因", "是什么",
    "配置", "设置", "部署", "安装", "开启", "关闭", "操作", "弄", "做",
    "一步一步", "一步", "步骤", "一下", "这个", "那个", "该", "的",
    "了", "吗", "呢", "啊", "吧", "和", "与", "及", "或者", "还是",
    "能不能", "可以", "是否", "请问", "麻烦", "指导下", "教我",
}

# ===== 候选领域词（用于在 query 中直接探测多字技术词）=====
# 这些是知识库中高频出现、jieba 可能切不完整的产品术语。
_KNOWN_DOMAIN_PHRASES = [
    "落地加解密", "落地加密", "落地解密",
    "透明加解密", "透明加密", "透明解密",
    "自动加密", "自动解密", "手动加密",
    "文件加密服务", "文件加密", "文件解密", "文件加解密",
    "加解密类型", "加解密策略", "加解密网关",
    "策略配置", "策略下发", "策略关联", "策略组",
    "辅助控制", "例外目录", "进程列表", "进程签名", "文件类型",
    "移动存储", "共享目录", "本地路径", "网络共享",
    "文件保护", "文件防勒索", "文件外发", "外发云盒",
    "安全网关", "中间件管控", "软件配置库", "基础配置库",
    "屏幕水印", "文档水印", "打印水印", "水印",
    "AD域同步", "域同步", "组织架构", "组织结构", "用户管理",
    "全盘加密", "全盘解密", "硬盘加密", "离线工作",
]


def _extract_core_terms(query: str) -> List[str]:
    """
    提取查询中的核心技术词。

    优先使用领域词典直接探测（jieba 可能切不准技术词），
    其次用 jieba 提取名词性关键词兜底。
    """
    terms = []

    # 1. 领域词直接探测（按长度降序，先匹配长词避免子串误伤）
    for phrase in sorted(_KNOWN_DOMAIN_PHRASES, key=len, reverse=True):
        if phrase in query:
            terms.append(phrase)

    # 2. jieba 名词性关键词兜底
    try:
        tags = jieba.analyse.extract_tags(
            query, topK=6, allowPOS=("n", "vn", "nz", "eng", "ns")
        )
        for t in tags:
            if t not in _STOPWORDS and len(t) >= 2:
                terms.append(t)
    except Exception:
        pass

    # 去重保序
    seen = set()
    deduped = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def expand_queries(query: str, category: str) -> List[str]:
    """
    生成检索子查询列表。

    Args:
        query: 用户原始问题
        category: 问题分类 (knowledge / implementation / troubleshooting)

    Returns:
        去重后的子查询列表（含原始 query 作为兜底）。
        knowledge 类只返回原始 query（不需要扩展）。
    """
    if category == "knowledge":
        return [query]

    core_terms = _extract_core_terms(query)

    # 区分"领域短语"（价值高）和"jieba 单字/普通词"（价值低）
    domain_terms = [t for t in core_terms if t in _KNOWN_DOMAIN_PHRASES]
    jieba_terms = [t for t in core_terms if t not in _KNOWN_DOMAIN_PHRASES]

    sub_queries: List[str] = []
    sub_queries.append(query)  # 原始 query 永远保留，作为兜底

    # 1. 领域短语单独成查询（最直接召回核心知识点）
    for term in domain_terms:
        if term not in sub_queries:
            sub_queries.append(term)

    # 2. 领域词的同义/近义扩展
    for term in domain_terms:
        for expansion in _DOMAIN_TERM_EXPANSIONS.get(term, []):
            if expansion not in sub_queries:
                sub_queries.append(expansion)

    # 3. 领域短语 + 维度后缀（覆盖服务端/配置/下发/验证等不同角度）
    suffixes = (
        _IMPL_DIMENSION_SUFFIXES if category == "implementation"
        else _TROUBLE_DIMENSION_SUFFIXES
    )
    for term in domain_terms[:2]:
        for suffix in suffixes:
            combined = f"{term} {suffix}"
            if combined not in sub_queries:
                sub_queries.append(combined)

    # 4. jieba 单字/普通词兜底（价值较低，放在最后）
    for term in jieba_terms:
        if term not in sub_queries:
            sub_queries.append(term)

    # 限制子查询数量，避免检索过慢
    return sub_queries[:12]
