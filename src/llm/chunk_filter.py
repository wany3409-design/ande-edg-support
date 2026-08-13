"""
检索结果过滤器 (Chunk Filter)

问题背景：
    HTML/PDF 解析会把"章节标题"单独切成一个 chunk（例如"落地加解密"），
    这类 chunk 与查询词匹配度极高但毫无正文，会占据 Top-K 高位、挤掉真正有内容的正文。
    此外同文档存在大量高度重复的 chunk，浪费检索名额。

过滤规则（保守，避免误删有效短 chunk）：
    1. 标题-only chunk：正文与 section 几乎相同，或正文极短且等于标题。
    2. 过短且无信息量的 chunk（< 20 字符，且不含配置参数/数字/特殊符号）。
    3. 高度重复的 chunk（正文归一化后相同或几乎相同）。

设计原则：
    不简单删除所有短 chunk —— 如果短 chunk 本身是完整配置项、警告、注意事项
    （含数字、路径、参数、符号），则允许保留。
"""

import re
from typing import List, Dict, Any


def _normalize(text: str) -> str:
    """归一化文本用于去重比较（去空白/换行/标点）"""
    return re.sub(r"[\s　]+", "", text or "").strip()


def _has_meaningful_content(text: str) -> bool:
    """
    判断短 chunk 是否含有效信息。
    有效信息特征：包含数字、英文字母、路径分隔符、或明显配置符号（@ # | 等）。
    """
    if len(text) >= 20:
        return True
    # 含数字/路径/参数符号 → 视为有效（如"@代表移动存储设备"、端口、路径）
    if re.search(r"[0-9]", text):
        return True
    if re.search(r"[A-Za-z]", text):
        return True
    if re.search(r"[@#|\\/<>{}]", text):
        return True
    return False


def is_garbage_chunk(doc: Dict[str, Any]) -> bool:
    """
    判断单条检索结果是否为垃圾 chunk。

    返回 True 表示应被过滤。
    """
    text = (doc.get("document", "") or doc.get("text", "") or "").strip()
    meta = doc.get("metadata", {}) or {}
    section = (meta.get("section", "") or "").strip()

    # 1. 空文本
    if not text:
        return True

    # 2. 标题-only chunk：正文与 section 相同或几乎相同（正文比标题多不了几个字）
    if section and len(section) >= 2:
        norm_text = _normalize(text)
        norm_section = _normalize(section)
        if norm_text and norm_section:
            # 正文与标题相同，或正文只是标题 + 极少量字符
            if norm_text == norm_section:
                return True
            if norm_text.startswith(norm_section) and (len(norm_text) - len(norm_section)) < 4:
                return True

    # 3. 过短且无信息量（< 20 字符，且无数字/英文/路径/参数符号）
    if len(text) < 20 and not _has_meaningful_content(text):
        return True

    return False


def filter_garbage_chunks(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    过滤垃圾 chunk，并对高度重复的 chunk 去重。

    去重策略：
        按 chunk_id 去重（多路检索会召回相同 chunk），
        再按归一化正文去重（不同 chunk 但正文几乎相同）。
    """
    seen_ids = set()
    seen_texts = set()
    kept = []

    for doc in docs:
        if is_garbage_chunk(doc):
            continue

        # chunk_id 去重
        cid = doc.get("chunk_id") or doc.get("id") or ""
        if cid and cid in seen_ids:
            continue
        if cid:
            seen_ids.add(cid)

        # 归一化正文去重
        norm = _normalize(doc.get("document", "") or doc.get("text", ""))
        if norm:
            if norm in seen_texts:
                continue
            # 短文本不做前缀去重（避免误删），长文本做前缀去重
            if len(norm) >= 30:
                # 检查是否已存在包含该文本的更长 chunk
                is_dup = False
                for seen in seen_texts:
                    if norm in seen or seen in norm:
                        is_dup = True
                        break
                if is_dup:
                    continue
            seen_texts.add(norm)

        kept.append(doc)

    return kept


def merge_query_results(
    results_by_query: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    合并多路检索结果。

    策略：
        1. 拼接所有子查询的召回结果。
        2. 按 chunk_id 去重，保留相似度最高的一条（同一 chunk 可能被多个子查询命中）。
        3. 过滤垃圾 chunk。
        4. 按相似度降序排序。
    """
    best_by_id: Dict[str, Dict[str, Any]] = {}

    for results in results_by_query:
        for doc in results:
            cid = doc.get("chunk_id") or doc.get("id") or ""
            # 用归一化正文作为兜底 id（某些 chunk 可能没有 chunk_id）
            if not cid:
                cid = "text:" + _normalize(doc.get("document", "") or doc.get("text", ""))

            sim = doc.get("similarity", 0)
            if cid not in best_by_id or sim > best_by_id[cid].get("similarity", 0):
                best_by_id[cid] = doc

    merged = list(best_by_id.values())

    # 过滤垃圾 chunk
    merged = filter_garbage_chunks(merged)

    # 按相似度降序
    merged.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return merged
