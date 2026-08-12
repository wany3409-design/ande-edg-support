"""
轻量可替换 Reranker

第一版：关键词匹配 + topic 匹配 + source_level 加权 + 原始向量相似度
设计为可替换接口，后续可接入 Cross-Encoder 等专业 Reranker。
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod
import re


class BaseReranker(ABC):
    """Reranker 抽象基类"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """重排序并返回 top_k"""
        ...


class LightweightReranker(BaseReranker):
    """
    轻量级 Reranker

    综合评分 = 关键词分 + topic分 + source_level分 + 向量相似度分
    后续可替换为 Cross-Encoder。
    """

    # source_level 权重
    SOURCE_WEIGHTS = {
        "official": 1.0,
        "training": 0.95,
        "inferred": 0.7,
    }

    # 关键词权重配置 (V3: 降低向量权重，大chunk向量相似度偏低)
    KEYWORD_WEIGHT = 0.35
    TOPIC_WEIGHT = 0.25
    SOURCE_WEIGHT = 0.10
    VECTOR_WEIGHT = 0.30

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """综合评分重排序"""
        scored = []
        for doc in documents:
            score = self._score_one(query, doc)
            scored.append({**doc, "rerank_score": round(score, 4)})

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]

    def _score_one(self, query: str, doc: Dict[str, Any]) -> float:
        """计算单条文档的综合评分"""
        meta = doc.get("metadata", {})
        text = doc.get("document", "") or doc.get("text", "")
        vec_sim = doc.get("similarity", 0) or (1 - doc.get("distance", 1))

        # 1. 关键词匹配分
        kw_score = self._keyword_score(query, text, meta)

        # 2. topic 匹配分
        topic_score = self._topic_score(query, meta.get("topic", ""))

        # 3. source_level 加权
        source_score = self.SOURCE_WEIGHTS.get(
            meta.get("source_level", "official"), 1.0
        )

        # 4. 综合
        total = (
            self.KEYWORD_WEIGHT * kw_score +
            self.TOPIC_WEIGHT * topic_score +
            self.SOURCE_WEIGHT * source_score +
            self.VECTOR_WEIGHT * vec_sim
        )
        return total

    def _keyword_score(self, query: str, text: str, meta: dict) -> float:
        """关键词匹配评分 (Jaccard-like)"""
        # 提取查询中的关键词
        query_words = set(self._tokenize(query))

        # 在文档文本和元数据中搜索
        search_text = (
            text + " " +
            meta.get("section", "") + " " +
            meta.get("topic", "") + " " +
            meta.get("source_file", "")
        ).lower()

        search_words = set(self._tokenize(search_text))

        if not query_words:
            return 0.0

        intersection = query_words & search_words
        return len(intersection) / len(query_words)

    def _topic_score(self, query: str, topic: str) -> float:
        """topic 匹配评分 (含同义词映射)"""
        if not topic:
            return 0.0

        topics = [t.strip() for t in topic.split(";")]
        query_lower = query.lower()

        # POC/测试/验证/部署 同义词映射
        synonym_map = {
            "poc": ["测试验证", "安装部署", "上线部署"],
            "测试": ["测试验证", "故障排查"],
            "验证": ["测试验证"],
            "部署": ["安装部署"],
            "安装": ["安装部署"],
        }

        matches = 0
        for t in topics:
            t_lower = t.lower()
            # 直接匹配
            if t_lower in query_lower:
                matches += 1
                continue
            # 同义词匹配
            for syn, targets in synonym_map.items():
                if syn in query_lower and t in targets:
                    matches += 0.5
                    break

        return min(1.0, matches / max(1, len(topics)))

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（中文按2-gram + 英文按空格）"""
        text = text.lower().strip()
        # 提取中英文词
        tokens = set()

        # 英文词
        en_words = re.findall(r"[a-z0-9]+", text)
        tokens.update(w for w in en_words if len(w) >= 2)

        # 中文 2-gram
        chinese = re.sub(r"[^\\u4e00-\\u9fff]", "", text)
        for i in range(len(chinese) - 1):
            tokens.add(chinese[i:i + 2])

        return list(tokens)


# 默认实例
_reranker: LightweightReranker = None


def get_reranker() -> LightweightReranker:
    global _reranker
    if _reranker is None:
        _reranker = LightweightReranker()
    return _reranker
