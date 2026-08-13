"""
RAG 回答全链路管线

用户问题 -> 问题分类 -> Query处理 -> Chroma召回(Top10) -> Rerank(Top3-5)
-> 证据筛选 -> DeepSeek API -> 技术支持回答 + 来源引用

设计为可独立运行、可单步调试的管线。
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    RETRIEVAL_TOP_K_CANDIDATE,
    RETRIEVAL_TOP_K,
)
from src.llm.reranker import get_reranker, BaseReranker
from src.llm.prompt_builder import (
    classify_query,
    ClassifiedQuery,
    build_messages,
    build_citation_text,
    ground_citations,
)
from src.llm.deepseek_client import get_client, DeepSeekClient
from src.llm.query_expander import expand_queries
from src.llm.chunk_filter import merge_query_results

logger = logging.getLogger(__name__)


# ===== Phase 4.5 使用 V3 collection (400-char chunks, improved HTML parser) =====
PHASE4_COLLECTION = "ande_edg_v3"
PHASE4_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 不可回答判断阈值：综合 rerank 评分低于此值时，标记为知识库外
# 250-char chunks 平衡版本
UNANSWERABLE_RERANK_THRESHOLD = 0.235


@dataclass
class RetrievalResult:
    """单条检索结果"""
    chunk_id: str
    text: str
    metadata: dict
    similarity: float
    distance: float
    rerank_score: float = 0.0


@dataclass
class AnswerResult:
    """RAG 回答完整结果"""
    query: str
    category: str
    category_confidence: str
    is_answerable: bool
    answer: str
    citations: str
    evidences: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_count: int = 0
    rerank_count: int = 0
    timing: Dict[str, float] = field(default_factory=dict)
    log_summary: str = ""
    grounding: Dict[str, Any] = field(default_factory=dict)


class AnswerPipeline:
    """RAG 回答全链路管线"""

    def __init__(
        self,
        collection_name: str = PHASE4_COLLECTION,
        embedding_model_name: str = PHASE4_EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        top_k_candidate: int = RETRIEVAL_TOP_K_CANDIDATE,
        top_k_final: int = RETRIEVAL_TOP_K,
    ):
        self.collection_name = collection_name
        self.top_k_candidate = top_k_candidate
        self.top_k_final = top_k_final

        # 延迟加载，避免导入时就加载模型
        self._embedding_model: Optional[SentenceTransformer] = None
        self._embedding_model_name = embedding_model_name
        self._device = device
        self._chroma_client: Any = None
        self._collection: Any = None
        self._reranker: Optional[BaseReranker] = None
        self._llm_client: Optional[DeepSeekClient] = None

    # ===== 延迟加载 =====

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            t0 = time.time()
            logger.info(f"加载 embedding 模型: {self._embedding_model_name}")
            self._embedding_model = SentenceTransformer(
                self._embedding_model_name, device=self._device
            )
            logger.info(
                f"模型加载完成 ({time.time() - t0:.1f}s), "
                f"dim={self._embedding_model.get_embedding_dimension()}"
            )
        return self._embedding_model

    @property
    def collection(self):
        if self._collection is None:
            self._chroma_client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._chroma_client.get_collection(
                self.collection_name
            )
            logger.info(
                f"ChromaDB collection '{self.collection_name}' "
                f"已加载, count={self._collection.count()}"
            )
        return self._collection

    @property
    def reranker(self) -> BaseReranker:
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    @property
    def llm_client(self) -> DeepSeekClient:
        if self._llm_client is None:
            self._llm_client = get_client()
        return self._llm_client

    # ===== 主流程 =====

    def answer(self, query: str) -> AnswerResult:
        """完整 RAG 问答流程"""
        timing = {}
        t_total = time.time()

        # Step 1: 问题分类
        t0 = time.time()
        classification = classify_query(query)
        timing["classification"] = round(time.time() - t0, 3)

        # Step 2: 多路检索（implementation/troubleshooting 做 query 扩展后合并去重）
        t0 = time.time()
        raw_evidences = self._multi_retrieve(
            query, classification.category, n_results=self.top_k_candidate
        )
        timing["retrieval"] = round(time.time() - t0, 3)

        if not raw_evidences:
            return AnswerResult(
                query=query,
                category=classification.category,
                category_confidence=classification.confidence,
                is_answerable=False,
                answer="当前知识库中未检索到相关内容，无法回答该问题。",
                citations="",
                evidences=[],
                retrieval_count=0,
                rerank_count=0,
                timing=timing,
                log_summary=self._build_log(query, classification, timing, 0, 0),
            )

        # Step 3: Rerank (候选池 -> Top 3-5)
        t0 = time.time()
        reranked = self.reranker.rerank(
            query, raw_evidences, top_k=self.top_k_final
        )
        timing["rerank"] = round(time.time() - t0, 3)

        # Step 4: 判断是否可回答（使用 rerank 综合分 + 向量相似度双重判断）
        top_rerank = reranked[0].get("rerank_score", 0) if reranked else 0
        top1_vec_sim = (
            reranked[0].get("similarity", 0)
            or (1 - reranked[0].get("distance", 1))
            if reranked else 0
        )
        # 综合判断：rerank 分 + 向量相似度双重门槛
        # V3: 250-char chunks，vec_sim 普遍偏低 → min_vec 降至 0.10
        is_answerable = (
            top1_vec_sim >= 0.10
            and top_rerank >= UNANSWERABLE_RERANK_THRESHOLD
        )
        # 低向量置信度标记：向量相似度很低但 rerank 较高时，可能是关键词巧合匹配
        # 此时应要求 LLM 更谨慎，不得把弱证据当成确定结论
        low_vector_confidence = top1_vec_sim < 0.20

        # Step 5: 构建 messages + 调用 DeepSeek
        t0 = time.time()
        if is_answerable:
            messages = build_messages(
                query, reranked, low_vector_confidence=low_vector_confidence
            )
            try:
                answer_text = self.llm_client.chat(messages)
            except Exception as e:
                logger.error(f"DeepSeek API 调用失败: {e}")
                answer_text = f"[LLM 调用失败: {e}]"
            # 兜底：推理模型若推理耗尽 token 预算会返回空 content，
            # 避免把空回答静默展示给用户。
            if not answer_text or not answer_text.strip():
                logger.warning("LLM 返回空内容，可能推理耗尽 max_tokens 预算")
                answer_text = (
                    "本次回答生成不完整（模型推理超时/超预算）。\n\n"
                    "请将您的问题重新发送一次，或补充更多上下文（如所在模块、"
                    "截图、版本信息），我将重新检索并作答。"
                )
            citations = build_citation_text(reranked)
        else:
            answer_text = (
                "当前知识库中没有找到足够依据确认该问题。\n\n"
                "建议排查方向（以下为推测/排查建议，非产品官方结论）：\n"
                "1. 请确认该问题是否属于安得EDG产品的功能范围\n"
                "2. 建议查阅产品最新版官方文档或联系安得技术支持\n"
                "3. 提供更多上下文信息有助于进一步排查"
            )
            citations = ""
        timing["llm"] = round(time.time() - t0, 3)
        timing["total"] = round(time.time() - t_total, 3)

        # Step 6: Citation Grounding 检查
        t0 = time.time()
        grounding = {}
        if is_answerable and answer_text and not answer_text.startswith("[LLM"):
            grounding = ground_citations(
                answer_text, reranked, low_vector_confidence=low_vector_confidence
            )
            # 如果 grounding 不足，在回答末尾添加警告
            if not grounding.get("grounded", True) and grounding.get("claims_checked", 0) > 0:
                unverified_count = len(grounding.get("unverified", []))
                if unverified_count > 0:
                    answer_text += (
                        f"\n\n> **注意**：以上回答中有部分内容在提供的知识库证据中未能充分验证。"
                        f"建议优先确认【参考资料】中的原文描述。"
                    )
        timing["grounding"] = round(time.time() - t0, 3)

        log_summary = self._build_log(
            query, classification, timing,
            len(raw_evidences), len(reranked),
        )

        return AnswerResult(
            query=query,
            category=classification.category,
            category_confidence=classification.confidence,
            is_answerable=is_answerable,
            answer=answer_text,
            citations=citations,
            evidences=reranked,
            retrieval_count=len(raw_evidences),
            rerank_count=len(reranked),
            timing=timing,
            log_summary=log_summary,
            grounding=grounding,
        )

    # ===== 检索 =====

    def _retrieve(
        self, query: str, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """ChromaDB 向量检索"""
        q_emb = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )

        if not res["ids"] or not res["ids"][0]:
            return []

        results = []
        for rank in range(len(res["ids"][0])):
            meta = res["metadatas"][0][rank]
            dist = res["distances"][0][rank]
            sim = round(1 - dist, 4)
            doc_text = res["documents"][0][rank]
            chunk_id = res["ids"][0][rank]

            results.append({
                "chunk_id": chunk_id,
                "document": doc_text,
                "metadata": meta,
                "similarity": sim,
                "distance": dist,
                "score": sim,
            })

        return results

    def _multi_retrieve(
        self, query: str, category: str, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        多路检索：对 implementation / troubleshooting 问题做 query 扩展，
        多角度向量召回后合并去重 + 过滤垃圾 chunk。

        返回一个更大的候选池（去重后按相似度降序）。
        """
        sub_queries = expand_queries(query, category)

        results_by_query = []
        for q in sub_queries:
            try:
                results = self._retrieve(q, n_results=n_results)
                results_by_query.append(results)
            except Exception as e:
                logger.warning(f"子查询检索失败 '{q[:30]}': {e}")
                continue

        merged = merge_query_results(results_by_query)

        # 限制候选池规模，避免送入 reranker 的文档过多影响性能
        return merged[:40]

    # ===== 日志 =====

    def _build_log(
        self,
        query: str,
        classification: ClassifiedQuery,
        timing: Dict[str, float],
        retrieval_count: int,
        rerank_count: int,
    ) -> str:
        parts = [
            f"query={query[:80]}",
            f"category={classification.category}({classification.confidence})",
            f"retrieval={retrieval_count}docs",
            f"rerank={rerank_count}docs",
            f"timing={timing}",
        ]
        return " | ".join(parts)


# ===== 全局单例 =====
_pipeline: Optional[AnswerPipeline] = None


def get_pipeline() -> AnswerPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AnswerPipeline()
    return _pipeline


# ===== 便捷函数 =====
def ask(query: str) -> AnswerResult:
    """便捷问答函数"""
    pipeline = get_pipeline()
    return pipeline.answer(query)
