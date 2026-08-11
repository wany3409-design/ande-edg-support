"""
文本切分器

将解析后的文档页面切分为适合检索的文本块（chunk）。
使用 LangChain 的 RecursiveCharacterTextSplitter 进行语义感知切分。
"""

from typing import List
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_parser.models import RawDocument, ParsedPage, DocumentChunk, generate_chunk_id
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextSplitter:
    """文档文本切分器"""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        # 中文友好的分隔符优先级
        self._splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",     # 段落
                "\n",       # 换行
                "。",       # 句号
                "；",       # 分号
                "，",       # 逗号
                ".",        # 英文句号
                ";",        # 英文分号
                " ",        # 空格
                "",         # 字符级
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(
        self, raw_doc: RawDocument, pages: List[ParsedPage]
    ) -> List[DocumentChunk]:
        """将文档页面切分为 chunk"""
        all_chunks = []

        for page in pages:
            # 构建带来源信息的文本前缀（帮助检索定位）
            page_header = self._build_page_header(raw_doc, page)
            full_text = page_header + page.text

            # 切分
            text_parts = self._splitter.split_text(full_text)

            # 没有切分出来则不添加
            if not text_parts or (len(text_parts) == 1 and not text_parts[0].strip()):
                continue

            total = len(text_parts)
            for i, part in enumerate(text_parts):
                if not part.strip():
                    continue

                chunk_id = generate_chunk_id(raw_doc.file_name, page.page_num, i)
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=part.strip(),
                    source_file=raw_doc.file_name,
                    source_path=raw_doc.file_path,
                    source_level="official",
                    page=page.page_num,
                    section=page.section,
                    doc_type=raw_doc.doc_type,
                    file_type=raw_doc.file_type,
                    topic=self._extract_topic(page.section, part),
                    chunk_index=i,
                    total_chunks=total,
                )
                all_chunks.append(chunk)

        return all_chunks

    def _build_page_header(self, raw_doc: RawDocument, page: ParsedPage) -> str:
        """构建页面头部标注（帮助检索上下文）"""
        parts = [f"【来源】{raw_doc.file_name}"]
        if page.section:
            parts.append(f"【章节】{page.section}")
        parts.append(f"【页码】第{page.page_num}页")
        return " ".join(parts) + "\n"

    def _extract_topic(self, section: str, text: str) -> str:
        """尝试从章节或文本中提取主题关键词"""
        # 简单规则：从章节标题或文本前几个词中提取
        keywords = []
        topic_hints = [
            ("加密", "文件加密"),
            ("解密", "文件解密"),
            ("外发", "文件外发"),
            ("策略", "策略配置"),
            ("客户端", "客户端"),
            ("服务端", "服务端"),
            ("AD", "AD域同步"),
            ("域", "域配置"),
            ("权限", "权限管理"),
            ("安装", "安装部署"),
            ("端口", "网络配置"),
            ("例外", "例外目录"),
            ("透明加密", "透明加密"),
            ("部署", "安装部署"),
            ("版本", "版本管理"),
            ("配置", "系统配置"),
        ]
        source_text = (section + " " + text[:100]).lower()
        for hint, topic in topic_hints:
            if hint.lower() in source_text:
                keywords.append(topic)
        return "; ".join(keywords[:3]) if keywords else ""
