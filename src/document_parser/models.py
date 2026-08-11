"""
文档解析数据模型

定义文档解析全流程中使用的核心数据结构：
- RawDocument: 原始文档
- ParsedPage: 解析后的单页
- DocumentChunk: 切分后的文本块（含完整来源元数据）
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib


@dataclass
class RawDocument:
    """原始文档"""
    file_path: str
    file_name: str
    file_type: str          # pdf, docx, pptx, html, txt, md
    file_size: int          # bytes
    doc_type: str = "unknown"  # product_manual, product_intro, training, unknown

    def __repr__(self) -> str:
        return f"RawDocument({self.file_name}, type={self.file_type})"


@dataclass
class ParsedPage:
    """解析后的单页内容"""
    page_num: int
    text: str
    section: str = ""       # 所在章节标题


@dataclass
class DocumentChunk:
    """切分后的文档块，含完整来源元数据"""
    chunk_id: str
    text: str

    # 来源信息（溯源性核心）
    source_file: str        # 文件名
    source_path: str        # 文件路径
    source_level: str = "official"  # official / training / inferred

    # 位置信息
    page: int = 1
    section: str = ""

    # 分类信息
    doc_type: str = "unknown"   # product_manual / product_intro / training
    file_type: str = ""         # pdf / docx / pptx / html / txt / md
    topic: str = ""             # 主题分类

    # 切分信息
    chunk_index: int = 0
    total_chunks: int = 0

    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __repr__(self) -> str:
        return (f"Chunk({self.chunk_id[:8]}..., "
                f"source={self.source_file}, "
                f"page={self.page}, section={self.section})")

    def to_dict(self) -> dict:
        """转为字典，用于后续向量存储"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "source_path": self.source_path,
            "source_level": self.source_level,
            "page": self.page,
            "section": self.section,
            "doc_type": self.doc_type,
            "file_type": self.file_type,
            "topic": self.topic,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at,
        }

    def to_metadata(self) -> dict:
        """返回 ChromaDB 兼容的 metadata（不含 text）"""
        d = self.to_dict()
        del d["text"]
        del d["chunk_id"]
        return d


def generate_chunk_id(source_file: str, page: int, chunk_index: int) -> str:
    """生成唯一 chunk ID"""
    raw = f"{source_file}:p{page}:c{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]
