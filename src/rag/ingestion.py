"""
文档摄入管线

完整的文档处理流程：
原始文件 → 解析 → 清洗 → 切分 → Chunk列表
（当前阶段不含向量化/Embedding，准备好 chunk 后暂停）
"""

import os
from pathlib import Path
from typing import List, Optional

from src.document_parser.parser_registry import get_registry, ParserRegistry
from src.document_parser.models import DocumentChunk
from src.text_processor.cleaner import TextCleaner
from src.text_processor.splitter import TextSplitter
from src.config import SUPPORTED_EXTENSIONS


class IngestionPipeline:
    """文档摄入管线"""

    def __init__(self):
        self.registry: ParserRegistry = get_registry()
        self.cleaner = TextCleaner()
        self.splitter = TextSplitter()

    def ingest_file(self, file_path: str) -> List[DocumentChunk]:
        """
        处理单个文件：解析 → 清洗 → 切分

        返回: List[DocumentChunk]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件类型: {ext}")

        print(f"  [INGEST] {path.name} ({ext})")

        # Step 1: 解析
        raw_doc = self.registry._parsers[0]._load_document(file_path)
        parser = self.registry.get_parser(file_path)
        if parser is None:
            raise ValueError(f"无可用解析器: {ext}")

        pages = parser.parse(file_path)
        print(f"    -> 解析: {len(pages)} 页")

        # Step 2: 清洗
        pages = self.cleaner.clean_pages(pages)
        print(f"    -> 清洗: {len(pages)} 有效页")

        # Step 3: 切分
        chunks = self.splitter.split(raw_doc, pages)
        print(f"    -> 切分: {len(chunks)} chunks")

        return chunks

    def ingest_directory(self, dir_path: str) -> List[DocumentChunk]:
        """
        批量处理目录下的所有支持文件

        返回: List[DocumentChunk] (所有文件的chunk合并)
        """
        all_chunks = []
        supported = self.registry.get_supported_extensions()

        for root, dirs, files in os.walk(dir_path):
            for fname in sorted(files):
                ext = Path(fname).suffix.lower()
                if ext in supported:
                    fpath = os.path.join(root, fname)
                    try:
                        chunks = self.ingest_file(fpath)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        print(f"  [ERROR] {fname}: {e}")
                        continue

        return all_chunks

    def ingest_files(self, file_paths: List[str]) -> List[DocumentChunk]:
        """批量处理指定文件列表"""
        all_chunks = []
        for fp in file_paths:
            try:
                chunks = self.ingest_file(fp)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  [ERROR] {fp}: {e}")
                continue
        return all_chunks
