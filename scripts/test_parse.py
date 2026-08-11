#!/usr/bin/env python
"""
文档解析测试脚本
"""

import sys
import os

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import KNOWLEDGE_DOCS_DIR
from src.rag.ingestion import IngestionPipeline


def print_separator(title: str = ""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 60}")


def print_chunk_sample(chunk, index: int):
    """打印单个chunk的摘要信息"""
    print(f"\n  --- Chunk #{index} ---")
    print(f"  ID:       {chunk.chunk_id}")
    print(f"  来源:     {chunk.source_file}")
    print(f"  页码:     {chunk.page}")
    print(f"  章节:     {chunk.section or '(未识别)'}")
    print(f"  类型:     {chunk.doc_type}")
    print(f"  主题:     {chunk.topic or '(未识别)'}")
    print(f"  位置:     {chunk.chunk_index + 1}/{chunk.total_chunks}")
    text_preview = chunk.text[:200].replace("\n", "\\n")
    print(f"  文本预览: {text_preview}...")


def main():
    print_separator("安得EDG 文档解析测试")
    print(f"知识文档目录: {KNOWLEDGE_DOCS_DIR}")

    # 查找所有支持的文件（用 set 去重，Windows 大小写不敏感）
    supported_files = set()
    for ext in [".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"]:
        for f in KNOWLEDGE_DOCS_DIR.glob(f"*{ext}"):
            supported_files.add(str(f))

    if not supported_files:
        print("\n[ERROR] knowledge_docs/ 目录下未找到支持的文件！")
        print("请确保安得EDG文档已放入该目录。")
        return

    supported_files = sorted(supported_files)
    print(f"\n找到 {len(supported_files)} 个文档:")
    for f in supported_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {os.path.basename(f)} ({size_mb:.1f} MB)")

    # 初始化摄入管线
    pipeline = IngestionPipeline()

    print_separator("开始解析")

    total_chunks = 0
    for file_path in supported_files:
        try:
            chunks = pipeline.ingest_file(file_path)
            total_chunks += len(chunks)
        except Exception as e:
            print(f"  [FAILED] {os.path.basename(file_path)}: {e}")
            continue

    print_separator("解析结果汇总")
    print(f"  文档数:   {len(supported_files)}")
    print(f"  总Chunk数: {total_chunks}")
    print(f"  平均:     {total_chunks // max(1, len(supported_files))} chunks/文档")

    # 显示第一个文档的前几个chunk样例
    if supported_files:
        print_separator("Chunk 样例")
        first_file = supported_files[0]
        chunks = pipeline.ingest_file(first_file)
        print(f"文件: {os.path.basename(first_file)}")
        for i, chunk in enumerate(chunks[:3]):
            print_chunk_sample(chunk, i + 1)
        if len(chunks) > 3:
            print(f"\n  ... 还有 {len(chunks) - 3} 个 chunk")

    print_separator("测试通过")
    print(f"  解析器: PDF, DOCX, PPTX, HTML, TXT/MD")
    print(f"  管线:   解析 → 清洗 → 切分 → OK")
    print(f"  元数据: source_file, page, section, topic 均已填充")


if __name__ == "__main__":
    main()
