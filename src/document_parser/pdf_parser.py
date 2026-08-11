"""
PDF 文档解析器

基于 PyMuPDF (fitz) 提取文本，支持：
- 逐页提取文本
- 保留页码信息
- 尝试识别章节标题
"""

import re
from typing import List

import pymupdf

from src.document_parser.base_parser import BaseDocumentParser
from src.document_parser.models import RawDocument, ParsedPage


class PDFParser(BaseDocumentParser):
    """PDF 文档解析器"""

    supported_extensions = [".pdf"]
    parser_name = "pdf"

    # 章节标题候选模式
    SECTION_PATTERNS = [
        re.compile(r"^第[一二三四五六七八九十\d]+章\s*.+", re.MULTILINE),
        re.compile(r"^第[一二三四五六七八九十\d]+节\s*.+", re.MULTILINE),
        re.compile(r"^\d+\.\d*\s*[^\d].+", re.MULTILINE),
        re.compile(r"^[一二三四五六七八九十]、\s*.+", re.MULTILINE),
    ]

    def _do_parse(self, raw_doc: RawDocument) -> List[ParsedPage]:
        pages = []
        current_section = ""

        try:
            doc = pymupdf.open(raw_doc.file_path)
        except Exception as e:
            raise RuntimeError(f"无法打开PDF文件: {raw_doc.file_path}, 错误: {e}")

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            if not text or not text.strip():
                continue

            # 尝试识别章节标题
            section_title = self._find_section_title(text)
            if section_title:
                current_section = section_title

            pages.append(ParsedPage(
                page_num=page_num + 1,
                text=text.strip(),
                section=current_section,
            ))

        doc.close()
        return pages

    def _find_section_title(self, text: str) -> str:
        """在文本中查找章节标题"""
        lines = text.strip().split("\n")
        for line in lines[:5]:
            line = line.strip()
            if not line or len(line) > 80:
                continue
            for pattern in self.SECTION_PATTERNS:
                if pattern.match(line):
                    return line
        return ""
