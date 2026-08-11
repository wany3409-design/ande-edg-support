"""
纯文本解析器

处理 TXT, Markdown 等纯文本格式。
"""

from typing import List

from src.document_parser.base_parser import BaseDocumentParser
from src.document_parser.models import RawDocument, ParsedPage


class TextParser(BaseDocumentParser):
    """纯文本 / Markdown 解析器"""

    supported_extensions = [".txt", ".md", ".markdown"]
    parser_name = "text"

    def _do_parse(self, raw_doc: RawDocument) -> List[ParsedPage]:
        # 尝试多种编码
        content = None
        for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                with open(raw_doc.file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            raise RuntimeError(f"无法解码文件: {raw_doc.file_path}")

        # 按空行分页
        paragraphs = content.split("\n\n")
        current_section = ""

        # 寻找 Markdown 标题作为章节
        for para in paragraphs[:10]:
            para = para.strip()
            if para.startswith("#"):
                current_section = para.lstrip("#").strip()
                break

        # 按段落数分组（每~50段算一页）
        group_size = 50
        pages = []
        for i in range(0, len(paragraphs), group_size):
            chunk = "\n\n".join(paragraphs[i:i + group_size]).strip()
            if chunk:
                pages.append(ParsedPage(
                    page_num=len(pages) + 1,
                    text=chunk,
                    section=current_section,
                ))

        return pages if pages else [ParsedPage(
            page_num=1, text=content.strip(), section=current_section
        )]
