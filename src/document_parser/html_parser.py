"""
HTML 文档解析器

基于 BeautifulSoup4 提取文本内容。
"""

from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

from src.document_parser.base_parser import BaseDocumentParser
from src.document_parser.models import RawDocument, ParsedPage


class HtmlParser(BaseDocumentParser):
    """HTML 文档解析器"""

    supported_extensions = [".html", ".htm"]
    parser_name = "html"

    def _do_parse(self, raw_doc: RawDocument) -> List[ParsedPage]:
        try:
            with open(raw_doc.file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        except UnicodeDecodeError:
            with open(raw_doc.file_path, "r", encoding="gbk") as f:
                html_content = f.read()

        soup = BeautifulSoup(html_content, "lxml")

        # 移除 script/style 标签
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # 提取标题
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        current_section = title
        all_text = []

        # 提取所有标题和段落
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th"]):
            text = tag.get_text(strip=True)
            if not text:
                continue

            if tag.name.startswith("h"):
                current_section = text

            all_text.append(text)

        # 按章节分组
        sections = self._group_by_sections(all_text)
        pages = [
            ParsedPage(page_num=i + 1, text=t, section=current_section)
            for i, t in enumerate(sections)
        ]

        return pages if pages else [ParsedPage(page_num=1, text="\n".join(all_text), section=title)]

    def _group_by_sections(self, texts: List[str], group_size: int = 50) -> List[str]:
        """将文本按大致行数分组"""
        result = []
        for i in range(0, len(texts), group_size):
            chunk = "\n".join(texts[i:i + group_size])
            if chunk.strip():
                result.append(chunk)
        return result
