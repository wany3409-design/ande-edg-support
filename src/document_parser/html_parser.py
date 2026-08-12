"""
HTML 文档解析器

基于 BeautifulSoup4 提取文本内容。
v2 改进：
- 按 <section data-title="..."> 幻灯片结构分组，保留标题
- 表格内容作为结构化文本保留行列关系
- 图片 alt 文本纳入提取
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

        # 移除不需要的标签
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # 提取文档标题
        doc_title = ""
        title_tag = soup.find("title")
        if title_tag:
            doc_title = title_tag.get_text(strip=True)

        pages = []

        # 优先使用 <section class="slide"> 结构（幻灯片型文档）
        slides = soup.find_all("section", class_="slide")
        if slides:
            for i, slide in enumerate(slides, 1):
                slide_title = slide.get("data-title", "")
                if not slide_title:
                    # 尝试从 h1/h2 提取标题
                    h_tag = slide.find(["h1", "h2"])
                    if h_tag:
                        slide_title = h_tag.get_text(strip=True)

                text = self._extract_slide_text(slide)
                if text.strip():
                    pages.append(ParsedPage(
                        page_num=i,
                        text=text.strip(),
                        section=slide_title or doc_title,
                    ))
        else:
            # 回退：普通 HTML 页面
            body = soup.find("body") or soup
            text = self._extract_body_text(body)
            pages.append(ParsedPage(
                page_num=1,
                text=text.strip(),
                section=doc_title,
            ))

        return pages if pages else [
            ParsedPage(page_num=1, text=soup.get_text(separator="\n", strip=True), section=doc_title)
        ]

    def _extract_slide_text(self, slide) -> str:
        """从单张幻灯片提取结构化文本"""
        parts = []

        # 1. 标题
        for h_tag in slide.find_all(["h1", "h2", "h3"], limit=1):
            title = h_tag.get_text(strip=True)
            if title:
                parts.append(title)

        # 2. 列表
        for ul in slide.find_all("ul"):
            items = []
            for li in ul.find_all("li"):
                text = li.get_text(strip=True)
                if text:
                    items.append(f"- {text}")
            if items:
                parts.extend(items)

        # 3. 表格 - 保留结构
        for table in slide.find_all("table"):
            table_text = self._extract_table(table)
            if table_text:
                parts.append(table_text)

        # 4. 段落文本（不含列表和表格内的）
        for tag_name in ["p", "div"]:
            for tag in slide.find_all(tag_name, recursive=True):
                # 跳过已在列表/表格内的
                if tag.find_parent(["li", "td", "th", "table"]):
                    continue
                # 跳过只包含其他块级元素的 div
                if tag_name == "div" and tag.find(["ul", "ol", "table", "section"]):
                    continue
                text = tag.get_text(strip=True)
                if text and len(text) > 3:  # 过滤太短的
                    # 避免重复（子孙元素可能已包含）
                    parts.append(text)

        # 5. 非列表/表格中的其他块级文本
        for tag in slide.find_all(["h4", "h5", "h6"]):
            text = tag.get_text(strip=True)
            if text and len(text) > 2:
                parts.append(text)

        # 去重但保留顺序
        seen = set()
        unique = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        return "\n".join(unique)

    def _extract_table(self, table) -> str:
        """提取表格为可读文本"""
        rows = []
        headers = []
        thead = table.find("thead")
        if thead:
            for th in thead.find_all("th"):
                headers.append(th.get_text(strip=True))

        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)

        if not rows:
            return ""

        lines = []
        if headers:
            lines.append(" | ".join(headers))
            lines.append("-" * len(lines[0]))

        for row in rows:
            # 补齐到 header 长度
            while len(row) < len(headers) and headers:
                row.append("")
            lines.append(" | ".join(row))

        return "\n".join(lines)

    def _extract_body_text(self, body) -> str:
        """回退方案：提取普通页面文本"""
        parts = []

        for tag in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
            if tag.name == "table":
                table_text = self._extract_table(tag)
                if table_text:
                    parts.append(table_text)
            else:
                text = tag.get_text(strip=True)
                if text:
                    parts.append(text)

        return "\n".join(parts)
