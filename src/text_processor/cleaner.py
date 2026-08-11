"""
文本清洗器

对解析后的文本进行标准化处理。
"""

import re
from typing import List

from src.document_parser.models import ParsedPage


class TextCleaner:
    """文本清洗器"""

    def __init__(self):
        # 匹配多余空白字符
        self.whitespace_re = re.compile(r"\s+")
        # 匹配空行（连续3个以上换行）
        self.blank_lines_re = re.compile(r"\n{3,}")
        # 波形符/乱码字符
        self.garbage_re = re.compile(r"[^一-鿿　-〿＀-￯a-zA-Z0-9\s\.\,\;\:\!\?\-\+\=\(\)\[\]\{\}\@\#\$\%\^\&\*\/\\\|\<\>\"\'`~＿—…《》「」『』【】（）％＊＋，－．／：；＜＝＞？＠［＼］＾＿｛｜｝～À-ɏ]")

    def clean_page(self, page: ParsedPage) -> ParsedPage:
        """清洗单个页面"""
        text = page.text
        text = self.whitespace_re.sub(" ", text)
        text = self.blank_lines_re.sub("\n\n", text)
        text = text.strip()

        # 去除过短的无意义行
        lines = text.split("\n")
        lines = [
            l for l in lines
            if len(l.strip()) > 1 or l.strip() == ""
        ]

        # 合并连续空行
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            is_empty = line.strip() == ""
            if is_empty and prev_empty:
                continue
            cleaned_lines.append(line)
            prev_empty = is_empty

        return ParsedPage(
            page_num=page.page_num,
            text="\n".join(cleaned_lines),
            section=page.section,
        )

    def clean_pages(self, pages: List[ParsedPage]) -> List[ParsedPage]:
        """清洗所有页面"""
        return [self.clean_page(p) for p in pages if p.text.strip()]
