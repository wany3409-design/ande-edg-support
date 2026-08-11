"""
解析器注册中心

根据文件扩展名路由到对应的解析器。
"""

from pathlib import Path
from typing import List, Optional

from src.document_parser.base_parser import BaseDocumentParser
from src.document_parser.pdf_parser import PDFParser
from src.document_parser.docx_parser import DocxParser
from src.document_parser.pptx_parser import PptxParser
from src.document_parser.html_parser import HtmlParser
from src.document_parser.text_parser import TextParser
from src.document_parser.models import ParsedPage


class ParserRegistry:
    """解析器注册中心"""

    def __init__(self):
        self._parsers: List[BaseDocumentParser] = []
        self._register_defaults()

    def _register_defaults(self):
        """注册默认解析器"""
        self.register(PDFParser())
        self.register(DocxParser())
        self.register(PptxParser())
        self.register(HtmlParser())
        self.register(TextParser())

    def register(self, parser: BaseDocumentParser):
        """注册一个解析器"""
        self._parsers.append(parser)

    def get_parser(self, file_path: str) -> Optional[BaseDocumentParser]:
        """获取能处理该文件的解析器"""
        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def parse(self, file_path: str) -> List[ParsedPage]:
        """自动选择合适的解析器并解析"""
        parser = self.get_parser(file_path)
        if parser is None:
            ext = Path(file_path).suffix
            raise ValueError(f"不支持的文件类型: {ext}")

        return parser.parse(file_path)

    def get_supported_extensions(self) -> List[str]:
        """返回所有支持的文件扩展名"""
        exts = []
        for parser in self._parsers:
            exts.extend(parser.supported_extensions)
        return sorted(set(exts))

    def get_parser_info(self) -> dict:
        """返回解析器信息"""
        return {
            p.parser_name: {
                "extensions": p.supported_extensions,
                "class": p.__class__.__name__,
            }
            for p in self._parsers
        }


# 全局单例
_registry: Optional[ParserRegistry] = None


def get_registry() -> ParserRegistry:
    """获取全局解析器注册中心"""
    global _registry
    if _registry is None:
        _registry = ParserRegistry()
    return _registry
