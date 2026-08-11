"""
文档解析器基类

所有解析器继承此基类，统一接口。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import os

from src.document_parser.models import RawDocument, ParsedPage


class BaseDocumentParser(ABC):
    """文档解析器抽象基类"""

    # 子类需定义的属性
    supported_extensions: List[str] = []
    parser_name: str = "base"

    def can_parse(self, file_path: str) -> bool:
        """判断是否能解析该文件"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions

    def parse(self, file_path: str) -> List[ParsedPage]:
        """解析文档，返回页面列表"""
        raw_doc = self._load_document(file_path)
        pages = self._do_parse(raw_doc)
        return pages

    def _load_document(self, file_path: str) -> RawDocument:
        """加载文档元数据"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        file_type = ext.lstrip(".")

        # 推断文档类型
        doc_type = self._infer_doc_type(path.name)

        return RawDocument(
            file_path=str(path.absolute()),
            file_name=path.name,
            file_type=file_type,
            file_size=path.stat().st_size,
            doc_type=doc_type,
        )

    def _infer_doc_type(self, filename: str) -> str:
        """根据文件名推断文档类型"""
        name_lower = filename.lower()
        if "手册" in filename or "使用" in filename or "manual" in name_lower:
            return "product_manual"
        elif "介绍" in filename or "intro" in name_lower or "产品" in filename:
            return "product_intro"
        elif "培训" in filename or "training" in name_lower or "教程" in filename:
            return "training"
        elif "安装" in filename or "install" in name_lower:
            return "install_guide"
        elif "管理" in filename or "admin" in name_lower:
            return "admin_manual"
        elif "FAQ" in filename or "常见" in filename:
            return "faq"
        return "unknown"

    @abstractmethod
    def _do_parse(self, raw_doc: RawDocument) -> List[ParsedPage]:
        """子类实现：实际解析逻辑"""
        ...
