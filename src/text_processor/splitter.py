"""
文本切分器

将解析后的文档页面切分为适合检索的文本块（chunk）。
使用 LangChain 的 RecursiveCharacterTextSplitter 进行语义感知切分。

v3 改进：
- chunk_size 400 → 保留标题+正文+必要上下文，适配 bge-small max_seq=512
- 增加 POC/测试/部署/验证 等 topic 关键词
- _build_embed_text 保留更长 section 标题
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_parser.models import RawDocument, ParsedPage, DocumentChunk, generate_chunk_id


# v3: 增大 chunk 以保留完整上下文（配置步骤、故障排查流程等）
V3_CHUNK_SIZE = 400
V3_CHUNK_OVERLAP = 60


class TextSplitter:
    """文档文本切分器"""

    def __init__(
        self,
        chunk_size: int = V3_CHUNK_SIZE,
        chunk_overlap: int = V3_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 中文友好的分隔符优先级：优先在段落/章节边界切分
        self._splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",     # 段落
                "\n",       # 换行
                "。",       # 中文句号
                "；",       # 中文分号
                "，",       # 中文逗号
                ". ",       # 英文句号+空格
                "; ",       # 英文分号+空格
                " ",        # 空格
                "",         # 字符级
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(
        self, raw_doc: RawDocument, pages: List[ParsedPage]
    ) -> List[DocumentChunk]:
        """将文档页面切分为 chunk"""
        all_chunks = []

        for page in pages:
            if not page.text.strip():
                continue

            # 嵌入正文：标题 + 正文
            embed_text = self._build_embed_text(page)

            # 切分
            text_parts = self._splitter.split_text(embed_text)

            if not text_parts:
                continue

            total = len(text_parts)
            for i, part in enumerate(text_parts):
                if not part.strip():
                    continue

                chunk_id = generate_chunk_id(raw_doc.file_name, page.page_num, i)
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=part.strip(),
                    source_file=raw_doc.file_name,
                    source_path=raw_doc.file_path,
                    source_level="official",
                    page=page.page_num,
                    section=page.section or "",
                    doc_type=raw_doc.doc_type,
                    file_type=raw_doc.file_type,
                    topic=self._extract_topic(page.section or "", part),
                    chunk_index=i,
                    total_chunks=total,
                )
                all_chunks.append(chunk)

        return all_chunks

    def _build_embed_text(self, page: ParsedPage) -> str:
        """
        构建嵌入正文
        包含章节/幻灯片标题作为上下文线索 + 正文
        """
        if page.section and len(page.section) < 80:
            return f"{page.section}\n{page.text}"
        return page.text

    def _extract_topic(self, section: str, text: str) -> str:
        """从章节或文本中提取主题关键词"""
        keywords = []
        topic_hints = [
            # 加密/解密
            ("加密", "文件加密"),
            ("解密", "文件解密"),
            ("透明加密", "透明加密"),
            ("加解密", "文件加解密"),
            # 外发/权限
            ("外发", "文件外发"),
            ("权限", "权限管理"),
            ("授权", "授权管理"),
            # 策略配置
            ("策略", "策略配置"),
            ("例外", "例外目录"),
            ("配置", "策略配置"),
            # 客户端/服务端
            ("客户端", "客户端"),
            ("服务端", "服务端"),
            ("控制台", "管理控制台"),
            # 部署/安装
            ("部署", "安装部署"),
            ("安装", "安装部署"),
            ("一键安装", "安装部署"),
            ("POC", "POC测试"),
            ("测试", "测试验证"),
            ("验证", "测试验证"),
            ("上线", "上线部署"),
            ("迁移", "系统迁移"),
            ("升级", "版本升级"),
            ("卸载", "卸载"),
            # 域同步/用户
            ("AD域", "AD域同步"),
            ("域同步", "AD域同步"),
            ("LDAP", "AD域同步"),
            ("用户同步", "用户同步"),
            ("用户", "用户管理"),
            # 端口/网络
            ("端口", "网络配置"),
            ("端口映射", "网络配置"),
            ("网络", "网络配置"),
            ("防火墙", "网络配置"),
            # 水印/打印
            ("水印", "水印"),
            ("打印", "打印控制"),
            # 日志/排查
            ("日志", "日志排查"),
            ("排查", "故障排查"),
            ("debug", "日志排查"),
            ("故障", "故障排查"),
            ("问题", "故障排查"),
            # 进程/驱动
            ("进程", "进程控制"),
            ("HOOK", "HOOK注入"),
            ("驱动", "驱动管理"),
            # 介质/设备
            ("介质", "介质管控"),
            ("USB", "介质管控"),
            ("设备", "设备管控"),
            # 邮件/DLP
            ("邮件", "邮件管控"),
            ("DLP", "数据防泄漏"),
            ("数据防泄", "数据防泄漏"),
            ("敏感", "数据防泄漏"),
            # 沙盒
            ("沙盒", "沙盒管控"),
            ("沙箱", "沙盒管控"),
            # 审批/流程
            ("审批", "流程审批"),
            ("流程", "流程审批"),
            ("工单", "流程审批"),
            # 备份
            ("备份", "备份恢复"),
            ("恢复", "备份恢复"),
            # 安全隔离
            ("安全隔离", "安全隔离"),
            ("隔离", "安全隔离"),
        ]

        source_text = (section + " " + text[:300]).lower()
        for hint, topic in topic_hints:
            if hint.lower() in source_text and topic not in keywords:
                keywords.append(topic)

        return "; ".join(keywords[:5]) if keywords else ""


# 兼容旧接口
def create_splitter(
    chunk_size: int = V3_CHUNK_SIZE,
    chunk_overlap: int = V3_CHUNK_OVERLAP,
) -> TextSplitter:
    return TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
