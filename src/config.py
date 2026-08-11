"""
安得EDG智能技术支持助手 - 配置管理
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# ========== DeepSeek API 配置 ==========
DEEPSEEK_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
DEEPSEEK_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
DEEPSEEK_MODEL = os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro")

# ========== 嵌入模型配置 ==========
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ========== ChromaDB 配置 ==========
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))

# ========== SQLite 配置 ==========
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(PROJECT_ROOT / "data" / "sqlite" / "ande_edg.db"))

# ========== 文本切分配置 ==========
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ========== 检索配置 ==========
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_TOP_K_CANDIDATE = int(os.getenv("RETRIEVAL_TOP_K_CANDIDATE", "10"))

# ========== 服务配置 ==========
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "8501"))

# ========== 知识文档目录 ==========
KNOWLEDGE_DOCS_DIR = PROJECT_ROOT / "knowledge_docs"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"

# ========== 支持的文档格式 ==========
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".doc": "DOC",
    ".pptx": "PPTX",
    ".xlsx": "XLSX",
    ".xls": "XLS",
    ".html": "HTML",
    ".htm": "HTML",
    ".txt": "TXT",
    ".md": "Markdown",
}

# ========== Collection 名称 ==========
CHROMA_COLLECTION_NAME = "ande_edg_knowledge"

# ========== System Prompt ==========
SYSTEM_PROMPT = """你是安得EDG（安得电子文档安全管理系统）的专业技术支持助手。

## 你的角色
- 你服务于企业售后工程师、内部技术人员、实施人员以及客户IT人员
- 你基于安得EDG官方产品文档提供技术支持

## 核心能力
1. 产品知识问答：功能、配置、文件加密/解密、透明加密、文件外发、策略配置、客户端、服务端、AD域同步、版本及部署等
2. 实施配置指导：服务端部署、客户端安装、策略配置、AD域同步、端口配置、POC测试、常见部署问题等
3. 故障诊断：根据现象逐步排查，不直接猜结论

## 回答原则
1. 所有安得EDG产品相关的具体结论，必须优先基于提供的知识库资料
2. 如果知识库资料无法支持某个结论，要明确说："当前知识库资料不足，无法确认该结论"
3. 可以提供排查建议，但必须明确标注这是"推测/排查建议"，而不是安得官方结论
4. 回答要专业、简洁、可执行，尽量给出具体操作步骤
5. 不使用大量空泛的AI套话，不为了显得专业而编造内容

## 故障问题回答格式
对于故障类问题，优先采用以下格式：
【问题判断】
【可能原因】
【排查步骤】
【解决方案】
【需要补充的信息】

必要时主动询问：操作系统、客户端版本、服务端版本、文件类型、用户/设备环境、策略配置、报错截图、日志等。

## 知识问答格式
根据问题自然回答，给出具体操作步骤即可。

## 引用要求
在回答末尾标注参考的知识来源。"""


def validate_config() -> bool:
    """验证必要的配置项是否完整"""
    issues = []

    if not DEEPSEEK_AUTH_TOKEN:
        issues.append("❌ ANTHROPIC_AUTH_TOKEN 未配置")

    if issues:
        print("配置检查发现问题：")
        for issue in issues:
            print(f"  {issue}")
        return False

    print("✅ 配置检查通过")
    return True
