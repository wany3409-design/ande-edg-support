# 安得EDG智能技术支持助手

基于RAG（检索增强生成）的安得电子文档安全管理系统（EDG）智能技术支持AI。

## 定位

面向企业售后工程师、内部技术人员、实施人员以及客户IT人员的技术支持工具。

## 核心能力

1. **产品知识问答**：产品功能、加密/解密、策略配置、客户端/服务端、AD域同步等
2. **实施配置指导**：部署、安装、策略配置、POC测试、常见部署问题
3. **故障诊断**：根据现象逐步诊断，不直接猜结论

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Streamlit |
| 后端 | FastAPI (Python) |
| 向量库 | ChromaDB |
| 嵌入模型 | BAAI/bge-large-zh-v1.5 |
| LLM | DeepSeek v4 |
| 文档解析 | PyMuPDF, python-docx, python-pptx, openpyxl |

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/Scripts/activate  # Windows
# 或 source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制环境变量模板
cp .env.example .env
# 编辑 .env 填入 DeepSeek API Key
```

### 3. 初始化知识库

```bash
# 将安得EDG产品文档放入 knowledge_docs/ 目录
# 运行初始化脚本
python scripts/init_kb.py
```

### 4. 启动服务

```bash
# 终端1: 启动 FastAPI 后端
python -m src.api.main

# 终端2: 启动 Streamlit 前端
streamlit run src/ui/app.py
```

### 5. 访问

- 前端: http://localhost:8501
- API 文档: http://localhost:8000/docs

## 项目结构

```
ande-edg-support/
├── data/                  # 数据存储
│   ├── uploads/           # 上传文档暂存
│   ├── chroma_db/         # 向量数据库
│   └── sqlite/            # SQLite 数据库
├── knowledge_docs/        # 知识文档
├── src/                   # 源代码
│   ├── api/               # FastAPI
│   ├── document_parser/   # 文档解析
│   ├── text_processor/    # 文本处理
│   ├── embeddings/        # 嵌入模型
│   ├── vector_store/      # 向量存储
│   ├── retriever/         # 检索引擎
│   ├── llm/               # LLM 调用
│   ├── rag/               # RAG 管线
│   ├── session/           # 会话管理
│   └── ui/                # Streamlit 前端
├── scripts/               # 运维脚本
├── tests/                 # 测试
└── docs/                  # 项目文档
```

## 开发阶段

- [x] Phase 1: 环境与基础项目
- [ ] Phase 2: 文档解析
- [ ] Phase 3: 知识库构建
- [ ] Phase 4: RAG 问答管线
- [ ] Phase 5: FastAPI 后端
- [ ] Phase 6: Streamlit 前端
- [ ] Phase 7: 测试与优化
- [ ] Phase 8: 文档与部署

## License

内部项目 — 安得科技
