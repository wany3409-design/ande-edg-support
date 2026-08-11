# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## 项目概述

安得EDG智能技术支持助手 — 基于RAG的企业内部技术支持AI系统。
技术栈: Python 3.13 + FastAPI + Streamlit + ChromaDB + DeepSeek API。

## 常用命令

```bash
# 启动后端
python -m src.api.main

# 启动前端
streamlit run src/ui/app.py

# 初始化知识库
python scripts/init_kb.py

# 运行测试
pytest tests/ -v

# 安装依赖
pip install -r requirements.txt
```

## 项目架构

```
Streamlit (前端 :8501) → FastAPI (后端 :8000) → RAG Pipeline
                                                   ├── ChromaDB (向量检索)
                                                   ├── SQLite (元数据/会话)
                                                   └── DeepSeek API (生成回答)
```

## 关键设计决策

1. **不使用 LangChain 全套** — RAG 核心流程自研，仅用其 TextSplitter
2. **Streamlit 而非 React** — Phase 1 追求开发效率，后续可替换
3. **ChromaDB 嵌入式** — 无 Docker 环境，零配置部署
4. **BGE 中文嵌入模型** — MIT 协议，中文优化，1024维
5. **所有安得产品结论必须来自知识库** — 不能编造内容

## API Key 安全

- `.env` 文件包含 API Key，已在 `.gitignore` 中
- 不修改或泄露 `ANTHROPIC_AUTH_TOKEN`
- 代码中只通过 `config.py` 引用环境变量

## 编码规范

- 所有注释和文档字符串使用中文
- 代码命名使用英文
- 类型注解优先
- 模块化设计，单一职责

## 知识文档位置

`knowledge_docs/` 目录包含安得EDG产品文档：
- 安得卫士（AndSec）产品使用手册.pdf
- 安得电子文档安全管理系统（EDG）产品介绍(1).pdf
- 安得卫士6.0.8技术培训.html
