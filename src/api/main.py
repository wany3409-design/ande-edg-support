"""
安得EDG智能技术支持助手 - FastAPI 应用入口
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import API_HOST, API_PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("🚀 安得EDG智能技术支持助手 API 启动中...")
    yield
    # 关闭时
    print("👋 安得EDG智能技术支持助手 API 已关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="安得EDG智能技术支持助手",
        description="基于RAG的安得EDG产品技术支持AI系统",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    from src.api.routes import router as api_router
    app.include_router(api_router, prefix="/api")

    return app


app = create_app()


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "安得EDG智能技术支持助手",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
