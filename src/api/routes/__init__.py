from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def api_root():
    """API 根路径"""
    return {
        "message": "安得EDG智能技术支持助手 API",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "upload": "/api/upload",
            "chat": "/api/chat",
            "knowledge": "/api/knowledge",
            "sessions": "/api/sessions",
        },
    }
